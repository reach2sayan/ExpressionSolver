#include "tests_common.hpp"


// ===========================================================================
// The public hessian() router on a raw callable.  It has one answer — the scalar
// O(m^2) probe driver — and this pins that across a spread of m, on an energy
// exercising +,-,*,/,log,exp and scalar*dual.
// ===========================================================================
namespace {
// Non-trivial multivariate function exercising +,-,*,/,log,exp and
// scalar*dual on the forward-dual element type.
template <Numeric T> T vf_sample(const T *y, std::size_t n) {
  using std::exp, std::log; // ADL selects the dual overloads
  T g = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    g = g + y[i] * log(y[i]);
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double c =
          0.1 * static_cast<double>(i + 1) - 0.05 * static_cast<double>(j);
      g = g + c * (y[i] * y[j]) / (T{1} + y[i]);
    }
  }
  g = g + exp(y[0] * y[n - 1]);
  return g;
}
} // namespace

TEST(HessianRouter, RawCallableTakesTheScalarDriver) {
  for (std::size_t n : {std::size_t{2}, std::size_t{5}, std::size_t{9},
                        std::size_t{12}, std::size_t{20}, std::size_t{40}}) {
    std::vector<double> x(n);
    for (std::size_t k = 0; k < n; ++k) {
      x[k] = 0.15 + 0.6 * (k + 1.0) / (n + 1.0);
    }
    auto f = [n](const auto *dof) { return vf_sample(dof, n); };
    const std::span<const double> xs{x.data(), x.size()};

    const auto Hs = diff::impl::detail::hessian(f, xs);
    const auto Hv = diff::impl::hessian(f, xs);

    ASSERT_EQ(hess_n(Hs), hess_n(Hv));
    EXPECT_NEAR(val_of(Hs), val_of(Hv), 1e-9) << "n=" << n;
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_NEAR(grad_at(Hs, i), grad_at(Hv, i), 1e-9)
          << "grad i=" << i << " n=" << n;
    }
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        EXPECT_NEAR(hess_at(Hs, i, j), hess_at(Hv, i, j), 1e-7)
            << "H(" << i << "," << j << ") n=" << n;
      }
    }
  }
}

// A compile-time expression *graph* can be handed straight to the public
// hessian(): the router detects CExpression, auto-bridges it via seeded_energy
// (no client wrapping), and routes to the scalar driver.  The result must be
// bit-close to both explicit drivers on the same graph.
TEST(SeededExprEnergy, GraphRoutesThroughPublicHessian) {
  using D = diff::impl::Dual<double>;
  using diff::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  Variable<D, FixedString{"x03"}> d;
  auto expr = a * log(a) + b * log(b) + c * log(c) + d * log(d) +
              0.50 * (a - b) * (a - b) + 0.51 * (b - c) * (b - c) +
              0.52 * (c - d) * (c - d) + exp(a * d);

  const std::array<double, 4> x{0.2, 0.4, 0.6, 0.8};
  const std::span<const double> xs{x.data(), x.size()};

  // Client just calls hessian() with the raw graph — no seeded_energy in sight.
  const auto Hrouted = diff::impl::hessian(expr, xs);

  // Explicit bridge for cross-checking against both numeric drivers.
  auto f = diff::impl::seeded_energy(expr);
  static_assert(diff::impl::CSeededExprEnergy<decltype(f)>,
                "seeded_energy() must advertise the routing tag");
  static_assert(decltype(f)::arity == 4, "arity deduced from symbol set");
  const auto Hscalar = diff::impl::detail::hessian(f, xs);

  EXPECT_NEAR(val_of(Hrouted), val_of(Hscalar), 1e-12);
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(grad_at(Hrouted, i), grad_at(Hscalar, i), 1e-12) << "grad " << i;
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_NEAR(hess_at(Hrouted, i, j), hess_at(Hscalar, i, j), 1e-12)
          << "scalar H(" << i << "," << j << ")";
    }
  }
}

// Forward-over-reverse is the third driver for the same Hessian, and the only
// one that is O(N) sweeps rather than O(N^2) probes.  Before the router may
// prefer it, it has to agree with the forward-over-forward driver on a real
// graph energy — not just the two-variable cases in HessianTest.
TEST(SeededExprEnergy, ForwardOverReverseAgreesWithNumericDrivers) {
  using D = diff::impl::Dual<double>;
  using diff::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  Variable<D, FixedString{"x03"}> d;
  auto expr = a * log(a) + b * log(b) + c * log(c) + d * log(d) +
              0.50 * (a - b) * (a - b) + 0.51 * (b - c) * (b - c) +
              0.52 * (c - d) * (c - d) + exp(a * d);

  const std::array<double, 4> x{0.2, 0.4, 0.6, 0.8};
  const std::span<const double> xs{x.data(), x.size()};

  auto f = diff::impl::seeded_energy(expr);
  const auto Hscalar = diff::impl::detail::hessian(f, xs);
  const auto Hrev = Equation{expr}.hessian(x);

  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_NEAR(Hrev[i][j], hess_at(Hscalar, i, j), 1e-9)
          << "scalar H(" << i << "," << j << ")";
    }
  }
}

// ===========================================================================
// Memory-ownership contract.
//
// Every result the library hands back is owned by the caller: the drivers
// return a HessianResult whose vectors are its own, and the symbolic API
// returns std::array by value.  The one raw pointer in the design points the
// other way — the driver lends the energy callable a `const Dof *` that is
// valid only for the call and carries no length.  These tests pin down both
// halves: the borrow is bounds-checked at the router, and the accessors that
// alias member storage stay usable on temporaries by copying out.
// ===========================================================================

TEST(Ownership, GraphHessianRejectsAPointShorterThanTheSymbolSet) {
  using D = diff::impl::Dual<double>;
  using diff::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  auto expr = a * b + exp(c);

  // Two values for a three-symbol graph.  seeded_energy() reads slots [0,3),
  // so without the guard this reads off the end of the driver's dof vector.
  const std::array<double, 2> shortx{0.2, 0.4};
  EXPECT_THROW(diff::impl::hessian(expr, std::span<const double>{shortx}),
               std::out_of_range);

  // Surplus values are just as wrong: the extra rows would come back zero.
  const std::array<double, 4> longx{0.2, 0.4, 0.6, 0.8};
  EXPECT_THROW(diff::impl::hessian(expr, std::span<const double>{longx}),
               std::out_of_range);

  // The exact point is accepted.
  const std::array<double, 3> okx{0.2, 0.4, 0.6};
  EXPECT_NO_THROW(diff::impl::hessian(expr, std::span<const double>{okx}));
}

TEST(Ownership, GraphHessianRejectsAnActiveIndexThatNamesNoSymbol) {
  using D = diff::impl::Dual<double>;
  using diff::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  auto expr = a * b;

  const std::array<double, 4> x{0.2, 0.4, 0.6, 0.8};
  const std::array<std::size_t, 2> bad{2, 3}; // no such symbols
  EXPECT_THROW(diff::impl::hessian(expr, std::span<const double>{x},
                             std::span<const std::size_t>{bad}),
               std::out_of_range);

  const std::array<std::size_t, 1> ok{1};
  EXPECT_NO_THROW(diff::impl::hessian(expr, std::span<const double>{x},
                                std::span<const std::size_t>{ok}));
}

TEST(Ownership, ResultOwnsItsBuffersAndTransfersThem) {
  // The drivers hand back plain owning std types, so ownership is the tuple's:
  // the buffers move with it and die with it.  There is no accessor, so an
  // accessor on a prvalue result cannot dangle.
  auto f = [](const auto *d) { return d[0] * d[0] * d[1]; };
  const std::array<double, 2> x{2.0, 3.0};

  auto H = diff::impl::hessian(f, std::span<const double>{x});
  EXPECT_DOUBLE_EQ(hess_at(H, 0, 1), 4.0); // d2/dx0dx1 of x0^2 x1 = 2 x0

  // Moving the result moves the buffers: the destination is intact and the
  // source has released them, which is what "the caller owns this" means.
  const double *const before = hess_ptr(H);
  auto moved = std::move(H);
  EXPECT_EQ(hess_ptr(moved), before) << "move must not copy the buffer";
  EXPECT_DOUBLE_EQ(hess_at(moved, 0, 1), 4.0);
  EXPECT_EQ(std::get<2>(H).get(), nullptr) << "moved-from must have released";

  // The extent travels with the buffers rather than being recoverable from
  // them -- a unique_ptr<double[]> does not know its own length.
  EXPECT_EQ(hess_n(moved), 2u);

  // Same for the value maps: lvalue borrows, rvalue copies.
  const auto m = values(named<"x">(2.0), named<"y">(3.0));
  static_assert(std::is_same_v<decltype(m.get<"x">()), const double &>,
                "get() on an lvalue map borrows");
  static_assert(
      std::is_same_v<decltype(values(named<"x">(2.0)).get<"x">()), double>,
      "get() on a temporary map must return by value");
  EXPECT_DOUBLE_EQ(values(named<"x">(2.0)).get<"x">(), 2.0);
  EXPECT_DOUBLE_EQ(bind(PV(0.0, "x") * PV(0.0, "y"), named<"x">(4.0),
                        named<"y">(5.0))
                       .get<"x">(),
                   4.0);
}

TEST(Ownership, EquationSubtreeAccessorsWorkOnTemporaries) {
  // The nodes are empty types now, so the rvalue overload copies nothing.
  auto x = PV(0.0, "x");
  auto y = PV(0.0, "y");
  EXPECT_DOUBLE_EQ(Equation(x * y).get<1>().eval(2.0), 2.0);
  EXPECT_DOUBLE_EQ(Equation(x * y)[idx<2>()].eval(4.0), 4.0);
}

TEST(Ownership, ReverseHessianAcceptsATemporaryExpression) {
  // A temporary expression has to reach hessian(), not just gradient().
  using D = diff::impl::Dual<double>;
  const auto H = Equation{PDV(0.0, "x") * PDV(0.0, "y")}.hessian(std::array{2.0, 3.0});
  const auto g = Equation{PDV(0.0, "x") * PDV(0.0, "y")}.gradient(std::array{D{2.0}, D{3.0}});
  EXPECT_DOUBLE_EQ(H[0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[0][0], 0.0);
  EXPECT_DOUBLE_EQ(g[0], 3.0);
}

// ===========================================================================
// Compile-time Hessian sparsity (coupling.hpp).  The pattern drives which
// entries the compressed driver writes, so a pattern that WRONGLY drops a pair
// silently zeroes a real Hessian entry — these pin the derivation down.
// ===========================================================================

TEST(HessianCoupling, ChainPatternIsTridiagonalPlusCorner) {
  using D = diff::impl::Dual<double>;
  using diff::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  Variable<D, FixedString{"x03"}> d;
  auto expr = a * log(a) + b * log(b) + c * log(c) + d * log(d) +
              0.50 * (a - b) * (a - b) + 0.51 * (b - c) * (b - c) +
              0.52 * (c - d) * (c - d) + exp(a * d);

  constexpr auto P = diff::impl::hessian_pattern<decltype(expr)>();
  using Row = diff::impl::symbol_set<4>;
  // Tridiagonal from the (xk - xk+1)^2 terms and the xk*log(xk) diagonal, plus
  // the (0, n-1) corner from exp(x0 * x3).
  static_assert(P[0] == Row{0b1011}, "row 0: self, neighbour, corner");
  static_assert(P[1] == Row{0b0111}, "row 1: tridiagonal");
  static_assert(P[2] == Row{0b1110}, "row 2: tridiagonal");
  static_assert(P[3] == Row{0b1101}, "row 3: self, neighbour, corner");

  // The literals 0.50.. are Constants and must contribute no coupling.
  const std::array<double, 4> x{0.2, 0.4, 0.6, 0.8};
  const std::span<const double> xs{x.data(), x.size()};
  const auto Hscalar =
      diff::impl::detail::hessian(diff::impl::seeded_energy(expr), xs);
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      const bool predicted = P[i][j];
      if (!predicted) {
        EXPECT_NEAR(hess_at(Hscalar, i, j), 0.0, 1e-12)
            << "pattern says (" << i << "," << j << ") is structurally zero";
      }
    }
  }
}

TEST(HessianCoupling, LinearExpressionHasEmptyPattern) {
  using D = diff::impl::Dual<double>;
  using diff::impl::FixedString;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto expr = 3.0 * x + 4.0 * y - x;
  constexpr auto P = diff::impl::hessian_pattern<decltype(expr)>();
  static_assert(P[0].none() && P[1].none(), "a linear form has a zero Hessian");
  // One colour, and every entry stays zero.
  constexpr auto C = diff::impl::color_columns<2>(P);
  static_assert(C.count == 1, "nothing conflicts, so one sweep suffices");
}

TEST(HessianCoupling, DenseExpressionDegradesToOneSweepPerColumn) {
  using D = diff::impl::Dual<double>;
  using diff::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  auto expr = exp(a * b * c);
  constexpr auto P = diff::impl::hessian_pattern<decltype(expr)>();
  static_assert(P[0].all() && P[1].all() && P[2].all(),
                "a fully coupled expression must not be pruned");
  constexpr auto C = diff::impl::color_columns<3>(P);
  static_assert(C.count == 3, "dense pattern gives one colour per column");
}

// A division-heavy energy exercises the quotient rule branch of the coupling
// pass, where curvature appears within the denominator but not the numerator.
TEST(HessianCoupling, CompressedDriverMatchesProbeDriverOnQuotients) {
  using D = diff::impl::Dual<double>;
  using diff::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  auto expr = a / b + b / c + a * c / (1.0 + b) + log(a * b * c);

  const std::array<double, 3> x{0.7, 1.3, 2.1};
  const std::span<const double> xs{x.data(), x.size()};

  const auto Hcompressed = diff::impl::hessian(expr, xs); // routed: compressed
  const auto Hprobe =
      diff::impl::detail::hessian(diff::impl::seeded_energy(expr), xs);

  EXPECT_NEAR(val_of(Hcompressed), val_of(Hprobe), 1e-12);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(grad_at(Hcompressed, i), grad_at(Hprobe, i), 1e-9) << i;
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_NEAR(hess_at(Hcompressed, i, j), hess_at(Hprobe, i, j), 1e-9)
          << "H(" << i << "," << j << ")";
    }
  }
}

// Trig and mixed products: a shape the chain energy does not cover, checked
// against the driver that makes no structural assumption at all.
TEST(HessianCoupling, CompressedDriverMatchesProbeDriverOnTrigProducts) {
  using D = diff::impl::Dual<double>;
  using diff::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  Variable<D, FixedString{"x03"}> d;
  auto expr = sin(a) * cos(b) + exp(c * d) + a * b * c + sqrt(d) + a * a;

  const std::array<double, 4> x{0.3, 0.6, 0.9, 1.2};
  const std::span<const double> xs{x.data(), x.size()};

  const auto Hc = diff::impl::hessian(expr, xs);
  const auto Hp = diff::impl::detail::hessian(diff::impl::seeded_energy(expr), xs);
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_NEAR(hess_at(Hc, i, j), hess_at(Hp, i, j), 1e-9)
          << "H(" << i << "," << j << ")";
    }
  }
}

// ===========================================================================
// Sparse Hessian.  The sparsity is a property of the expression TYPE, so the
// path writes only the entries the compile-time pattern predicts: a pattern
// that is wrong shows up here as a missing entry rather than as a wrong number.
// These compare it against the structure-blind dense driver.
//
// There is no linear-algebra library on this boundary.  What the library hands
// over is a compressed-column triple (outer, inner, values) plus the extent,
// and `densify` below is the whole of what a caller does with it -- it stands
// in for the one-line Eigen::Map a client would write instead.
// ===========================================================================
namespace {
// CSC triple -> dense row-major, exactly as a consumer would read it.
template <typename Sparse>
  requires requires { Sparse::rows; Sparse::outer(); Sparse::inner(); }
std::vector<double> densify(const Sparse &h) {
  std::vector<double> dense(Sparse::rows * Sparse::rows, 0.0);
  const auto outer = Sparse::outer();
  const auto inner = Sparse::inner();
  const auto values = h.values();
  for (std::size_t j = 0; j < Sparse::rows; ++j) {
    for (auto k = static_cast<std::size_t>(outer[j]);
         k < static_cast<std::size_t>(outer[j + 1]); ++k) {
      dense[static_cast<std::size_t>(inner[k]) * Sparse::rows + j] = values[k];
    }
  }
  return dense;
}
} // namespace

// The dense Hessian buffer is row-major with the extent alongside it, and that
// layout is the contract a caller maps its own matrix type onto.  Pinning it
// here is what makes `H[i * n + j]` safe to write in client code.
TEST(SparseHessian, DenseBufferIsRowMajorWithItsExtent) {
  using D = diff::impl::Dual<double>;
  using diff::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  // Deliberately asymmetric in the SOURCE so a transposed read would be caught
  // if the driver did not symmetrise: d2/da db of a*a*b is 2a, of b*b*a is 2b.
  auto expr = a * log(a) + b * log(b) + 0.5 * (a - b) * (a - b);

  const std::array<double, 2> x{0.3, 0.7};
  const std::span<const double> xs{x.data(), x.size()};
  const auto H = diff::impl::hessian(expr, xs);

  const double *const h = hess_ptr(H);
  const std::size_t n = hess_n(H);
  ASSERT_EQ(n, 2u);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      EXPECT_DOUBLE_EQ(h[i * n + j], hess_at(H, i, j))
          << "row-major H(" << i << "," << j << ")";
    }
  }
}

TEST(SparseHessian, MatchesDenseDriver) {
  using D = diff::impl::Dual<double>;
  using diff::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  Variable<D, FixedString{"x03"}> d;
  auto expr = a * log(a) + b * log(b) + c * log(c) + d * log(d) +
              0.50 * (a - b) * (a - b) + 0.51 * (b - c) * (b - c) +
              0.52 * (c - d) * (c - d) + exp(a * d);

  const std::array<double, 4> x{0.2, 0.4, 0.6, 0.8};
  const std::span<const double> xs{x.data(), x.size()};

  const auto dense = diff::impl::detail::hessian(diff::impl::seeded_energy(expr), xs);
  const auto sparse = diff::impl::sparse_hessian(expr, xs);

  // Tridiagonal plus the (0,3) corner: 4 + 2*3 + 2 = 12 nonzeros, not 16.
  static_assert(decltype(sparse)::nnz == 12,
                "chain energy stores 12 of 16 entries");
  EXPECT_EQ(sparse.values().size(), 12u);
  // The CSC triple is self-consistent: outer ends at nnz, inner has that many.
  EXPECT_EQ(decltype(sparse)::outer().back(), 12);
  EXPECT_EQ(decltype(sparse)::inner().size(), 12u);

  const auto M = densify(sparse);
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_NEAR(M[i * 4 + j], hess_at(dense, i, j), 1e-9)
          << "H(" << i << "," << j << ")";
    }
  }
  // Everything the pattern excludes must be an exact structural zero.
  EXPECT_DOUBLE_EQ(M[0 * 4 + 2], 0.0);
  EXPECT_DOUBLE_EQ(M[1 * 4 + 3], 0.0);
}

TEST(SparseHessian, IndexesLikeADenseMatrix) {
  using D = diff::impl::Dual<double>;
  using diff::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  // a and c never meet: (0,2) and (2,0) are structural zeros.
  auto expr = a * b + b * c + log(a) + exp(c);

  const std::array<double, 3> x{0.7, 1.3, 2.1};
  const std::span<const double> xs{x.data(), x.size()};

  const auto sparse = diff::impl::sparse_hessian(expr, xs);
  const auto dense = diff::impl::detail::hessian(diff::impl::seeded_energy(expr), xs);

  // Indexed as if dense, while storing only the nonzeros.
  EXPECT_LT(decltype(sparse)::nnz, 9u);
  EXPECT_EQ(sparse.values().size(), decltype(sparse)::nnz);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_NEAR((sparse[i, j]), hess_at(dense, i, j), 1e-9)
          << "H(" << i << "," << j << ")";
    }
  }

  // Structural zeros read as exact zero through the shared sink cell.
  EXPECT_FALSE(decltype(sparse)::structural(0, 2));
  EXPECT_FALSE(decltype(sparse)::structural(2, 0));
  EXPECT_DOUBLE_EQ((sparse[0, 2]), 0.0);
  EXPECT_DOUBLE_EQ((sparse[2, 0]), 0.0);
  EXPECT_TRUE(decltype(sparse)::structural(0, 1));
}

TEST(SparseHessian, IsSymmetricAndSortedWithinEachColumn) {
  using D = diff::impl::Dual<double>;
  using diff::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  auto expr = a / b + b / c + log(a * b * c) + exp(a * c);

  const std::array<double, 3> x{0.7, 1.3, 2.1};
  const std::span<const double> xs{x.data(), x.size()};

  const auto sparse = diff::impl::sparse_hessian(expr, xs);
  const auto dense = diff::impl::detail::hessian(diff::impl::seeded_energy(expr), xs);
  const auto M = densify(sparse);

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_NEAR(M[i * 3 + j], hess_at(dense, i, j), 1e-9);
    }
  }
  // No symmetrization pass runs on the sparse path, so symmetry has to come out
  // of the sweeps themselves.
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_NEAR(M[i * 3 + j], M[j * 3 + i], 1e-9);
    }
  }
  // Compressed and sorted: every column's row indices strictly ascend, which is
  // what a CSC consumer is entitled to assume without a re-sort.
  const auto outer = decltype(sparse)::outer();
  const auto inner = decltype(sparse)::inner();
  for (std::size_t j = 0; j < 3; ++j) {
    for (auto k = static_cast<std::size_t>(outer[j]) + 1;
         k < static_cast<std::size_t>(outer[j + 1]); ++k) {
      EXPECT_LT(inner[k - 1], inner[k]) << "column " << j << " is unsorted";
    }
  }
}

// The degenerate end of the storage range.  A linear expression has a
// structurally empty Hessian, so nnz is 0 and the buffer collapses to nothing
// but the sink cell — the one place an off-by-one in the sizing would show up
// as a zero-length array rather than a wrong answer.
TEST(SparseHessian, OfALinearExpressionIsEmpty) {
  using D = diff::impl::Dual<double>;
  using diff::impl::FixedString;
  Variable<D, FixedString{"x00"}> x;
  Variable<D, FixedString{"x01"}> y;
  auto expr = 3.0 * x + 4.0 * y - x;

  const std::array<double, 2> pt{0.7, 1.3};
  const std::span<const double> xs{pt.data(), pt.size()};

  const auto sparse = diff::impl::sparse_hessian(expr, xs);
  static_assert(decltype(sparse)::nnz == 0,
                "a linear expression couples nothing, so it stores nothing");
  EXPECT_EQ(sparse.values().size(), 0u);
  EXPECT_EQ(decltype(sparse)::outer().back(), 0);

  // Every entry is a structural zero, and each one still reads back as exactly
  // 0.0 through the shared sink cell rather than as garbage.
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      EXPECT_FALSE(decltype(sparse)::structural(i, j));
      EXPECT_DOUBLE_EQ((sparse[i, j]), 0.0);
    }
  }
}


// A plain arithmetic energy lambda carries no tag and is not a CExpression, so
// it must keep routing to the raw-callable branch — the expr-graph path is
// auto-detected, never forced onto a lambda that merely looks numeric.
TEST(SeededExprEnergy, RawLambdaIsNotMistakenForAGraph) {
  auto f = [](const auto *y) {
    using std::log;
    return y[0] * log(y[0]) + y[1] * log(y[1]);
  };
  static_assert(!diff::impl::CSeededExprEnergy<decltype(f)>,
                "untagged lambda must not be treated as expr-template energy");
  static_assert(!diff::impl::CExpression<decltype(f)>,
                "a lambda is not an expression graph");
}

TEST(HessianRouter, IdealMixingClosedForm) {
  constexpr double R = 8.31446261815324, T = 1000.0;
  auto f = [](const auto *y) {
    using std::log;
    return R * T * (y[0] * log(y[0]) + y[1] * log(y[1]));
  };
  const std::array<double, 2> y{0.3, 0.7};
  const auto H =
      diff::impl::hessian(f, std::span<const double>{y.data(), y.size()});
  EXPECT_NEAR(val_of(H), R * T * (0.3 * std::log(0.3) + 0.7 * std::log(0.7)),
              1e-6);
  EXPECT_NEAR(grad_at(H, 0), R * T * (std::log(0.3) + 1.0), 1e-6);
  EXPECT_NEAR(grad_at(H, 1), R * T * (std::log(0.7) + 1.0), 1e-6);
  EXPECT_NEAR(hess_at(H, 0, 0), R * T / 0.3, 1e-3);
  EXPECT_NEAR(hess_at(H, 1, 1), R * T / 0.7, 1e-3);
  EXPECT_NEAR(hess_at(H, 0, 1), 0.0, 1e-6);
}

// ===========================================================================
// Reusing driver overloads.  The point of these is that a caller sweeping many
// points allocates once rather than once per call; the tests here are about the
// buffers staying *correct* under that reuse, which is the part a benchmark
// cannot see.
// ===========================================================================

namespace {
// Deliberately not separable and not symmetric in the variables, so a stale
// cell from a previous call cannot coincidentally be the right answer.
auto reuse_energy = [](const auto *q) {
  using std::exp, std::log;
  return q[0] * q[0] * q[1] + exp(q[0] * q[2]) + q[1] * log(q[1] + 2.0);
};
} // namespace

TEST(ForwardDriverReuse, WritingOverloadMatchesOwningOverload) {
  const std::array<double, 3> pt{0.4, 0.9, 1.3};
  const std::span<const double> x{pt};

  const auto owned = diff::impl::detail::hessian(reuse_energy, x);

  // Caller-owned output: the library writes into memory it did not create.
  diff::impl::HessianWorkspace ws;
  std::array<double, 3> grad{};
  std::array<double, 9> hess{};
  const double value = diff::impl::detail::hessian(
      reuse_energy, x, diff::impl::detail::all_indices(3), ws,
      std::span<double>{grad}, std::span<double>{hess});

  EXPECT_DOUBLE_EQ(value, val_of(owned));
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(grad[i], grad_at(owned, i));
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(hess[i * 3 + j], hess_at(owned, i, j));
    }
  }
}

TEST(ForwardDriverReuse, WorkspaceSurvivesAShrinkingExtent) {
  // The scratch workspace is grow-only.  A second call with a SMALLER point must
  // seed only that point's slots and read only those -- if the sweep were to run
  // over the stale tail from the larger call it would read another problem's
  // numbers, which is silently wrong rather than a crash.
  diff::impl::HessianWorkspace ws;

  const std::array<double, 3> big{0.4, 0.9, 1.3};
  std::array<double, 3> g3{};
  std::array<double, 9> h3{};
  diff::impl::detail::hessian(reuse_energy, std::span<const double>{big},
                               diff::impl::detail::all_indices(3), ws,
                               std::span<double>{g3}, std::span<double>{h3});

  auto planar = [](const auto *q) { return q[0] * q[0] * q[1]; };
  const std::array<double, 2> small{0.4, 0.9};
  std::array<double, 2> g2{};
  std::array<double, 4> h2{};
  diff::impl::detail::hessian(planar, std::span<const double>{small},
                               diff::impl::detail::all_indices(2), ws,
                               std::span<double>{g2}, std::span<double>{h2});

  // f = x0^2 x1 at (0.4, 0.9): d2/dx0^2 = 2 x1, d2/dx0dx1 = 2 x0, d2/dx1^2 = 0
  EXPECT_DOUBLE_EQ(h2[0 * 2 + 0], 2.0 * 0.9);
  EXPECT_DOUBLE_EQ(h2[0 * 2 + 1], 2.0 * 0.4);
  EXPECT_DOUBLE_EQ(h2[1 * 2 + 0], 2.0 * 0.4);
  EXPECT_DOUBLE_EQ(h2[1 * 2 + 1], 0.0);
}

TEST(ForwardDriverReuse, AllIndicesMatchesAnExplicitSubset) {
  // all_indices() is a view, not a container; it must behave identically to the
  // materialised span it replaced.
  const std::array<double, 3> pt{0.4, 0.9, 1.3};
  const std::span<const double> x{pt};
  const std::array<std::size_t, 3> idx{0, 1, 2};

  const auto viewed = diff::impl::detail::hessian(
      reuse_energy, x, diff::impl::detail::all_indices(3));
  const auto spanned = diff::impl::detail::hessian(
      reuse_energy, x, std::span<const std::size_t>{idx});

  ASSERT_EQ(hess_n(viewed), hess_n(spanned));
  for (std::size_t i = 0; i < hess_n(viewed); ++i) {
    EXPECT_DOUBLE_EQ(grad_at(viewed, i), grad_at(spanned, i));
    for (std::size_t j = 0; j < hess_n(viewed); ++j) {
      EXPECT_DOUBLE_EQ(hess_at(viewed, i, j), hess_at(spanned, i, j));
    }
  }
}

TEST(ForwardDriverReuse, GradientReusingOverloadMatchesOwning) {
  const std::array<double, 3> pt{0.4, 0.9, 1.3};
  const std::span<const double> x{pt};

  const auto owned = diff::impl::gradient(reuse_energy, x);

  diff::impl::GradientWorkspace ws;
  std::array<double, 3> out{};
  diff::impl::gradient(reuse_energy, x, diff::impl::detail::all_indices(3), ws,
                 std::span<double>{out});

  ASSERT_EQ(out.size(), owned.size());
  for (std::size_t i = 0; i < owned.size(); ++i) {
    EXPECT_DOUBLE_EQ(out[i], owned[i]);
  }
}

TEST(ForwardDriverReuse, PointAcceptsAnyContiguousSizedRange) {
  // vector, array, C array and span must all bind and agree to the last bit --
  // they are the same numbers by construction, so anything other than exact
  // equality means a conversion is doing work.
  auto f = [](const auto *q) { return q[0] * q[0] * q[1] + q[2]; };

  const std::vector<double> v{0.4, 0.9, 1.3};
  const std::array<double, 3> a{0.4, 0.9, 1.3};
  const double c[3]{0.4, 0.9, 1.3};
  const std::span<const double> sp{a};

  const auto Hv = diff::impl::hessian(f, v);
  const auto Ha = diff::impl::hessian(f, a);
  const auto Hc = diff::impl::hessian(f, c);
  const auto Hs = diff::impl::hessian(f, sp);

  ASSERT_EQ(hess_n(Hv), 3u);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(grad_at(Hv, i), grad_at(Ha, i));
    EXPECT_DOUBLE_EQ(grad_at(Hv, i), grad_at(Hc, i));
    EXPECT_DOUBLE_EQ(grad_at(Hv, i), grad_at(Hs, i));
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(hess_at(Hv, i, j), hess_at(Ha, i, j));
      EXPECT_DOUBLE_EQ(hess_at(Hv, i, j), hess_at(Hc, i, j));
      EXPECT_DOUBLE_EQ(hess_at(Hv, i, j), hess_at(Hs, i, j));
    }
  }

  // gradient() takes the same range set.
  const auto gv = diff::impl::gradient(f, v);
  const auto ga = diff::impl::gradient(f, a);
  ASSERT_EQ(gv.size(), ga.size());
  for (std::size_t i = 0; i < gv.size(); ++i) EXPECT_DOUBLE_EQ(gv[i], ga[i]);
}
