#include "dual/tests_dual_common.hpp"
#include "tests_drivers_fixtures.hpp"

TEST(HessianRouter, RawCallableTakesTheScalarDriver) {
  for (std::size_t n : {std::size_t{2}, std::size_t{5}, std::size_t{9},
                        std::size_t{12}, std::size_t{20}, std::size_t{40}}) {
    std::vector<double> x(n);
    for (std::size_t k = 0; k < n; ++k) {
      x[k] = 0.15 + 0.6 * (k + 1.0) / (n + 1.0);
    }
    auto f = [n](const auto *dof) { return vf_sample(dof, n); };
    const std::span<const double> xs{x.data(), x.size()};

    const auto Hs = ddx::impl::detail::hessian(f, xs);
    const auto Hv = ddx::impl::hessian(f, xs);

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
// The router detects CExpression and auto-bridges it via seeded_energy.
TEST(SeededExprEnergy, GraphRoutesThroughPublicHessian) {
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  Variable<D, FixedString{"x03"}> d;
  auto expr = a * log(a) + b * log(b) + c * log(c) + d * log(d) +
              0.50 * (a - b) * (a - b) + 0.51 * (b - c) * (b - c) +
              0.52 * (c - d) * (c - d) + exp(a * d);

  const std::array<double, 4> x{0.2, 0.4, 0.6, 0.8};
  const std::span<const double> xs{x.data(), x.size()};

  const auto Hrouted = ddx::impl::hessian(expr, xs);

  auto f = ddx::impl::seeded_energy(expr);
  static_assert(ddx::impl::CSeededExprEnergy<decltype(f)>,
                "seeded_energy() must advertise the routing tag");
  static_assert(decltype(f)::arity == 4, "arity deduced from symbol set");
  const auto Hscalar = ddx::impl::detail::hessian(f, xs);

  EXPECT_NEAR(val_of(Hrouted), val_of(Hscalar), 1e-12);
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(grad_at(Hrouted, i), grad_at(Hscalar, i), 1e-12)
        << "grad " << i;
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_NEAR(hess_at(Hrouted, i, j), hess_at(Hscalar, i, j), 1e-12)
          << "scalar H(" << i << "," << j << ")";
    }
  }
}
// Forward-over-reverse: O(N) sweeps rather than O(N^2) probes, and it must
// agree with forward-over-forward on a real graph energy.
TEST(SeededExprEnergy, ForwardOverReverseAgreesWithNumericDrivers) {
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  Variable<D, FixedString{"x03"}> d;
  auto expr = a * log(a) + b * log(b) + c * log(c) + d * log(d) +
              0.50 * (a - b) * (a - b) + 0.51 * (b - c) * (b - c) +
              0.52 * (c - d) * (c - d) + exp(a * d);

  const std::array<double, 4> x{0.2, 0.4, 0.6, 0.8};
  const std::span<const double> xs{x.data(), x.size()};

  auto f = ddx::impl::seeded_energy(expr);
  const auto Hscalar = ddx::impl::detail::hessian(f, xs);
  const auto Hrev = Equation{expr}.hessian(x);

  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_NEAR(Hrev[i][j], hess_at(Hscalar, i, j), 1e-9)
          << "scalar H(" << i << "," << j << ")";
    }
  }
}
TEST(Ownership, GraphHessianRejectsAPointShorterThanTheSymbolSet) {
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  auto expr = a * b + exp(c);

  // seeded_energy() reads slots [0,3), so without the guard this reads off
  // the end of the driver's dof vector.
  const std::array<double, 2> shortx{0.2, 0.4};
  EXPECT_EQ(
      ddx::impl::hessian(expr, std::span<const double>{shortx}).error().code,
      ddx::errc::wrong_arity);

  // Surplus values are just as wrong: the extra rows would come back zero.
  const std::array<double, 4> longx{0.2, 0.4, 0.6, 0.8};
  EXPECT_EQ(
      ddx::impl::hessian(expr, std::span<const double>{longx}).error().code,
      ddx::errc::wrong_arity);

  const std::array<double, 3> okx{0.2, 0.4, 0.6};
  EXPECT_TRUE(
      ddx::impl::hessian(expr, std::span<const double>{okx}).has_value());
}
TEST(Ownership, GraphHessianRejectsAnActiveIndexThatNamesNoSymbol) {
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  auto expr = a * b;

  const std::array<double, 4> x{0.2, 0.4, 0.6, 0.8};
  const std::array<std::size_t, 2> bad{2, 3}; // no such symbols
  EXPECT_EQ(ddx::impl::hessian(expr, std::span<const double>{x},
                               std::span<const std::size_t>{bad})
                .error()
                .code,
            ddx::errc::index_out_of_range);

  const std::array<std::size_t, 1> ok{1};
  EXPECT_TRUE(ddx::impl::hessian(expr, std::span<const double>{x},
                                 std::span<const std::size_t>{ok})
                  .has_value());
}
TEST(Ownership, ResultOwnsItsBuffersAndTransfersThem) {
  // Owning std types: the buffers move with the result and die with it.
  auto f = [](const auto *d) { return d[0] * d[0] * d[1]; };
  const std::array<double, 2> x{2.0, 3.0};

  auto H = *ddx::impl::hessian(f, std::span<const double>{x});
  EXPECT_DOUBLE_EQ(hess_at(H, 0, 1), 4.0); // d2/dx0dx1 of x0^2 x1 = 2 x0

  // Moving the result moves the buffers.
  const double *const before = hess_ptr(H);
  auto moved = std::move(H);
  EXPECT_EQ(hess_ptr(moved), before) << "move must not copy the buffer";
  EXPECT_DOUBLE_EQ(hess_at(moved, 0, 1), 4.0);
  EXPECT_EQ(H.hessian.get(), nullptr) << "moved-from must have released";

  // The extent travels with the buffers: unique_ptr<double[]> has no length.
  EXPECT_EQ(hess_n(moved), 2u);

  // Same for the value maps: lvalue borrows, rvalue copies.
  const auto m = values(named<"x">(2.0), named<"y">(3.0));
  static_assert(std::is_same_v<decltype(m.get<"x">()), const double &>,
                "get() on an lvalue map borrows");
  static_assert(
      std::is_same_v<decltype(values(named<"x">(2.0)).get<"x">()), double>,
      "get() on a temporary map must return by value");
  EXPECT_DOUBLE_EQ(values(named<"x">(2.0)).get<"x">(), 2.0);
  EXPECT_DOUBLE_EQ(
      bind(var<"x"> * var<"y">, named<"x">(4.0), named<"y">(5.0)).get<"x">(),
      4.0);
}
TEST(Ownership, ReverseHessianAcceptsATemporaryExpression) {
  // A temporary expression has to reach hessian(), not just jacobian().
  using D = ddx::impl::Dual<double>;
  const auto H =
      Equation{var<"x", dual> * var<"y", dual>}.hessian(std::array{2.0, 3.0});
  const auto g = Equation{var<"x", dual> * var<"y", dual>}.jacobian(
      std::array{D{2.0}, D{3.0}});
  EXPECT_DOUBLE_EQ(H[0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[0][0], 0.0);
  EXPECT_DOUBLE_EQ(g[0], 3.0);
}
TEST(HessianCoupling, ChainPatternIsTridiagonalPlusCorner) {
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  Variable<D, FixedString{"x03"}> d;
  auto expr = a * log(a) + b * log(b) + c * log(c) + d * log(d) +
              0.50 * (a - b) * (a - b) + 0.51 * (b - c) * (b - c) +
              0.52 * (c - d) * (c - d) + exp(a * d);

  constexpr auto P = ddx::impl::hessian_pattern<decltype(expr)>();
  using Row = ddx::impl::symbol_set<4>;
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
      ddx::impl::detail::hessian(ddx::impl::seeded_energy(expr), xs);
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
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto expr = 3.0 * x + 4.0 * y - x;
  constexpr auto P = ddx::impl::hessian_pattern<decltype(expr)>();
  static_assert(P[0].none() && P[1].none(), "a linear form has a zero Hessian");
  constexpr auto C = ddx::impl::color_columns<2>(P);
  static_assert(C.count == 1, "nothing conflicts, so one sweep suffices");
}
TEST(HessianCoupling, DenseExpressionDegradesToOneSweepPerColumn) {
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  auto expr = exp(a * b * c);
  constexpr auto P = ddx::impl::hessian_pattern<decltype(expr)>();
  static_assert(P[0].all() && P[1].all() && P[2].all(),
                "a fully coupled expression must not be pruned");
  constexpr auto C = ddx::impl::color_columns<3>(P);
  static_assert(C.count == 3, "dense pattern gives one colour per column");
}
// Quotient-rule branch of the coupling pass: curvature in the denominator
// only.
TEST(HessianCoupling, CompressedDriverMatchesProbeDriverOnQuotients) {
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  auto expr = a / b + b / c + a * c / (1.0 + b) + log(a * b * c);

  const std::array<double, 3> x{0.7, 1.3, 2.1};
  const std::span<const double> xs{x.data(), x.size()};

  const auto Hcompressed = ddx::impl::hessian(expr, xs); // routed: compressed
  const auto Hprobe =
      ddx::impl::detail::hessian(ddx::impl::seeded_energy(expr), xs);

  EXPECT_NEAR(val_of(Hcompressed), val_of(Hprobe), 1e-12);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(grad_at(Hcompressed, i), grad_at(Hprobe, i), 1e-9) << i;
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_NEAR(hess_at(Hcompressed, i, j), hess_at(Hprobe, i, j), 1e-9)
          << "H(" << i << "," << j << ")";
    }
  }
}
// Trig and mixed products, against the structure-blind driver.
TEST(HessianCoupling, CompressedDriverMatchesProbeDriverOnTrigProducts) {
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  Variable<D, FixedString{"x03"}> d;
  auto expr = sin(a) * cos(b) + exp(c * d) + a * b * c + sqrt(d) + a * a;

  const std::array<double, 4> x{0.3, 0.6, 0.9, 1.2};
  const std::span<const double> xs{x.data(), x.size()};

  const auto Hc = ddx::impl::hessian(expr, xs);
  const auto Hp =
      ddx::impl::detail::hessian(ddx::impl::seeded_energy(expr), xs);
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_NEAR(hess_at(Hc, i, j), hess_at(Hp, i, j), 1e-9)
          << "H(" << i << "," << j << ")";
    }
  }
}
// The dense Hessian buffer is row-major with the extent alongside it: the
// layout `H[i * n + j]` a caller maps its own matrix type onto.
TEST(SparseHessian, DenseBufferIsRowMajorWithItsExtent) {
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  // Asymmetric in the SOURCE, so a transposed read would show: d2/da db of
  // a*a*b is 2a, of b*b*a is 2b.
  auto expr = a * log(a) + b * log(b) + 0.5 * (a - b) * (a - b);

  const std::array<double, 2> x{0.3, 0.7};
  const std::span<const double> xs{x.data(), x.size()};
  const auto H = ddx::impl::hessian(expr, xs);

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
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  Variable<D, FixedString{"x03"}> d;
  auto expr = a * log(a) + b * log(b) + c * log(c) + d * log(d) +
              0.50 * (a - b) * (a - b) + 0.51 * (b - c) * (b - c) +
              0.52 * (c - d) * (c - d) + exp(a * d);

  const std::array<double, 4> x{0.2, 0.4, 0.6, 0.8};
  const std::span<const double> xs{x.data(), x.size()};

  const auto dense =
      ddx::impl::detail::hessian(ddx::impl::seeded_energy(expr), xs);
  const auto sparse = ddx::impl::sparse_hessian(expr, xs);

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
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  // a and c never meet: (0,2) and (2,0) are structural zeros.
  auto expr = a * b + b * c + log(a) + exp(c);

  const std::array<double, 3> x{0.7, 1.3, 2.1};
  const std::span<const double> xs{x.data(), x.size()};

  const auto sparse = ddx::impl::sparse_hessian(expr, xs);
  const auto dense =
      ddx::impl::detail::hessian(ddx::impl::seeded_energy(expr), xs);

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
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  Variable<D, FixedString{"x00"}> a;
  Variable<D, FixedString{"x01"}> b;
  Variable<D, FixedString{"x02"}> c;
  auto expr = a / b + b / c + log(a * b * c) + exp(a * c);

  const std::array<double, 3> x{0.7, 1.3, 2.1};
  const std::span<const double> xs{x.data(), x.size()};

  const auto sparse = ddx::impl::sparse_hessian(expr, xs);
  const auto dense =
      ddx::impl::detail::hessian(ddx::impl::seeded_energy(expr), xs);
  const auto M = densify(sparse);

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_NEAR(M[i * 3 + j], hess_at(dense, i, j), 1e-9);
    }
  }
  // No symmetrization pass on the sparse path: symmetry comes out of the
  // sweeps themselves.
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_NEAR(M[i * 3 + j], M[j * 3 + i], 1e-9);
    }
  }
  // Every column's row indices strictly ascend, as a CSC consumer assumes.
  const auto outer = decltype(sparse)::outer();
  const auto inner = decltype(sparse)::inner();
  for (std::size_t j = 0; j < 3; ++j) {
    for (auto k = static_cast<std::size_t>(outer[j]) + 1;
         k < static_cast<std::size_t>(outer[j + 1]); ++k) {
      EXPECT_LT(inner[k - 1], inner[k]) << "column " << j << " is unsorted";
    }
  }
}
// A constexpr dropped between sparse_hessian and the sweeps would leave every
// runtime test green; these fail to compile instead.  Dual<double>, NOT
// dual2nd: hessian_values_sparse harvests grad.get<1>() as double.
TEST(ConstexprContract, BothHessianDriversAreConstantEvaluated) {
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;

  // f(x, y) = x*y + 0.5*x*x  ->  H = [[1, 1], [1, 0]], exact and point-free.
  constexpr Variable<D, FixedString{"x00"}> a;
  constexpr Variable<D, FixedString{"x01"}> b;
  constexpr auto expr = a * b + 0.5 * a * a;
  constexpr std::array<double, 2> at{0.3, 0.7};

  constexpr auto sparse =
      ddx::impl::sparse_hessian(expr, std::span<const double>{at});
  static_assert(sparse.rows == 2);
  static_assert(sparse.structural(0, 1), "the cross term is stored");
  static_assert(sparse[0, 1] == 1.0, "d2f/dx dy");
  static_assert(sparse[1, 0] == 1.0, "and its mirror");
  static_assert(sparse[0, 0] == 1.0, "d2f/dx2");
  static_assert(sparse.values().size() == sparse.nnz);

  // The dense driver is row-major, hence i * n + j.
  constexpr auto dense = ddx::impl::hessian(expr, std::span<const double>{at});
  static_assert(dense.has_value());
  static_assert(dense->hessian[0 * 2 + 1] == 1.0);
  static_assert(dense->hessian[0 * 2 + 0] == 1.0);
  static_assert(dense->hessian[1 * 2 + 1] == 0.0, "y appears only linearly");
}

// A linear expression has nnz 0, so the buffer collapses to the sink cell --
// where an off-by-one in the sizing shows up as a zero-length array.
TEST(SparseHessian, OfALinearExpressionIsEmpty) {
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  Variable<D, FixedString{"x00"}> x;
  Variable<D, FixedString{"x01"}> y;
  auto expr = 3.0 * x + 4.0 * y - x;

  const std::array<double, 2> pt{0.7, 1.3};
  const std::span<const double> xs{pt.data(), pt.size()};

  const auto sparse = ddx::impl::sparse_hessian(expr, xs);
  static_assert(decltype(sparse)::nnz == 0,
                "a linear expression couples nothing, so it stores nothing");
  EXPECT_EQ(sparse.values().size(), 0u);
  EXPECT_EQ(decltype(sparse)::outer().back(), 0);

  // Every entry reads back as exactly 0.0 through the shared sink cell.
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      EXPECT_FALSE(decltype(sparse)::structural(i, j));
      EXPECT_DOUBLE_EQ((sparse[i, j]), 0.0);
    }
  }
}
TEST(HessianRouter, IdealMixingClosedForm) {
  constexpr double R = 8.31446261815324, T = 1000.0;
  auto f = [](const auto *y) {
    using std::log;
    return R * T * (y[0] * log(y[0]) + y[1] * log(y[1]));
  };
  const std::array<double, 2> y{0.3, 0.7};
  const auto H =
      ddx::impl::hessian(f, std::span<const double>{y.data(), y.size()});
  EXPECT_NEAR(val_of(H), R * T * (0.3 * std::log(0.3) + 0.7 * std::log(0.7)),
              1e-6);
  EXPECT_NEAR(grad_at(H, 0), R * T * (std::log(0.3) + 1.0), 1e-6);
  EXPECT_NEAR(grad_at(H, 1), R * T * (std::log(0.7) + 1.0), 1e-6);
  EXPECT_NEAR(hess_at(H, 0, 0), R * T / 0.3, 1e-3);
  EXPECT_NEAR(hess_at(H, 1, 1), R * T / 0.7, 1e-3);
  EXPECT_NEAR(hess_at(H, 0, 1), 0.0, 1e-6);
}
TEST(ForwardDriverReuse, WritingOverloadMatchesOwningOverload) {
  const std::array<double, 3> pt{0.4, 0.9, 1.3};
  const std::span<const double> x{pt};

  const auto owned = ddx::impl::detail::hessian(reuse_energy, x);

  // Caller-owned output: the library writes into memory it did not create.
  ddx::impl::HessianWorkspace ws;
  std::array<double, 3> grad{};
  std::array<double, 9> hess{};
  const double value = ddx::impl::detail::hessian(
      reuse_energy, x, ddx::impl::detail::all_indices(3), ws,
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
  // The scratch workspace is grow-only: a second, SMALLER point must touch
  // only its own slots, or the stale tail is read as live.
  ddx::impl::HessianWorkspace ws;

  const std::array<double, 3> big{0.4, 0.9, 1.3};
  std::array<double, 3> g3{};
  std::array<double, 9> h3{};
  ddx::impl::detail::hessian(reuse_energy, std::span<const double>{big},
                             ddx::impl::detail::all_indices(3), ws,
                             std::span<double>{g3}, std::span<double>{h3});

  auto planar = [](const auto *q) { return q[0] * q[0] * q[1]; };
  const std::array<double, 2> small{0.4, 0.9};
  std::array<double, 2> g2{};
  std::array<double, 4> h2{};
  ddx::impl::detail::hessian(planar, std::span<const double>{small},
                             ddx::impl::detail::all_indices(2), ws,
                             std::span<double>{g2}, std::span<double>{h2});

  EXPECT_DOUBLE_EQ(h2[0 * 2 + 0], 2.0 * 0.9);
  EXPECT_DOUBLE_EQ(h2[0 * 2 + 1], 2.0 * 0.4);
  EXPECT_DOUBLE_EQ(h2[1 * 2 + 0], 2.0 * 0.4);
  EXPECT_DOUBLE_EQ(h2[1 * 2 + 1], 0.0);
}
TEST(ForwardDriverReuse, AllIndicesMatchesAnExplicitSubset) {
  // all_indices() is a view, not a container, and must behave identically to a
  // materialised span.
  const std::array<double, 3> pt{0.4, 0.9, 1.3};
  const std::span<const double> x{pt};
  const std::array<std::size_t, 3> idx{0, 1, 2};

  const auto viewed = ddx::impl::detail::hessian(
      reuse_energy, x, ddx::impl::detail::all_indices(3));
  const auto spanned = ddx::impl::detail::hessian(
      reuse_energy, x, std::span<const std::size_t>{idx});

  ASSERT_EQ(hess_n(viewed), hess_n(spanned));
  for (std::size_t i = 0; i < hess_n(viewed); ++i) {
    EXPECT_DOUBLE_EQ(grad_at(viewed, i), grad_at(spanned, i));
    for (std::size_t j = 0; j < hess_n(viewed); ++j) {
      EXPECT_DOUBLE_EQ(hess_at(viewed, i, j), hess_at(spanned, i, j));
    }
  }
}
TEST(ForwardDriverReuse, JacobianReusingOverloadMatchesOwning) {
  const std::array<double, 3> pt{0.4, 0.9, 1.3};
  const std::span<const double> x{pt};

  const auto owned = ddx::impl::jacobian(reuse_energy, x);

  ddx::impl::JacobianWorkspace ws;
  std::array<double, 3> out{};
  ddx::impl::jacobian(reuse_energy, x, ddx::impl::detail::all_indices(3), ws,
                      std::span<double>{out});

  ASSERT_EQ(out.size(), owned.size());
  for (std::size_t i = 0; i < owned.size(); ++i) {
    EXPECT_DOUBLE_EQ(out[i], owned[i]);
  }
}
TEST(ForwardDriverReuse, PointAcceptsAnyContiguousSizedRange) {
  // vector, array, C array and span agree bit-for-bit, or a conversion is
  // doing work.
  auto f = [](const auto *q) { return q[0] * q[0] * q[1] + q[2]; };

  const std::vector<double> v{0.4, 0.9, 1.3};
  const std::array<double, 3> a{0.4, 0.9, 1.3};
  const double c[3]{0.4, 0.9, 1.3};
  const std::span<const double> sp{a};

  const auto Hv = ddx::impl::hessian(f, v);
  const auto Ha = ddx::impl::hessian(f, a);
  const auto Hc = ddx::impl::hessian(f, c);
  const auto Hs = ddx::impl::hessian(f, sp);

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

  // jacobian() takes the same range set.
  const auto gv = ddx::impl::jacobian(f, v);
  const auto ga = ddx::impl::jacobian(f, a);
  ASSERT_EQ(gv.size(), ga.size());
  for (std::size_t i = 0; i < gv.size(); ++i)
    EXPECT_DOUBLE_EQ(gv[i], ga[i]);
}
