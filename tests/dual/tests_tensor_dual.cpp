#include "dual/tests_dual_common.hpp"
#include "tests_tensor_fixtures.hpp"

TEST(BitExactness, EveryDriverIsBitStableAcrossBuilds) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  Variable<double, FixedString{"z"}> z;
  const auto e = sin(x * y) * exp(x + y) + log(y * y + 1.0) / (x + 2.0) -
                 tanh(x) + sqrt(z * z + 1.0) + atan(x * z) + erf(y) +
                 cbrt(z + 3.0) + pow(z, 2.5);
  BitHash feed;
  for (int i = 1; i <= 120; ++i) {
    const double a = 0.01 * i, b = 0.02 * i + 0.3, c = 0.015 * i + 0.5;
    const std::array<double, 3> p{a, b, c};
    for (double v : Equation{e}.jacobian(p))
      feed(v);
    const auto g1 = Equation{e}.template derivative_tensor<1>(p);
    for (std::size_t k = 0; k < 3; ++k)
      feed(g1.data()[k]);
    // Packed symmetric storage -- a 3x3 second-order tensor stores 6 elements,
    // not 9 -- so walk the index space, not the buffer.
    const auto g2 = Equation{e}.template derivative_tensor<2>(p);
    for (std::size_t r = 0; r < 3; ++r)
      for (std::size_t cc = 0; cc < 3; ++cc)
        feed(g2[r, cc]);
    feed(e.eval(a, b, c));
    feed(e.eval_with_tangent<"x">(a, b, c).deriv());

    const std::vector<double> xs{a, b, c};
    auto fn = [](const auto *q) {
      using std::exp, std::sin;
      return sin(q[0] * q[1]) * exp(q[0]) + q[2] * q[2] * q[1];
    };
    const auto H = ddx::impl::hessian(fn, std::span<const double>{xs});
    feed(val_of(H));
    for (std::size_t k = 0; k < hess_n(H); ++k)
      feed(grad_at(H, k));
    for (std::size_t r = 0; r < 3; ++r)
      for (std::size_t cc = 0; cc < 3; ++cc)
        feed(hess_at(H, r, cc));
    for (auto v : ddx::impl::jacobian(fn, std::span<const double>{xs}))
      feed(v);

    const auto u = sin(x * x) + exp(x) + atan(x) + asinh(x);
    feed(Equation{u}.template univariate_derivative<1>(a));
    feed(Equation{u}.template univariate_derivative<2>(a));
    feed(Equation{u}.template univariate_derivative<4>(a));
  }
  // Always reported, so an A/B is a matter of reading two runs.
  std::printf("[ BIT HASH ] %016llx\n",
              static_cast<unsigned long long>(feed.h));
#ifdef DDX_PIN_BIT_HASH
  EXPECT_EQ(feed.h, static_cast<std::uint64_t>(DDX_PIN_BIT_HASH))
      << "a derivative moved in the last bit -- see the note above this test";
#endif
}
// detail::max_impl/min_impl spell `a < b`, so a univariate_derivative sweep
// through max or min needs TaylorDual to be ordered.
TEST(UnivariateDerivative, MaxAndMinAreDifferentiable) {
  auto x = var<"x">;
  EXPECT_DOUBLE_EQ(
      Equation{max(x * x, 1.0)}.template univariate_derivative<1>(2.0), 4.0);
  EXPECT_DOUBLE_EQ(
      Equation{min(x * x, 1.0)}.template univariate_derivative<1>(2.0), 0.0);
  EXPECT_DOUBLE_EQ(
      Equation{max(x * x, 1.0)}.template univariate_derivative<2>(2.0), 2.0);
}
// The guard on "everything is constexpr", differentiation entry points
// included, and the tripwire for anything (a foreign matrix type, a
// std::vector, an allocation) leaking into the symbolic core: a leak makes
// these static_asserts fail to compile rather than merely slow something down.
TEST(ConstexprContract, DifferentiationEntryPointsAreConstantEvaluated) {
  constexpr Variable<double, FixedString{"x"}> x;
  constexpr Variable<double, FixedString{"y"}> y;

  constexpr auto g = Equation{x * y}.jacobian(std::array{3.0, 4.0});
  static_assert(g[0] == 4.0 && g[1] == 3.0, "reverse Jacobian of x*y");

  constexpr auto gf =
      Equation{x * y}.template derivative_tensor<1>(std::array{3.0, 4.0});
  static_assert(gf[0] == 4.0 && gf[1] == 3.0, "forward Jacobian of x*y");

  constexpr auto H =
      Equation{x * y}.template derivative_tensor<2>(std::array{3.0, 4.0});
  static_assert(H[0][1] == 1.0 && H[1][0] == 1.0 && H[0][0] == 0.0,
                "d2(x*y)/dxdy == 1");

  // TaylorDual path: multiply-only, so it stays within the constexpr subset.
  constexpr auto d2 =
      Equation{x * x * x}.template univariate_derivative<2>(2.0);
  static_assert(d2 == 12.0, "d2(x^3)/dx2 at x=2");

  // A three-variable Jacobian, which also has to be usable in constant
  // evaluation.
  constexpr Variable<double, FixedString{"z"}> z;
  constexpr auto g3 = Equation{x * y * z}.template derivative_tensor<1>(
      std::array{2.0, 3.0, 4.0});
  static_assert(g3[0] == 12.0 && g3[1] == 8.0 && g3[2] == 6.0,
                "forward Jacobian path is constexpr");

  // Forward-over-reverse seeds tangents, so its expression carries Dual<double>
  // numbers.
  using D = Dual<double>;
  constexpr Variable<D, FixedString{"a"}> a;
  constexpr Variable<D, FixedString{"b"}> b;
  constexpr auto Hr = Equation{a * b}.hessian(std::array{3.0, 4.0});
  static_assert(Hr[0][1] == 1.0 && Hr[0][0] == 0.0,
                "forward-over-reverse Hessian in constant evaluation");

  SUCCEED();
}
TEST(BoundTest, EvalAsCarriesDualsThroughTheGraph) {
  // eval_as is the deeper-numeric-type entry point the drivers use.
  auto x = var<"x">;
  auto y = var<"y">;
  const auto b = bind(x * x + y, named<"x">(3.0), named<"y">(4.0));

  using D = Dual<double>;
  const std::array<D, 2> seed{D{3.0, 1.0}, D{4.0, 0.0}}; // seed d/dx
  const auto r = b.eval_as<D>(seed);
  EXPECT_DOUBLE_EQ(r.get<0>(), 13.0);
  EXPECT_DOUBLE_EQ(r.get<1>(), 6.0); // d/dx (x² + y) = 2x = 6
}
TEST(MdLayout, LeadingSimplexKeepsOutputsApart) {
  // Two outputs, each a symmetric 3x3 Hessian: 2 * 6 cells, not 2 * 9.
  using T = ddx::impl::md_tensor<double, ddx::impl::stacked_extents_t<2, 3, 2>,
                                 ddx::impl::layout_leading_simplex<1>>;
  EXPECT_EQ(T::size(), 12u);

  T t{};
  t[0, 1, 2] = 5.0;
  t[1, 1, 2] = 9.0;
  EXPECT_DOUBLE_EQ((t[0, 2, 1]), 5.0); // symmetric within an output
  EXPECT_DOUBLE_EQ((t[1, 2, 1]), 9.0);
  EXPECT_NE((t[0, 1, 2]), (t[1, 1, 2])); // but outputs never alias
}
TEST(DerivativeTensorTest, ForwardAndReverseHessiansAgree) {
  // derivative_tensor<2> is forward-over-forward; hessian<Reverse> is
  // forward-over-reverse.  Two algorithms, same tensor and same type --
  // which is what lets a caller pick on cost alone.
  using D = ddx::impl::Dual<double>;
  using ddx::impl::FixedString;
  {
    Variable<D, FixedString{"a"}> a;
    Variable<D, FixedString{"b"}> b;
    Variable<D, FixedString{"c"}> c;
    Variable<D, FixedString{"d"}> d;
    auto e = a * log(a) + b * log(b) + c * c * d + exp(a * d);
    const std::array<double, 4> p{0.3, 0.4, 0.5, 0.6};
    const auto T = Equation{e}.template derivative_tensor<2>(p);
    const auto R = ddx::impl::detail::reverse_mode_hessian(e, p);
    for (std::size_t i = 0; i < 4; ++i) {
      for (std::size_t j = 0; j < 4; ++j) {
        EXPECT_NEAR((T[i, j]), (R[i, j]), 1e-12)
            << "N=4 at (" << i << "," << j << ")";
      }
    }
  }
  {
    Variable<D, FixedString{"a"}> a;
    Variable<D, FixedString{"b"}> b;
    Variable<D, FixedString{"c"}> c;
    Variable<D, FixedString{"d"}> d;
    Variable<D, FixedString{"e"}> e5;
    auto e = a * log(a) + b * log(b) + c * c * d + exp(a * d) + e5 * e5 * b;
    const std::array<double, 5> p{0.3, 0.4, 0.5, 0.6, 0.7};
    const std::span<const double> xs{p.data(), p.size()};
    const auto T = Equation{e}.template derivative_tensor<2>(p);
    const auto R = ddx::impl::detail::reverse_mode_hessian(e, p);
    const auto H =
        ddx::impl::hessian(e, xs); // independent driver, as a cross-check
    for (std::size_t i = 0; i < 5; ++i) {
      for (std::size_t j = 0; j < 5; ++j) {
        EXPECT_NEAR((T[i, j]), (R[i, j]), 1e-12)
            << "N=5 at (" << i << "," << j << ")";
        EXPECT_NEAR((T[i, j]), hess_at(H, i, j), 1e-9)
            << "N=5 vs driver at (" << i << "," << j << ")";
      }
    }
  }
}
TEST(ForwardDriver, DriverHessianAgreesWithEquationHessian) {
  // Two routes to the same Hessian with two return shapes: the driver hands
  // back a tuple of owning row-major buffers, Equation an md_tensor.
  Variable<ddx::impl::Dual<double>, ddx::impl::FixedString{"x"}> x;
  Variable<ddx::impl::Dual<double>, ddx::impl::FixedString{"y"}> y;
  auto expr = x * y + x * x + sin(y);
  const std::array<double, 2> p{0.7, 1.3};
  const std::span<const double> xs{p.data(), p.size()};

  const auto H = ddx::impl::hessian(expr, xs); // tuple {value, grad, hess}
  const auto T = Equation{expr}.hessian(p);    // md_tensor

  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      EXPECT_NEAR(hess_at(H, i, j), (T[i, j]), 1e-9)
          << "driver vs Equation at (" << i << "," << j << ")";
    }
  }
  // Row-major is part of the driver's contract: (0,1) is element 1 of the flat
  // buffer.
  EXPECT_DOUBLE_EQ(hess_ptr(H)[1], hess_at(H, 0, 1));
}
TEST(MdTensor, PackedSymmetricTensorMatchesDenseEvaluation) {
  ddx::impl::Variable<double, ddx::impl::FixedString{"x"}> x;
  ddx::impl::Variable<double, ddx::impl::FixedString{"y"}> y;
  ddx::impl::Variable<double, ddx::impl::FixedString{"z"}> z;

  ExpectPackedMatchesDense<2>(x * y + x * x, std::array{2.0, 3.0});
  ExpectPackedMatchesDense<3>(x * y * y, std::array{2.0, 3.0});
  ExpectPackedMatchesDense<2>(exp(x * y) + sin(z), std::array{0.3, 0.4, 0.5});
  ExpectPackedMatchesDense<3>(exp(x * y) + sin(z), std::array{0.3, 0.4, 0.5});
  ExpectPackedMatchesDense<4>(x * y * z, std::array{1.5, 2.5, 0.5});
}
