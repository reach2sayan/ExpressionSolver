#include "dual/tests_dual_common.hpp"
#include "tests_math_fixtures.hpp"

TEST(NewMathFunctions, MaxMinTiesAverageAndNaNPropagatesSymmetrically) {
  auto x = var<"x">;
  auto y = var<"y">;
  // A tie is the mean of the two subgradients in every engine.
  auto tie = Equation{max(x, y)}.gradient(2.0, 2.0);
  EXPECT_DOUBLE_EQ(tie[0], 0.5);
  EXPECT_DOUBLE_EQ(tie[1], 0.5);
  // A NaN operand propagates from either side -- fmax-style dropping would
  // depend on the operand order the graph builder happens to store -- and it
  // poisons the derivative along with the value.
  const double qnan = std::nan("");
  EXPECT_TRUE(std::isnan(max(x, y).eval(qnan, 2.0)));
  EXPECT_TRUE(std::isnan(max(x, y).eval(2.0, qnan)));
  EXPECT_TRUE(std::isnan(min(x, y).eval(qnan, 2.0)));
  EXPECT_TRUE(std::isnan(min(x, y).eval(2.0, qnan)));
  auto gnan = Equation{max(x, y)}.gradient(2.0, qnan);
  EXPECT_TRUE(std::isnan(gnan[0]));
  EXPECT_TRUE(std::isnan(gnan[1]));
  // abs: sign(0) is 0 and sign(NaN) is NaN.
  EXPECT_DOUBLE_EQ(abs(Dual<double>{0.0, 1.0}).deriv(), 0.0);
  EXPECT_TRUE(std::isnan(abs(Dual<double>{qnan, 1.0}).deriv()));
}
// ---- Forward lazy-dual mode (Variable<Dual>) ------------------------------
TEST(NewMathFunctions, ForwardDualUnary) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  {
    auto [f, df] = erf(x).eval(Dual<double>{x0, 1.0});
    EXPECT_DOUBLE_EQ(f, std::erf(x0));
    EXPECT_DOUBLE_EQ(df, k2OverSqrtPi * std::exp(-x0 * x0));
  }
  {
    auto [f, df] = atanh(x).eval(Dual<double>{x0, 1.0});
    EXPECT_DOUBLE_EQ(f, std::atanh(x0));
    EXPECT_DOUBLE_EQ(df, 1.0 / (1.0 - x0 * x0));
  }
}
TEST(NewMathFunctions, ForwardDualBinary) {
  double x0 = 3.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  {
    auto [f, df] = hypot(x, 4.0).eval(Dual<double>{x0, 1.0});
    EXPECT_DOUBLE_EQ(f, 5.0);
    EXPECT_DOUBLE_EQ(df, x0 / 5.0);
  }
  {
    auto [f, df] = pow(x, 3.0).eval(Dual<double>{x0, 1.0});
    EXPECT_DOUBLE_EQ(f, 27.0);
    EXPECT_DOUBLE_EQ(df, 3.0 * x0 * x0);
  }
}
// ---- Lazy dual used directly (Dual<T>) ------------------------------------
TEST(NewMathFunctions, DualDirect) {
  Dual<double> x{0.7, 1.0};
  Dual<double> r = asinh(x);
  EXPECT_DOUBLE_EQ(r.get<0>(), std::asinh(0.7));
  EXPECT_DOUBLE_EQ(r.get<1>(), 1.0 / std::sqrt(0.49 + 1.0));

  Dual<double> a{1.0, 1.0}, b{2.0, 0.0}, c{2.0, 0.0};
  Dual<double> h = hypot(a, b, c);
  EXPECT_DOUBLE_EQ(h.get<0>(), 3.0);       // sqrt(1+4+4)
  EXPECT_DOUBLE_EQ(h.get<1>(), 1.0 / 3.0); // a/h
}
// ---- Taylor higher-order (value coeff drives expr + Taylor together) -------
TEST(NewMathFunctions, TaylorFirstAndSecondDerivative) {
  EXPECT_NEAR(Equation{asinh(var<"x">)}.template univariate_derivative<1>(0.7),
              1.0 / std::sqrt(0.49 + 1.0), 1e-12);
  EXPECT_NEAR(Equation{erf(var<"x">)}.template univariate_derivative<1>(0.5),
              k2OverSqrtPi * std::exp(-0.25), 1e-12);
  EXPECT_NEAR(Equation{cbrt(var<"x">)}.template univariate_derivative<1>(2.0),
              1.0 / (3.0 * std::cbrt(2.0) * std::cbrt(2.0)), 1e-12);

  {
    double x0 = 0.7; // asinh'' = -x/(1+x²)^{3/2}
    EXPECT_NEAR(
        Equation{asinh(var_of<"x">(x0))}.template univariate_derivative<2>(x0),
        -x0 / std::pow(1.0 + x0 * x0, 1.5), 1e-10);
  }
  {
    double x0 = 0.5; // erf'' = -2x * 2/√π * e^{-x²}
    EXPECT_NEAR(
        Equation{erf(var_of<"x">(x0))}.template univariate_derivative<2>(x0),
        -2.0 * x0 * k2OverSqrtPi * std::exp(-x0 * x0), 1e-10);
  }
  {
    double x0 = 0.3; // atanh'' = 2x/(1-x²)²
    EXPECT_NEAR(
        Equation{atanh(var_of<"x">(x0))}.template univariate_derivative<2>(x0),
        2.0 * x0 / std::pow(1.0 - x0 * x0, 2.0), 1e-10);
  }
  {
    double x0 = 3.0; // log10'' = -1/(x² ln10)
    EXPECT_NEAR(
        Equation{log10(var_of<"x">(x0))}.template univariate_derivative<2>(x0),
        -1.0 / (x0 * x0 * kLn10), 1e-10);
  }
  {
    double x0 = 2.0; // cbrt'' = -2/9 x^{-5/3}
    EXPECT_NEAR(
        Equation{cbrt(var_of<"x">(x0))}.template univariate_derivative<2>(x0),
        -2.0 / 9.0 * std::pow(x0, -5.0 / 3.0), 1e-10);
  }
  {
    double x0 = 2.0; // acosh'' = -x/(x²-1)^{3/2}
    EXPECT_NEAR(
        Equation{acosh(var_of<"x">(x0))}.template univariate_derivative<2>(x0),
        -x0 / std::pow(x0 * x0 - 1.0, 1.5), 1e-10);
  }
}
TEST(NewMathFunctions, TaylorBinaryUnivariate) {
  EXPECT_NEAR(
      Equation{pow(var<"x">, 3.0)}.template univariate_derivative<1>(2.0), 12.0,
      1e-10);
  EXPECT_NEAR(
      Equation{pow(var<"x">, 3.0)}.template univariate_derivative<2>(2.0), 12.0,
      1e-10);
  {
    double x0 = 3.0, h = std::sqrt(x0 * x0 + 4.0);
    EXPECT_NEAR(
        Equation{hypot(var_of<"x">(x0), 2.0)}.template univariate_derivative<1>(
            x0),
        x0 / h, 1e-10);
  }
  {
    double x0 = 2.0;
    EXPECT_NEAR(
        Equation{atan2(var_of<"x">(x0), 2.0)}.template univariate_derivative<1>(
            x0),
        2.0 / (4.0 + x0 * x0), 1e-10);
  }
}
// Composite arguments: the asin/atan/asinh/acosh/atanh recurrences must use the
// argument's actual derivative series (u'), not assume the bare seed (u'=1).
TEST(NewMathFunctions, TaylorCompositeArguments) {
  {
    double x0 = 0.6, x4 = x0 * x0 * x0 * x0, d = 1.0 + x4;
    auto x = var_of<"x">(x0);
    EXPECT_NEAR(Equation{atan(x * x)}.template univariate_derivative<1>(x0),
                2.0 * x0 / d, 1e-10);
    EXPECT_NEAR(Equation{atan(x * x)}.template univariate_derivative<2>(x0),
                (2.0 - 6.0 * x4) / (d * d), 1e-10);
  }
  {
    double x0 = 1.0, a = 0.5, r = 1.0 - a * a * x0 * x0;
    auto x = var_of<"x">(x0);
    EXPECT_NEAR(Equation{asin(0.5 * x)}.template univariate_derivative<1>(x0),
                a / std::sqrt(r), 1e-10);
    EXPECT_NEAR(Equation{asin(0.5 * x)}.template univariate_derivative<2>(x0),
                a * a * a * x0 * std::pow(r, -1.5), 1e-10);
  }
  {
    double x0 = 0.3, a = 2.0, r = 1.0 + a * a * x0 * x0;
    auto x = var_of<"x">(x0);
    EXPECT_NEAR(Equation{asinh(2.0 * x)}.template univariate_derivative<1>(x0),
                a / std::sqrt(r), 1e-10);
    EXPECT_NEAR(Equation{asinh(2.0 * x)}.template univariate_derivative<2>(x0),
                -a * a * a * x0 * std::pow(r, -1.5), 1e-10);
  }
  {
    auto x = var<"x">;
    EXPECT_NEAR(
        Equation{atan2(sin(x), cos(x))}.template univariate_derivative<1>(0.7),
        1.0, 1e-10);
    EXPECT_NEAR(
        Equation{atan2(sin(x), cos(x))}.template univariate_derivative<2>(0.7),
        0.0, 1e-10);
    EXPECT_NEAR(
        Equation{atan2(sin(x), cos(x))}.template univariate_derivative<3>(0.7),
        0.0, 1e-10);
  }
  {
    double x0 = 0.5;
    auto x = var_of<"x">(x0);
    EXPECT_NEAR(
        Equation{pow(sin(x), 2.0)}.template univariate_derivative<1>(x0),
        std::sin(2.0 * x0), 1e-10);
    EXPECT_NEAR(
        Equation{pow(sin(x), 2.0)}.template univariate_derivative<2>(x0),
        2.0 * std::cos(2.0 * x0), 1e-10);
  }
}

// ===========================================================================
// Forward tangent sweep (eval_with_tangent)
//
// eval_with_tangent<"s"> walks the *existing* expression tree once, seeded with
// Dual<T>, and returns a Dual whose .value()/.deriv() are the value and the
// tangent.  It is the ordinary eval_seeded sweep -- there is no separate
// forward engine -- so what these tests pin is the seeding and the Dual
// arithmetic, cross-checked against the two implementations that remain
// genuinely independent of it: the reverse sweep (backward()/adjoints()) and
// the symbolic derivative tree (derivative()).
// ===========================================================================

// Single-variable expression: value must match eval(), tangent must match the
// reverse-mode gradient computed by the separate backward()/adjoints() sweep.
// The two agree algebraically but not bit-for-bit — e.g. the quotient rule
// evaluates (a'b - ab')/b² forward and {adj/b, -adj·a/b²} in reverse — so the
// tangent is compared to a tolerance a few ULP wide rather than exactly.
#define EXPECT_TANGENT_MATCHES_REVERSE(expr_, xv_)                             \
  do {                                                                         \
    const auto e_ = (expr_);                                                   \
    const std::array<double, 1> pt_{xv_};                                      \
    const auto t_ = e_.template eval_with_tangent<"x">(xv_);                   \
    EXPECT_DOUBLE_EQ(t_.value(), e_.eval(xv_));                                \
    EXPECT_NEAR(t_.deriv(), Equation{e_}.gradient(pt_)[0], 1e-12);             \
  } while (0)

TEST(ForwardTangentSweep, LeafSeeding) {
  auto x = var<"x">;
  auto y = var<"y">;

  const auto tx = x.eval_with_tangent<"x">(2.0);
  EXPECT_DOUBLE_EQ(tx.value(), 2.0);
  EXPECT_DOUBLE_EQ(tx.deriv(), 1.0);

  // Every other variable carries 0, including one whose name differs only in
  // length (FixedString comparison across sizes).
  const auto ty = y.eval_with_tangent<"x">(3.0);
  EXPECT_DOUBLE_EQ(ty.value(), 3.0);
  EXPECT_DOUBLE_EQ(ty.deriv(), 0.0);
  EXPECT_DOUBLE_EQ(x.eval_with_tangent<"xx">(2.0).deriv(), 0.0);

  const auto tc = constant(5.0).eval_with_tangent<"x">(); // no symbols
  EXPECT_DOUBLE_EQ(tc.value(), 5.0);
  EXPECT_DOUBLE_EQ(tc.deriv(), 0.0);
}
TEST(ForwardTangentSweep, ArithmeticRules) {
  double x0 = 1.7;
  auto x = var_of<"x">(x0);

  EXPECT_TANGENT_MATCHES_REVERSE(x + x * x, 0.7); // sum + product
  EXPECT_TANGENT_MATCHES_REVERSE(x - 3.0 * x * x,
                                 0.7); // negate (a - b => a + -b)
  EXPECT_TANGENT_MATCHES_REVERSE((x + 1.0) / x, 0.7); // quotient
  EXPECT_TANGENT_MATCHES_REVERSE(-(x / (x * x + 2.0)), 0.7);

  // Cross-check the same sweep against the symbolic derivative tree, which is a
  // third independent path (derivative() is a total derivative, so this is only
  // meaningful for a single-variable expression).
  const auto e = (x * x + 1.0) / (x + 2.0);
  EXPECT_DOUBLE_EQ(e.eval_with_tangent<"x">(x0).deriv(),
                   e.derivative().eval(x0));
}
TEST(ForwardTangentSweep, UnaryMathRules) {
  // One probe per op in the DDX_UNARY_MATH_OP macro, plus AbsOp's hand-written
  // forward().  Arguments stay inside each function's domain.
  EXPECT_TANGENT_MATCHES_REVERSE(sin(var<"x">), 0.6);
  EXPECT_TANGENT_MATCHES_REVERSE(cos(var<"x">), 0.6);
  EXPECT_TANGENT_MATCHES_REVERSE(tan(var<"x">), 0.6);
  EXPECT_TANGENT_MATCHES_REVERSE(exp(var<"x">), 0.6);
  EXPECT_TANGENT_MATCHES_REVERSE(log(var<"x">), 1.6);
  EXPECT_TANGENT_MATCHES_REVERSE(log10(var<"x">), 1.6);
  EXPECT_TANGENT_MATCHES_REVERSE(sqrt(var<"x">), 1.6);
  EXPECT_TANGENT_MATCHES_REVERSE(cbrt(var<"x">), 1.6);
  EXPECT_TANGENT_MATCHES_REVERSE(asin(var<"x">), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(acos(var<"x">), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(atan(var<"x">), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(sinh(var<"x">), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(cosh(var<"x">), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(tanh(var<"x">), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(asinh(var<"x">), 0.7);
  EXPECT_TANGENT_MATCHES_REVERSE(acosh(var<"x">), 2.0);
  EXPECT_TANGENT_MATCHES_REVERSE(atanh(var<"x">), 0.3);
  EXPECT_TANGENT_MATCHES_REVERSE(erf(var<"x">), 0.5);

  // abs() on both sides of the kink: sign(u)·u'.
  EXPECT_DOUBLE_EQ(abs(var<"x">).eval_with_tangent<"x">(2.0).deriv(), 1.0);
  EXPECT_DOUBLE_EQ(abs(var<"x">).eval_with_tangent<"x">(-2.0).deriv(), -1.0);

  // Composite argument: the chain rule must fire at every level.
  EXPECT_TANGENT_MATCHES_REVERSE(log(sin(var<"x">) + 2.0), 0.9);
}
TEST(ForwardTangentSweep, BinaryMathRules) {
  double x0 = 3.0, y0 = 4.0;
  auto x = var_of<"x">(x0);
  auto y = var_of<"y">(y0);

  // Symbols sort alphabetically, so gradient slot 0 is x and slot 1 is y.
  const auto check = [x0, y0](const auto &e, double dx, double dy) {
    EXPECT_DOUBLE_EQ(e.template eval_with_tangent<"x">(x0, y0).value(),
                     e.eval(x0, y0));
    EXPECT_DOUBLE_EQ(e.template eval_with_tangent<"x">(x0, y0).deriv(), dx);
    EXPECT_DOUBLE_EQ(e.template eval_with_tangent<"y">(x0, y0).deriv(), dy);
  };

  check(hypot(x, y), x0 / 5.0, y0 / 5.0);
  check(pow(x, y), y0 * std::pow(x0, y0 - 1.0),
        std::pow(x0, y0) * std::log(x0));
  // atan2(y, x) — first operand is the numerator.
  const double q = x0 * x0 + y0 * y0;
  check(atan2(y, x), -y0 / q, x0 / q);
  // max/min pass the selected branch's tangent through untouched.
  check(max(x, y), 0.0, 1.0);
  check(min(x, y), 1.0, 0.0);
}
TEST(ForwardTangentSweep, MultiVariableMatchesReverseMode) {
  auto x = var<"x">;
  auto y = var<"y">;

  // Deep composite: shared subexpressions, nested transcendentals, a quotient
  // and a binary math op, so most forward rules participate in one tree.
  const auto e = sin(x * y) * exp(x + y) + log(y * y + 1.0) / (x + 2.0) -
                 hypot(x, y) * tanh(x);

  const auto g = Equation{e}.gradient(0.8, 1.3);
  const auto tx = e.eval_with_tangent<"x">(0.8, 1.3);
  const auto ty = e.eval_with_tangent<"y">(0.8, 1.3);

  EXPECT_DOUBLE_EQ(tx.value(), e.eval(0.8, 1.3));
  EXPECT_DOUBLE_EQ(ty.value(), e.eval(0.8, 1.3));
  EXPECT_NEAR(tx.deriv(), g[0], 1e-12);
  EXPECT_NEAR(ty.deriv(), g[1], 1e-12);

  EXPECT_DOUBLE_EQ(e.eval_with_tangent<"z">(0.8, 1.3).deriv(), 0.0);
}
TEST(ForwardTangentSweep, IsConstexpr) {
  // The sweep runs entirely at compile time (no std:: transcendentals here, so
  // this holds on every toolchain).
  constexpr auto x = var<"x">;
  constexpr auto e = x * x + 3.0 * x;
  constexpr auto t = e.eval_with_tangent<"x">(2.0);
  static_assert(t.value() == 10.0, "x² + 3x at x = 2");
  static_assert(t.deriv() == 7.0, "2x + 3 at x = 2");
  EXPECT_DOUBLE_EQ(t.deriv(), 7.0);
}
// A frozen symbol is a constant: same value lookup, zero derivative.  All three
// engines have to say so — the seeded sweep that the drivers use, the forward
// tangent sweep, and the symbolic derivative tree.
TEST(FrozenVariable, EveryEngineSeesZeroDerivative) {
  auto e =
      make_const_variable<ddx::impl::FixedString{"x"}>(var<"x"> * var<"y">);
  const std::array<double, 2> pt{2.0, 3.0};

  EXPECT_DOUBLE_EQ(e.eval(2.0, 3.0), 6.0);

  const auto g = Equation{e}.template derivative_tensor<1>(pt);
  EXPECT_DOUBLE_EQ(g.data()[0], 0.0); // d/dx — frozen
  EXPECT_DOUBLE_EQ(g.data()[1], 2.0); // d/dy — still live

  EXPECT_DOUBLE_EQ(e.eval_with_tangent<"x">(2.0, 3.0).deriv(), 0.0);
  EXPECT_DOUBLE_EQ(e.eval_with_tangent<"y">(2.0, 3.0).deriv(), 2.0);
}

#undef EXPECT_TANGENT_MATCHES_REVERSE

// ===========================================================================
// ValueMap / Bound — root-held values.
//
// The point lives in one slot per *variable* rather than one per leaf
// occurrence.  These tests pin the behaviour that the stateless-leaf change
// depends on: canonical ordering, symbol lookup, and bit-identity with the
// existing leaf-storage eval() path.
// ===========================================================================

// derivative_tensor<1> has to support every unary math function at any arity:
// each one below must *instantiate* at 3 variables and agree with reverse mode.
TEST(ForwardGradient, CoversEveryUnaryMathFunction) {
  const std::array<double, 3> pt{0.6, 0.4, 1.3};
  auto y = var<"y">;
  auto z = var<"z">;

}
