#include "tests_common.hpp"


// ===========================================================================
// New math functions (parity with autodiff): log10, cbrt, asinh, acosh,
// atanh, erf (unary) and pow, atan2, hypot, min, max (binary), exercised
// across all three mechanisms — expression templates, lazy dual, Taylor.
// ===========================================================================

namespace {
constexpr double kLn10 = std::numbers::ln10;
constexpr double k2OverSqrtPi = 2.0 * std::numbers::inv_sqrtpi;
} // namespace

// ---- Expression templates: value + symbolic first derivative --------------
TEST(NewMathFunctions, ExprUnaryValueAndDerivative) {
  {
    double x0 = 3.0;
    auto x = PV(x0, "x");
    EXPECT_DOUBLE_EQ(log10(x).eval(x0), std::log10(x0));
    EXPECT_DOUBLE_EQ(log10(x).derivative().eval(x0), 1.0 / (x0 * kLn10));
  }
  {
    double x0 = 2.0;
    auto x = PV(x0, "x");
    EXPECT_DOUBLE_EQ(cbrt(x).eval(x0), std::cbrt(x0));
    EXPECT_DOUBLE_EQ(cbrt(x).derivative().eval(x0),
                     1.0 / (3.0 * std::cbrt(x0) * std::cbrt(x0)));
  }
  {
    double x0 = 0.7;
    auto x = PV(x0, "x");
    EXPECT_DOUBLE_EQ(asinh(x).eval(x0), std::asinh(x0));
    EXPECT_DOUBLE_EQ(asinh(x).derivative().eval(x0),
                     1.0 / std::sqrt(x0 * x0 + 1.0));
  }
  {
    double x0 = 2.0;
    auto x = PV(x0, "x");
    EXPECT_DOUBLE_EQ(acosh(x).eval(x0), std::acosh(x0));
    EXPECT_DOUBLE_EQ(acosh(x).derivative().eval(x0),
                     1.0 / std::sqrt(x0 * x0 - 1.0));
  }
  {
    double x0 = 0.3;
    auto x = PV(x0, "x");
    EXPECT_DOUBLE_EQ(atanh(x).eval(x0), std::atanh(x0));
    EXPECT_DOUBLE_EQ(atanh(x).derivative().eval(x0), 1.0 / (1.0 - x0 * x0));
  }
  {
    double x0 = 0.5;
    auto x = PV(x0, "x");
    EXPECT_DOUBLE_EQ(erf(x).eval(x0), std::erf(x0));
    EXPECT_DOUBLE_EQ(erf(x).derivative().eval(x0),
                     k2OverSqrtPi * std::exp(-x0 * x0));
  }
}

TEST(NewMathFunctions, ExprBinaryValue) {
  auto x = PV(3.0, "x");
  auto y = PV(4.0, "y");
  EXPECT_DOUBLE_EQ(hypot(x, y).eval(3.0, 4.0), 5.0);
  EXPECT_DOUBLE_EQ(pow(x, y).eval(3.0, 4.0), std::pow(3.0, 4.0));
  EXPECT_DOUBLE_EQ(atan2(y, x).eval(3.0, 4.0), std::atan2(4.0, 3.0));
  EXPECT_DOUBLE_EQ(max(x, y).eval(3.0, 4.0), 4.0);
  EXPECT_DOUBLE_EQ(min(x, y).eval(3.0, 4.0), 3.0);
  // scalar-promotion overloads
  EXPECT_DOUBLE_EQ(hypot(x, 4.0).eval(3.0), 5.0);
  EXPECT_DOUBLE_EQ(pow(x, 2.0).eval(3.0), 9.0);
  EXPECT_DOUBLE_EQ(pow(2.0, x).eval(3.0), 8.0);
}

// ---- Reverse mode (partials per variable) ---------------------------------
TEST(NewMathFunctions, ReverseUnary) {
  EXPECT_DOUBLE_EQ(Equation{log10(PV(3.0, "x"))}.gradient(3.0)[0],
                   1.0 / (3.0 * kLn10));
  EXPECT_DOUBLE_EQ(Equation{cbrt(PV(2.0, "x"))}.gradient(2.0)[0],
                   1.0 / (3.0 * std::cbrt(2.0) * std::cbrt(2.0)));
  EXPECT_DOUBLE_EQ(Equation{asinh(PV(0.7, "x"))}.gradient(0.7)[0],
                   1.0 / std::sqrt(0.49 + 1.0));
  EXPECT_DOUBLE_EQ(Equation{acosh(PV(2.0, "x"))}.gradient(2.0)[0],
                   1.0 / std::sqrt(4.0 - 1.0));
  EXPECT_DOUBLE_EQ(Equation{atanh(PV(0.3, "x"))}.gradient(0.3)[0],
                   1.0 / (1.0 - 0.09));
  EXPECT_DOUBLE_EQ(Equation{erf(PV(0.5, "x"))}.gradient(0.5)[0],
                   k2OverSqrtPi * std::exp(-0.25));
}

TEST(NewMathFunctions, ReverseBinaryPartials) {
  {
    auto x = PV(3.0, "x");
    auto y = PV(4.0, "y");
    auto g = Equation{hypot(x, y)}.gradient(3.0, 4.0);
    EXPECT_DOUBLE_EQ(g[0], 3.0 / 5.0);
    EXPECT_DOUBLE_EQ(g[1], 4.0 / 5.0);
  }
  {
    auto x = PV(2.0, "x");
    auto y = PV(3.0, "y");
    auto g = Equation{pow(x, y)}.gradient(2.0, 3.0);
    EXPECT_DOUBLE_EQ(g[0], 3.0 * std::pow(2.0, 2.0));      // y*x^(y-1)
    EXPECT_DOUBLE_EQ(g[1], std::pow(2.0, 3.0) * std::log(2.0)); // x^y*ln x
  }
  {
    // atan2(y, x): lhs is numerator y, rhs is x. The gradient is ordered by
    // symbol (x at index 0, y at index 1).
    auto y = PV(1.0, "y");
    auto x = PV(2.0, "x");
    auto g = Equation{atan2(y, x)}.gradient(2.0, 1.0);
    const double q = 1.0 + 4.0;
    EXPECT_DOUBLE_EQ(g[0], -1.0 / q); // d/dx = -y/q
    EXPECT_DOUBLE_EQ(g[1], 2.0 / q);  // d/dy =  x/q
  }
}

TEST(NewMathFunctions, ReverseMaxMinSelectsBranch) {
  auto x = PV(3.0, "x");
  auto y = PV(4.0, "y");
  auto gmax = Equation{max(x, y)}.gradient(3.0, 4.0);
  EXPECT_DOUBLE_EQ(gmax[0], 0.0);
  EXPECT_DOUBLE_EQ(gmax[1], 1.0);
  auto gmin = Equation{min(x, y)}.gradient(3.0, 4.0);
  EXPECT_DOUBLE_EQ(gmin[0], 1.0);
  EXPECT_DOUBLE_EQ(gmin[1], 0.0);
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

  // 3-argument hypot
  Dual<double> a{1.0, 1.0}, b{2.0, 0.0}, c{2.0, 0.0};
  Dual<double> h = hypot(a, b, c);
  EXPECT_DOUBLE_EQ(h.get<0>(), 3.0);          // sqrt(1+4+4)
  EXPECT_DOUBLE_EQ(h.get<1>(), 1.0 / 3.0);    // a/h
}

// ---- Taylor higher-order (value coeff drives expr + Taylor together) -------
TEST(NewMathFunctions, TaylorFirstAndSecondDerivative) {
  // First derivatives via order-1 univariate_derivative.
  EXPECT_NEAR(Equation{asinh(PV(0.7, "x"))}.template univariate_derivative<1>(0.7),
              1.0 / std::sqrt(0.49 + 1.0), 1e-12);
  EXPECT_NEAR(Equation{erf(PV(0.5, "x"))}.template univariate_derivative<1>(0.5),
              k2OverSqrtPi * std::exp(-0.25), 1e-12);
  EXPECT_NEAR(Equation{cbrt(PV(2.0, "x"))}.template univariate_derivative<1>(2.0),
              1.0 / (3.0 * std::cbrt(2.0) * std::cbrt(2.0)), 1e-12);

  // Second derivatives vs analytic.
  {
    double x0 = 0.7; // asinh'' = -x/(1+x²)^{3/2}
    EXPECT_NEAR(Equation{asinh(PV(x0, "x"))}.template univariate_derivative<2>(x0),
                -x0 / std::pow(1.0 + x0 * x0, 1.5), 1e-10);
  }
  {
    double x0 = 0.5; // erf'' = -2x * 2/√π * e^{-x²}
    EXPECT_NEAR(Equation{erf(PV(x0, "x"))}.template univariate_derivative<2>(x0),
                -2.0 * x0 * k2OverSqrtPi * std::exp(-x0 * x0), 1e-10);
  }
  {
    double x0 = 0.3; // atanh'' = 2x/(1-x²)²
    EXPECT_NEAR(Equation{atanh(PV(x0, "x"))}.template univariate_derivative<2>(x0),
                2.0 * x0 / std::pow(1.0 - x0 * x0, 2.0), 1e-10);
  }
  {
    double x0 = 3.0; // log10'' = -1/(x² ln10)
    EXPECT_NEAR(Equation{log10(PV(x0, "x"))}.template univariate_derivative<2>(x0),
                -1.0 / (x0 * x0 * kLn10), 1e-10);
  }
  {
    double x0 = 2.0; // cbrt'' = -2/9 x^{-5/3}
    EXPECT_NEAR(Equation{cbrt(PV(x0, "x"))}.template univariate_derivative<2>(x0),
                -2.0 / 9.0 * std::pow(x0, -5.0 / 3.0), 1e-10);
  }
  {
    double x0 = 2.0; // acosh'' = -x/(x²-1)^{3/2}
    EXPECT_NEAR(Equation{acosh(PV(x0, "x"))}.template univariate_derivative<2>(x0),
                -x0 / std::pow(x0 * x0 - 1.0, 1.5), 1e-10);
  }
}

TEST(NewMathFunctions, TaylorBinaryUnivariate) {
  // pow(x, 3): f'=3x², f''=6x
  EXPECT_NEAR(Equation{pow(PV(2.0, "x"), 3.0)}.template univariate_derivative<1>(2.0), 12.0, 1e-10);
  EXPECT_NEAR(Equation{pow(PV(2.0, "x"), 3.0)}.template univariate_derivative<2>(2.0), 12.0, 1e-10);
  // hypot(x, 2): f = sqrt(x²+4); f' = x/sqrt(x²+4)
  {
    double x0 = 3.0, h = std::sqrt(x0 * x0 + 4.0);
    EXPECT_NEAR(Equation{hypot(PV(x0, "x"), 2.0)}.template univariate_derivative<1>(x0), x0 / h,
                1e-10);
  }
  // atan2(x, 2): d/dx atan2(x,2) = 2/(4+x²)
  {
    double x0 = 2.0;
    EXPECT_NEAR(Equation{atan2(PV(x0, "x"), 2.0)}.template univariate_derivative<1>(x0),
                2.0 / (4.0 + x0 * x0), 1e-10);
  }
}

// Composite arguments: the asin/atan/asinh/acosh/atanh recurrences must use the
// argument's actual derivative series (u'), not assume the bare seed (u'=1).
TEST(NewMathFunctions, TaylorCompositeArguments) {
  // atan(x²): f' = 2x/(1+x⁴); f'' = (2 - 6x⁴)/(1+x⁴)².
  {
    double x0 = 0.6, x4 = x0 * x0 * x0 * x0, d = 1.0 + x4;
    auto x = PV(x0, "x");
    EXPECT_NEAR(Equation{atan(x * x)}.template univariate_derivative<1>(x0), 2.0 * x0 / d, 1e-10);
    EXPECT_NEAR(Equation{atan(x * x)}.template univariate_derivative<2>(x0), (2.0 - 6.0 * x4) / (d * d),
                1e-10);
  }
  // asin(x/2): f' = a/√(1-a²x²), f'' = a³x(1-a²x²)^{-3/2}, a = 1/2.
  {
    double x0 = 1.0, a = 0.5, r = 1.0 - a * a * x0 * x0;
    auto x = PV(x0, "x");
    EXPECT_NEAR(Equation{asin(0.5 * x)}.template univariate_derivative<1>(x0), a / std::sqrt(r),
                1e-10);
    EXPECT_NEAR(Equation{asin(0.5 * x)}.template univariate_derivative<2>(x0),
                a * a * a * x0 * std::pow(r, -1.5), 1e-10);
  }
  // asinh(2x): f'' = -a³x(1+a²x²)^{-3/2}, a = 2.
  {
    double x0 = 0.3, a = 2.0, r = 1.0 + a * a * x0 * x0;
    auto x = PV(x0, "x");
    EXPECT_NEAR(Equation{asinh(2.0 * x)}.template univariate_derivative<1>(x0), a / std::sqrt(r),
                1e-10);
    EXPECT_NEAR(Equation{asinh(2.0 * x)}.template univariate_derivative<2>(x0),
                -a * a * a * x0 * std::pow(r, -1.5), 1e-10);
  }
  // Identity: atan2(sin x, cos x) == x on (-π, π) ⇒ f'=1, f''=0, f'''=0.
  {
    auto x = PV(0.7, "x");
    EXPECT_NEAR(Equation{atan2(sin(x), cos(x))}.template univariate_derivative<1>(0.7), 1.0, 1e-10);
    EXPECT_NEAR(Equation{atan2(sin(x), cos(x))}.template univariate_derivative<2>(0.7), 0.0, 1e-10);
    EXPECT_NEAR(Equation{atan2(sin(x), cos(x))}.template univariate_derivative<3>(0.7), 0.0, 1e-10);
  }
  // pow(sin x, 2) = sin²x ⇒ f' = sin(2x), f'' = 2cos(2x) (powser composite).
  {
    double x0 = 0.5;
    auto x = PV(x0, "x");
    EXPECT_NEAR(Equation{pow(sin(x), 2.0)}.template univariate_derivative<1>(x0), std::sin(2.0 * x0),
                1e-10);
    EXPECT_NEAR(Equation{pow(sin(x), 2.0)}.template univariate_derivative<2>(x0),
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
    EXPECT_DOUBLE_EQ(t_.value(), e_.eval(xv_));                                  \
    EXPECT_NEAR(t_.deriv(), Equation{e_}.gradient(pt_)[0], 1e-12);               \
  } while (0)

TEST(ForwardTangentSweep, LeafSeeding) {
  auto x = PV(2.0, "x");
  auto y = PV(3.0, "y");

  // The seeded variable carries tangent 1 ...
  const auto tx = x.eval_with_tangent<"x">(2.0);
  EXPECT_DOUBLE_EQ(tx.value(), 2.0);
  EXPECT_DOUBLE_EQ(tx.deriv(), 1.0);

  // ... every other variable carries 0, including one whose name differs only
  // in length (FixedString comparison across sizes).
  const auto ty = y.eval_with_tangent<"x">(3.0);
  EXPECT_DOUBLE_EQ(ty.value(), 3.0);
  EXPECT_DOUBLE_EQ(ty.deriv(), 0.0);
  EXPECT_DOUBLE_EQ(x.eval_with_tangent<"xx">(2.0).deriv(), 0.0);

  // Constants are always zero-tangent.
  const auto tc = PC(5.0).eval_with_tangent<"x">();  // no symbols
  EXPECT_DOUBLE_EQ(tc.value(), 5.0);
  EXPECT_DOUBLE_EQ(tc.deriv(), 0.0);
}

TEST(ForwardTangentSweep, ArithmeticRules) {
  double x0 = 1.7;
  auto x = PV(x0, "x");

  EXPECT_TANGENT_MATCHES_REVERSE(x + x * x, 0.7);         // sum + product
  EXPECT_TANGENT_MATCHES_REVERSE(x - 3.0 * x * x, 0.7);   // negate (a - b => a + -b)
  EXPECT_TANGENT_MATCHES_REVERSE((x + 1.0) / x, 0.7);     // quotient
  EXPECT_TANGENT_MATCHES_REVERSE(-(x / (x * x + 2.0)), 0.7);

  // Cross-check the same sweep against the symbolic derivative tree, which is a
  // third independent path (derivative() is a total derivative, so this is only
  // meaningful for a single-variable expression).
  const auto e = (x * x + 1.0) / (x + 2.0);
  EXPECT_DOUBLE_EQ(e.eval_with_tangent<"x">(x0).deriv(), e.derivative().eval(x0));
}

TEST(ForwardTangentSweep, UnaryMathRules) {
  // One probe per op in the DDX_UNARY_MATH_OP macro, plus AbsOp's hand-written
  // forward().  Arguments stay inside each function's domain.
  EXPECT_TANGENT_MATCHES_REVERSE(sin(PV(0.6, "x")), 0.6);
  EXPECT_TANGENT_MATCHES_REVERSE(cos(PV(0.6, "x")), 0.6);
  EXPECT_TANGENT_MATCHES_REVERSE(tan(PV(0.6, "x")), 0.6);
  EXPECT_TANGENT_MATCHES_REVERSE(exp(PV(0.6, "x")), 0.6);
  EXPECT_TANGENT_MATCHES_REVERSE(log(PV(1.6, "x")), 1.6);
  EXPECT_TANGENT_MATCHES_REVERSE(log10(PV(1.6, "x")), 1.6);
  EXPECT_TANGENT_MATCHES_REVERSE(sqrt(PV(1.6, "x")), 1.6);
  EXPECT_TANGENT_MATCHES_REVERSE(cbrt(PV(1.6, "x")), 1.6);
  EXPECT_TANGENT_MATCHES_REVERSE(asin(PV(0.4, "x")), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(acos(PV(0.4, "x")), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(atan(PV(0.4, "x")), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(sinh(PV(0.4, "x")), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(cosh(PV(0.4, "x")), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(tanh(PV(0.4, "x")), 0.4);
  EXPECT_TANGENT_MATCHES_REVERSE(asinh(PV(0.7, "x")), 0.7);
  EXPECT_TANGENT_MATCHES_REVERSE(acosh(PV(2.0, "x")), 2.0);
  EXPECT_TANGENT_MATCHES_REVERSE(atanh(PV(0.3, "x")), 0.3);
  EXPECT_TANGENT_MATCHES_REVERSE(erf(PV(0.5, "x")), 0.5);

  // abs() on both sides of the kink: sign(u)·u'.
  EXPECT_DOUBLE_EQ(abs(PV(2.0, "x")).eval_with_tangent<"x">(2.0).deriv(), 1.0);
  EXPECT_DOUBLE_EQ(abs(PV(-2.0, "x")).eval_with_tangent<"x">(-2.0).deriv(), -1.0);

  // Composite argument: the chain rule must fire at every level.
  EXPECT_TANGENT_MATCHES_REVERSE(log(sin(PV(0.9, "x")) + 2.0), 0.9);
}

TEST(ForwardTangentSweep, BinaryMathRules) {
  double x0 = 3.0, y0 = 4.0;
  auto x = PV(x0, "x");
  auto y = PV(y0, "y");

  // Symbols sort alphabetically, so gradient slot 0 is x and slot 1 is y.
  const auto check = [x0, y0](const auto &e, double dx, double dy) {
    EXPECT_DOUBLE_EQ(e.template eval_with_tangent<"x">(x0, y0).value(),
                     e.eval(x0, y0));
    EXPECT_DOUBLE_EQ(e.template eval_with_tangent<"x">(x0, y0).deriv(), dx);
    EXPECT_DOUBLE_EQ(e.template eval_with_tangent<"y">(x0, y0).deriv(), dy);
  };

  // hypot: d/dx = x/h, d/dy = y/h.
  check(hypot(x, y), x0 / 5.0, y0 / 5.0);
  // pow(x, y): d/dx = y·x^(y-1), d/dy = x^y·ln x.
  check(pow(x, y), y0 * std::pow(x0, y0 - 1.0), std::pow(x0, y0) * std::log(x0));
  // atan2(y, x) — first operand is the numerator.
  const double q = x0 * x0 + y0 * y0;
  check(atan2(y, x), -y0 / q, x0 / q);
  // max/min pass the selected branch's tangent through untouched.
  check(max(x, y), 0.0, 1.0);
  check(min(x, y), 1.0, 0.0);
}

TEST(ForwardTangentSweep, MultiVariableMatchesReverseMode) {
  auto x = PV(0.8, "x");
  auto y = PV(1.3, "y");

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

  // A symbol not present in the tree differentiates to zero everywhere.
  EXPECT_DOUBLE_EQ(e.eval_with_tangent<"z">(0.8, 1.3).deriv(), 0.0);
}

TEST(ForwardTangentSweep, IsConstexpr) {
  // The sweep runs entirely at compile time (no std:: transcendentals here, so
  // this holds on every toolchain).
  constexpr auto x = PV(2.0, "x");
  constexpr auto e = x * x + 3.0 * x;
  constexpr auto t = e.eval_with_tangent<"x">(2.0);
  static_assert(t.value() == 10.0, "x² + 3x at x = 2");
  static_assert(t.deriv() == 7.0, "2x + 3 at x = 2");
  EXPECT_DOUBLE_EQ(t.deriv(), 7.0);
}

// A frozen symbol is a constant: same value lookup, zero derivative.  All three
// engines have to say so — the seeded sweep that the drivers use, the forward
// tangent sweep, and the symbolic derivative tree.  Before eval_seeded grew its
// Frozen guard the first of the three reported the *unfrozen* derivative, so
// derivative_tensor and eval_with_tangent disagreed on the same expression.
TEST(FrozenVariable, EveryEngineSeesZeroDerivative) {
  auto e = make_const_variable<ddx::impl::FixedString{"x"}>(PV(2.0, "x") *
                                                       PV(3.0, "y"));
  const std::array<double, 2> pt{2.0, 3.0};

  // Value is untouched by freezing.
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

TEST(ValueMapTest, SlotsAreInCanonicalOrderNotArgumentOrder) {
  // Canonical order is alphabetical, so {w,x,y} regardless of how they arrive.
  const auto m = values(named<"y">(3.0), named<"w">(1.0), named<"x">(2.0));
  EXPECT_EQ(decltype(m)::arity, 3u);
  EXPECT_DOUBLE_EQ(m.slots[0], 1.0); // w
  EXPECT_DOUBLE_EQ(m.slots[1], 2.0); // x
  EXPECT_DOUBLE_EQ(m.slots[2], 3.0); // y
  EXPECT_DOUBLE_EQ(m.get<"w">(), 1.0);
  EXPECT_DOUBLE_EQ(m.get<"x">(), 2.0);
  EXPECT_DOUBLE_EQ(m.get<"y">(), 3.0);
}

TEST(ValueMapTest, SetUpdatesTheNamedSlot) {
  auto m = values(named<"x">(2.0), named<"y">(3.0));
  m.set<"x">(7.0);
  EXPECT_DOUBLE_EQ(m.get<"x">(), 7.0);
  EXPECT_DOUBLE_EQ(m.get<"y">(), 3.0);
}

// ---------------------------------------------------------------------------
// operator[] — the subscript spelling of get/set.  It is an alias, so what
// these pin is that it stays one: same slot, same errors, same ownership
// rules, plus the assignment form that get/set cannot express.
// ---------------------------------------------------------------------------

TEST(ValueMapTest, SubscriptReadsTheSameSlotAsGet) {
  const auto m = values(named<"y">(3.0), named<"w">(1.0), named<"x">(2.0));
  // Both spellings of the key, since sym<"x"> is what "x"_s expands to.
  EXPECT_DOUBLE_EQ(m[sym<"w">], m.get<"w">());
  EXPECT_DOUBLE_EQ(m["x"_s], 2.0);
  EXPECT_DOUBLE_EQ(m["y"_s], 3.0);
  static_assert(std::is_same_v<decltype(sym<"x">), const symbol_type<
                                                      FixedString{"x"}>>,
                R"("x"_s and sym<"x"> must name the same key type)");
}

TEST(ValueMapTest, SubscriptAssignsIntoTheNamedSlot) {
  auto m = values(named<"x">(2.0), named<"y">(3.0));
  m["x"_s] = 7.0;
  EXPECT_DOUBLE_EQ(m.get<"x">(), 7.0); // the point of having a subscript
  EXPECT_DOUBLE_EQ(m.get<"y">(), 3.0);
  m[sym<"y">] += 1.0;                // it is a real reference, not a proxy
  EXPECT_DOUBLE_EQ(m.slots[1], 4.0); // canonical order: {x,y}
  EXPECT_DOUBLE_EQ(m.slots[0], 7.0);
}

TEST(ValueMapTest, SubscriptOwnershipMatchesGet) {
  const auto m = values(named<"x">(2.0), named<"y">(3.0));
  auto mut = m;
  static_assert(std::is_same_v<decltype(m["x"_s]), const double &>,
                "subscript on a const lvalue map borrows");
  static_assert(std::is_same_v<decltype(mut["x"_s]), double &>,
                "subscript on a mutable lvalue map hands out the slot");
  static_assert(std::is_same_v<decltype(values(named<"x">(2.0))["x"_s]), double>,
                "subscript on a temporary map must return by value");
  EXPECT_DOUBLE_EQ(values(named<"x">(2.0))["x"_s], 2.0);

  // get<S>() is the same body reached by a different spelling, so it owns its
  // result exactly as the subscript does - on either side of the
  // __cpp_explicit_this_parameter split.
  static_assert(std::is_same_v<decltype(m.get<"x">()), const double &>);
  static_assert(std::is_same_v<decltype(mut.get<"x">()), double &>);
  static_assert(std::is_same_v<decltype(values(named<"x">(2.0)).get<"x">()),
                               double>,
                "get on a temporary map must return by value");
  EXPECT_DOUBLE_EQ(values(named<"x">(2.0)).get<"x">(), 2.0);
}

TEST(ValueMapTest, SubscriptIsConstexpr) {
  constexpr double folded = [] {
    auto m = values(named<"x">(2.0), named<"y">(3.0));
    m["x"_s] = 7.0;
    return m["x"_s] + m["y"_s];
  }();
  static_assert(folded == 10.0, "read and write both fold at compile time");
  EXPECT_DOUBLE_EQ(folded, 10.0);
}

TEST(BoundTest, EvalMatchesLeafStorageBitForBit) {
  // The seeded path and the leaf-storage path must agree exactly - this is a
  // storage change, not a numerics change.
  auto x = PV(0.8, "x");
  auto y = PV(1.3, "y");
  const auto e = sin(x * y) * exp(x + y) + log(y * y + 1.0) / (x + 2.0) -
                 hypot(x, y) * tanh(x);

  const auto b = bind(e, named<"x">(0.8), named<"y">(1.3));
  EXPECT_EQ(b.eval(), e.eval(0.8, 1.3)); // bitwise, not EXPECT_DOUBLE_EQ
}

TEST(BoundTest, ArgumentOrderIsIrrelevantAndSupersetsAreAllowed) {
  auto x = PV(2.0, "x");
  auto y = PV(3.0, "y");
  const auto e = (x + y) * x; // symbols {x,y}

  EXPECT_DOUBLE_EQ(bind(e, named<"x">(2.0), named<"y">(3.0)).eval(), 10.0);
  EXPECT_DOUBLE_EQ(bind(e, named<"y">(3.0), named<"x">(2.0)).eval(), 10.0);

  // A map carrying extra symbols still binds; the permutation picks the ones
  // the expression actually uses.
  const auto wide =
      values(named<"z">(9.0), named<"x">(2.0), named<"y">(3.0), named<"a">(9.0));
  EXPECT_DOUBLE_EQ(bind(e, wide).eval(), 10.0);
}

TEST(BoundTest, SetChangesTheResult) {
  auto x = PV(2.0, "x");
  auto y = PV(3.0, "y");
  auto b = bind((x + y) * x, named<"x">(2.0), named<"y">(3.0));
  EXPECT_DOUBLE_EQ(b.eval(), 10.0);
  b.set<"x">(4.0);
  EXPECT_DOUBLE_EQ(b.eval(), 28.0); // (4+3)*4
}

TEST(BoundTest, SubscriptForwardsToTheMap) {
  auto x = PV(2.0, "x");
  auto y = PV(3.0, "y");
  auto b = bind((x + y) * x, named<"x">(2.0), named<"y">(3.0));
  EXPECT_DOUBLE_EQ(b["x"_s], 2.0);

  b["x"_s] = 4.0; // must move the same slot set<"x"> does
  EXPECT_DOUBLE_EQ(b.eval(), 28.0);
  EXPECT_DOUBLE_EQ(b.get<"x">(), 4.0);
  EXPECT_DOUBLE_EQ(b.map.get<"x">(), 4.0);

  const auto &cb = b;
  static_assert(std::is_same_v<decltype(cb["x"_s]), const double &>,
                "subscript on a const lvalue Bound borrows");
  static_assert(
      std::is_same_v<decltype(bind(x * y, named<"x">(1.0), named<"y">(1.0))
                                  ["x"_s]),
                     double>,
      "subscript on a temporary Bound must return by value");
  EXPECT_DOUBLE_EQ(bind(x * y, named<"x">(6.0), named<"y">(1.0))["x"_s], 6.0);

  // Same body, other spelling - see the note in ValueMapTest.
  static_assert(std::is_same_v<decltype(cb.get<"x">()), const double &>);
  static_assert(std::is_same_v<decltype(b.get<"x">()), double &>);
  static_assert(
      std::is_same_v<
          decltype(bind(x * y, named<"x">(1.0), named<"y">(1.0)).get<"x">()),
          double>,
      "get on a temporary Bound must return by value");
  EXPECT_DOUBLE_EQ(bind(x * y, named<"x">(6.0), named<"y">(1.0)).get<"x">(),
                   6.0);
}

TEST(BoundTest, IsConstexpr) {
  // constexpr constrains the function, not the caller: the same eval() serves
  // compile-time and runtime values.
  constexpr auto x = PV(1.0, "x");
  constexpr auto y = PV(1.0, "y");
  constexpr auto b = bind(x * x + y, named<"x">(3.0), named<"y">(4.0));
  static_assert(b.eval() == 13.0, "x*x + y at (3,4)");
  EXPECT_DOUBLE_EQ(b.eval(), 13.0);
}

TEST(PositionalEvalTest, SymbolOrderIsAlphabeticalNotSourceOrder) {
  auto y = PV(1.0, "y");
  auto x = PV(1.0, "x");
  const auto e = y * x; // written y-first
  constexpr auto order = symbol_order<decltype(e)>();
  static_assert(order.size() == 2);
  EXPECT_EQ(order[0], "x"); // canonical order is alphabetical
  EXPECT_EQ(order[1], "y");
}

TEST(PositionalEvalTest, VariadicAndRangeAgreeWithNamed) {
  auto x = PV(1.0, "x");
  auto y = PV(1.0, "y");
  const auto e = (x + y) * x; // canonical order {x,y}

  const double expected = 10.0; // (2+3)*2
  EXPECT_DOUBLE_EQ(eval(e, 2.0, 3.0), expected);
  EXPECT_DOUBLE_EQ(eval(e, std::array{2.0, 3.0}), expected);
  EXPECT_DOUBLE_EQ(eval(e, std::vector{2.0, 3.0}), expected);
  EXPECT_DOUBLE_EQ(bind(e, named<"x">(2.0), named<"y">(3.0)).eval(), expected);
}

TEST(PositionalEvalTest, AcceptsALazyNonSizedInputRange) {
  auto x = PV(1.0, "x");
  auto y = PV(1.0, "y");
  const auto e = (x + y) * x;

  // A view, single-pass friendly: 2.0, 3.0
  auto lazy = std::views::iota(2, 4) |
              std::views::transform([](int i) { return double(i); });
  EXPECT_DOUBLE_EQ(eval(e, lazy), 10.0);
}

TEST(PositionalEvalTest, ShortRangeThrowsAtRuntime) {
  auto x = PV(1.0, "x");
  auto y = PV(1.0, "y");
  const auto e = (x + y) * x;
  const std::vector<double> too_few{2.0};
  EXPECT_THROW((void)eval(e, too_few), std::out_of_range);
}

TEST(PositionalEvalTest, IsConstexpr) {
  constexpr auto x = PV(1.0, "x");
  constexpr auto y = PV(1.0, "y");
  constexpr auto e = (x + y) * x;
  static_assert(eval(e, 2.0, 3.0) == 10.0, "variadic positional eval");
  static_assert(eval(e, std::array{2.0, 3.0}) == 10.0, "range positional eval");
  EXPECT_DOUBLE_EQ(eval(e, 2.0, 3.0), 10.0);
}

// derivative_tensor<1> has to support every unary math function at any arity:
// each one below must *instantiate* at 3 variables and agree with reverse mode.
TEST(ForwardGradient, CoversEveryUnaryMathFunction) {
  const std::array<double, 3> pt{0.6, 0.4, 1.3};
  auto y = PV(0.4, "y");
  auto z = PV(1.3, "z");

  // Each case: 3 variables, compared against the same expression differentiated
  // by reverse mode, which shares none of the forward machinery.
#define EXPECT_FASTPATH_MATCHES_REVERSE(expr_)                                 \
  do {                                                                         \
    const auto e_ = (expr_) + y * z;                                           \
    const auto fwd_ = Equation{e_}.template derivative_tensor<1>(pt);                      \
    const auto rev_ = Equation{e_}.gradient(pt);               \
    EXPECT_NEAR(fwd_.data()[0], rev_[0], 1e-12);                               \
    EXPECT_NEAR(fwd_.data()[1], rev_[1], 1e-12);                               \
    EXPECT_NEAR(fwd_.data()[2], rev_[2], 1e-12);                               \
  } while (0)

  auto x = PV(0.6, "x");
  EXPECT_FASTPATH_MATCHES_REVERSE(sin(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(cos(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(tan(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(exp(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(log(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(log10(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(sqrt(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(cbrt(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(abs(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(asin(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(acos(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(atan(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(sinh(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(cosh(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(tanh(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(asinh(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(atanh(x));
  EXPECT_FASTPATH_MATCHES_REVERSE(erf(x));
  // acosh needs an argument > 1.
  {
    const std::array<double, 3> hi{1.7, 0.4, 1.3};
    const auto e_ = acosh(PV(1.7, "x")) + y * z;
    const auto fwd_ = Equation{e_}.template derivative_tensor<1>(hi);
    const auto rev_ = Equation{e_}.gradient(hi);
    EXPECT_NEAR(fwd_.data()[0], rev_[0], 1e-12);
  }
  // The BINARY math functions, same guard.
  {
    auto a = PV(1.3, "a");
    auto b = PV(0.7, "b");
    auto c = PV(0.5, "c");
    const std::array<double, 3> q{1.3, 0.7, 0.5};
#define EXPECT_BINARY_FASTPATH(expr_)                                          \
  do {                                                                         \
    const auto e_ = (expr_);                                                   \
    const auto fwd_ = Equation{e_}.template derivative_tensor<1>(q);                       \
    const auto rev_ = Equation{e_}.gradient(q);                \
    for (std::size_t k_ = 0; k_ < 3; ++k_) {                                   \
      EXPECT_NEAR(fwd_.data()[k_], rev_[k_], 1e-12);                           \
    }                                                                          \
  } while (0)
    EXPECT_BINARY_FASTPATH(hypot(a, b) + c * c);
    EXPECT_BINARY_FASTPATH(atan2(a, b) + c * c);
    EXPECT_BINARY_FASTPATH(pow(a, b) + c * c);
    EXPECT_BINARY_FASTPATH(max(a, b) + c * c);
    EXPECT_BINARY_FASTPATH(min(a, b) + c * c);
#undef EXPECT_BINARY_FASTPATH
  }
#undef EXPECT_FASTPATH_MATCHES_REVERSE
}
