#include "dual.hpp"
#include "equation.hpp"
#include "gradient.hpp"
#include "operations.hpp"
#include "traits.hpp"
#include "values.hpp"
#include <gtest/gtest.h>
#include <numbers>

using namespace diff;

// ===========================================================================
// Math function tests — ported from autodiff's test suite
// Covers: tan, log, sqrt, abs, asin, acos, atan, sinh, cosh, tanh,
//         identity checks, and reverse/forward mode coverage for all.
// ===========================================================================

TEST(MathFunctionTest, TanEvalAndDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  ASSERT_DOUBLE_EQ(tan(x).eval(), std::tan(x0));
  ASSERT_DOUBLE_EQ(tan(x).derivative().eval(),
                   1.0 / (std::cos(x0) * std::cos(x0)));
}

TEST(MathFunctionTest, LogEvalAndDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  ASSERT_DOUBLE_EQ(log(x).eval(), std::log(x0));
  ASSERT_DOUBLE_EQ(log(x).derivative().eval(), 1.0 / x0);
}

TEST(MathFunctionTest, SqrtEvalAndDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  ASSERT_DOUBLE_EQ(sqrt(x).eval(), std::sqrt(x0));
  ASSERT_DOUBLE_EQ(sqrt(x).derivative().eval(), 0.5 / std::sqrt(x0));
}

TEST(MathFunctionTest, AsinDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  ASSERT_DOUBLE_EQ(asin(x).eval(), std::asin(x0));
  ASSERT_DOUBLE_EQ(asin(x).derivative().eval(), 1.0 / std::sqrt(1.0 - x0 * x0));
}

TEST(MathFunctionTest, AcosDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  ASSERT_DOUBLE_EQ(acos(x).eval(), std::acos(x0));
  ASSERT_DOUBLE_EQ(acos(x).derivative().eval(),
                   -1.0 / std::sqrt(1.0 - x0 * x0));
}

TEST(MathFunctionTest, AtanDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  ASSERT_DOUBLE_EQ(atan(x).eval(), std::atan(x0));
  ASSERT_DOUBLE_EQ(atan(x).derivative().eval(), 1.0 / (1.0 + x0 * x0));
}

TEST(MathFunctionTest, SinhDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  ASSERT_DOUBLE_EQ(sinh(x).eval(), std::sinh(x0));
  ASSERT_DOUBLE_EQ(sinh(x).derivative().eval(), std::cosh(x0));
}

TEST(MathFunctionTest, CoshDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  ASSERT_DOUBLE_EQ(cosh(x).eval(), std::cosh(x0));
  ASSERT_DOUBLE_EQ(cosh(x).derivative().eval(), std::sinh(x0));
}

TEST(MathFunctionTest, TanhDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  ASSERT_DOUBLE_EQ(tanh(x).eval(), std::tanh(x0));
  double c = std::cosh(x0);
  ASSERT_DOUBLE_EQ(tanh(x).derivative().eval(), 1.0 / (c * c));
}

TEST(MathFunctionTest, AbsEval) {
  ASSERT_DOUBLE_EQ(abs(PV(3.0, 'x')).eval(), 3.0);
  ASSERT_DOUBLE_EQ(abs(PV(-3.0, 'x')).eval(), 3.0);
  ASSERT_DOUBLE_EQ(abs(PV(0.0, 'x')).eval(), 0.0);
}

// Chain rule with new ops
TEST(MathFunctionTest, ChainRuleTan) {
  // d/dx tan(2x) = 2/cos²(2x)
  double x0 = 0.4;
  auto x = PV(x0, 'x');
  auto expr = tan(PC(2.0) * x);
  ASSERT_DOUBLE_EQ(expr.derivative().eval(),
                   2.0 / (std::cos(2.0 * x0) * std::cos(2.0 * x0)));
}

TEST(MathFunctionTest, ChainRuleLog) {
  // d/dx log(x²) = 2/x
  double x0 = 0.8;
  auto x = PV(x0, 'x');
  auto expr = log(x * x);
  ASSERT_DOUBLE_EQ(expr.derivative().eval(), 2.0 / x0);
}

TEST(MathFunctionTest, ChainRuleSqrt) {
  // d/dx sqrt(sin(x)) = cos(x) / (2·sqrt(sin(x)))
  double x0 = 1.0;
  auto x = PV(x0, 'x');
  auto expr = sqrt(sin(x));
  double expected = std::cos(x0) / (2.0 * std::sqrt(std::sin(x0)));
  ASSERT_NEAR(expr.derivative().eval(), expected, 1e-12);
}

// Mathematical identities — value must be constant, derivative must be zero
TEST(MathFunctionTest, PythagoreanIdentity) {
  // sin²(x) + cos²(x) = 1, derivative = 0
  for (double v : {0.0, 0.5, 1.0, 2.0}) {
    auto x = PV(v, 'x');
    auto expr = sin(x) * sin(x) + cos(x) * cos(x);
    ASSERT_NEAR(expr.eval(), 1.0, 1e-12);
    ASSERT_NEAR(expr.derivative().eval(), 0.0, 1e-12);
  }
}

TEST(MathFunctionTest, HyperbolicIdentity) {
  // cosh²(x) - sinh²(x) = 1, derivative = 0
  for (double v : {0.0, 0.5, 1.0, 2.0}) {
    auto x = PV(v, 'x');
    auto expr = cosh(x) * cosh(x) - sinh(x) * sinh(x);
    ASSERT_NEAR(expr.eval(), 1.0, 1e-12);
    ASSERT_NEAR(expr.derivative().eval(), 0.0, 1e-12);
  }
}

TEST(MathFunctionTest, ExpLogIdentity) {
  // exp(log(x)) = x, derivative = 1
  for (double v : {0.3, 0.5, 1.0, 2.0}) {
    auto x = PV(v, 'x');
    auto expr = exp(log(x));
    ASSERT_NEAR(expr.eval(), v, 1e-12);
    ASSERT_NEAR(expr.derivative().eval(), 1.0, 1e-12);
  }
}

TEST(MathFunctionTest, QuotientSelfIsConstant) {
  // x/x = 1, derivative = 0
  for (double v : {1.0, 2.0, 5.0}) {
    auto x = PV(v, 'x');
    auto expr = x / x;
    ASSERT_NEAR(expr.eval(), 1.0, 1e-12);
    ASSERT_NEAR(expr.derivative().eval(), 0.0, 1e-12);
  }
}

TEST(MathFunctionTest, TanEqualsRatio) {
  // tan(x) = sin(x)/cos(x), derivatives agree
  double x0 = 0.7;
  auto x1 = PV(x0, 'x');
  auto x2 = PV(x0, 'x');
  ASSERT_NEAR(tan(x1).derivative().eval(),
              (sin(x2) / cos(x2)).derivative().eval(), 1e-12);
}

// ===========================================================================
// Reverse-mode gradients for new math functions
// ===========================================================================

TEST(ReverseModeAD, TanDerivative) {
  auto x = PV(0.5, 'x');
  auto g = reverse_mode_grad(tan(x));
  ASSERT_DOUBLE_EQ(g[0], 1.0 / (std::cos(0.5) * std::cos(0.5)));
}

TEST(ReverseModeAD, LogDerivative) {
  auto x = PV(0.5, 'x');
  auto g = reverse_mode_grad(log(x));
  ASSERT_DOUBLE_EQ(g[0], 2.0);
}

TEST(ReverseModeAD, SqrtDerivative) {
  auto x = PV(4.0, 'x');
  auto g = reverse_mode_grad(sqrt(x));
  ASSERT_DOUBLE_EQ(g[0], 0.25); // 0.5/sqrt(4) = 0.25
}

TEST(ReverseModeAD, AsinDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  auto g = reverse_mode_grad(asin(x));
  ASSERT_DOUBLE_EQ(g[0], 1.0 / std::sqrt(1.0 - x0 * x0));
}

TEST(ReverseModeAD, AcosDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  auto g = reverse_mode_grad(acos(x));
  ASSERT_DOUBLE_EQ(g[0], -1.0 / std::sqrt(1.0 - x0 * x0));
}

TEST(ReverseModeAD, AtanDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  auto g = reverse_mode_grad(atan(x));
  ASSERT_DOUBLE_EQ(g[0], 1.0 / (1.0 + x0 * x0));
}

TEST(ReverseModeAD, SinhDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  auto g = reverse_mode_grad(sinh(x));
  ASSERT_DOUBLE_EQ(g[0], std::cosh(x0));
}

TEST(ReverseModeAD, CoshDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  auto g = reverse_mode_grad(cosh(x));
  ASSERT_DOUBLE_EQ(g[0], std::sinh(x0));
}

TEST(ReverseModeAD, TanhDerivative) {
  double x0 = 0.5;
  auto x = PV(x0, 'x');
  auto g = reverse_mode_grad(tanh(x));
  double c = std::cosh(x0);
  ASSERT_DOUBLE_EQ(g[0], 1.0 / (c * c));
}

TEST(ReverseModeAD, AbsDerivativePositive) {
  auto x = PV(1.0, 'x');
  ASSERT_DOUBLE_EQ(reverse_mode_grad(abs(x))[0], 1.0);
}

TEST(ReverseModeAD, AbsDerivativeNegative) {
  auto x = PV(-1.0, 'x');
  ASSERT_DOUBLE_EQ(reverse_mode_grad(abs(x))[0], -1.0);
}

TEST(ReverseModeAD, AbsDerivativeAtZero) {
  auto x = PV(0.0, 'x');
  ASSERT_DOUBLE_EQ(reverse_mode_grad(abs(x))[0], 0.0);
}

// ===========================================================================
// Forward-mode (Dual) for new math functions
// ===========================================================================

TEST(ForwardModeAD, TanDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto [f, df] = tan(x).eval();
  ASSERT_DOUBLE_EQ(f, std::tan(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / (std::cos(x0) * std::cos(x0)));
}

TEST(ForwardModeAD, LogDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto [f, df] = log(x).eval();
  ASSERT_DOUBLE_EQ(f, std::log(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / x0);
}

TEST(ForwardModeAD, SqrtDerivative) {
  double x0 = 4.0;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto [f, df] = sqrt(x).eval();
  ASSERT_DOUBLE_EQ(f, 2.0);
  ASSERT_DOUBLE_EQ(df, 0.25);
}

TEST(ForwardModeAD, AsinDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto [f, df] = asin(x).eval();
  ASSERT_DOUBLE_EQ(f, std::asin(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / std::sqrt(1.0 - x0 * x0));
}

TEST(ForwardModeAD, AcosDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto [f, df] = acos(x).eval();
  ASSERT_DOUBLE_EQ(f, std::acos(x0));
  ASSERT_DOUBLE_EQ(df, -1.0 / std::sqrt(1.0 - x0 * x0));
}

TEST(ForwardModeAD, AtanDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto [f, df] = atan(x).eval();
  ASSERT_DOUBLE_EQ(f, std::atan(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / (1.0 + x0 * x0));
}

TEST(ForwardModeAD, SinhDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto [f, df] = sinh(x).eval();
  ASSERT_DOUBLE_EQ(f, std::sinh(x0));
  ASSERT_DOUBLE_EQ(df, std::cosh(x0));
}

TEST(ForwardModeAD, CoshDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto [f, df] = cosh(x).eval();
  ASSERT_DOUBLE_EQ(f, std::cosh(x0));
  ASSERT_DOUBLE_EQ(df, std::sinh(x0));
}

TEST(ForwardModeAD, TanhDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto [f, df] = tanh(x).eval();
  double c = std::cosh(x0);
  ASSERT_DOUBLE_EQ(f, std::tanh(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / (c * c));
}

TEST(ForwardModeAD, AbsDerivativePositive) {
  Variable<Dual<double>, 'x'> x{Dual<double>{2.0, 1.0}};
  auto [f, df] = abs(x).eval();
  ASSERT_DOUBLE_EQ(f, 2.0);
  ASSERT_DOUBLE_EQ(df, 1.0);
}

TEST(ForwardModeAD, AbsDerivativeNegative) {
  Variable<Dual<double>, 'x'> x{Dual<double>{-2.0, 1.0}};
  auto [f, df] = abs(x).eval();
  ASSERT_DOUBLE_EQ(f, 2.0);
  ASSERT_DOUBLE_EQ(df, -1.0);
}

// ===========================================================================
// Concept satisfaction (static_assert — compile-time contract tests)
// ===========================================================================

TEST(ConceptTest, NumericSatisfied) {
  static_assert(Numeric<int>);
  static_assert(Numeric<double>);
  static_assert(Numeric<float>);
  static_assert(Numeric<long double>);
  static_assert(!Numeric<std::string>);
}

TEST(ConceptTest, ExpressionConceptSatisfied) {
  static_assert(CExpression<Constant<double>>);
  static_assert(CExpression<Variable<double, 'x'>>);
  using SumExpr = decltype(std::declval<Variable<double, 'x'>>() +
                           std::declval<Constant<double>>());
  static_assert(CExpression<SumExpr>);
  static_assert(!CExpression<int>);
  static_assert(!CExpression<double>);
}

TEST(ConceptTest, AnOpSatisfied) {
  static_assert(COp<SumOp<double>>);
  static_assert(COp<MultiplyOp<float>>);
  static_assert(COp<SineOp<double>>);
  static_assert(COp<CosineOp<double>>);
  static_assert(COp<ExpOp<double>>);
  static_assert(COp<NegateOp<int>>);
  static_assert(COp<DivideOp<double>>);
}

// ===========================================================================
// Symbol extraction
// ===========================================================================

TEST(SymbolTest, SingleVariable) {
  using E = Variable<double, 'x'>;
  using Syms = extract_symbols_from_expr_t<E>;
  static_assert(boost::mp11::mp_size<Syms>::value == 1);
  static_assert(std::is_same_v<boost::mp11::mp_at_c<Syms, 0>,
                               std::integral_constant<char, 'x'>>);
}

TEST(SymbolTest, TwoVariables) {
  using E = decltype(std::declval<Variable<double, 'x'>>() *
                     std::declval<Variable<double, 'y'>>());
  using Syms = extract_symbols_from_expr_t<E>;
  static_assert(boost::mp11::mp_size<Syms>::value == 2);
  // Symbols are sorted by char value: 'x' < 'y'
  static_assert(std::is_same_v<boost::mp11::mp_at_c<Syms, 0>,
                               std::integral_constant<char, 'x'>>);
  static_assert(std::is_same_v<boost::mp11::mp_at_c<Syms, 1>,
                               std::integral_constant<char, 'y'>>);
}

TEST(SymbolTest, DuplicateSymbolsDeduplicated) {
  // x * x has only one distinct symbol
  using E = decltype(std::declval<Variable<double, 'x'>>() *
                     std::declval<Variable<double, 'x'>>());
  using Syms = extract_symbols_from_expr_t<E>;
  static_assert(boost::mp11::mp_size<Syms>::value == 1);
}

TEST(SymbolTest, ThreeVariables) {
  auto x = PV(1.0, 'x');
  auto y = PV(2.0, 'y');
  auto z = PV(3.0, 'z');
  auto expr = x + y + z;
  using Syms = extract_symbols_from_expr_t<decltype(expr)>;
  static_assert(boost::mp11::mp_size<Syms>::value == 3);
}

// ===========================================================================
// Expression / operator tests
// ===========================================================================

TEST(ExpressionTest, StaticTests) {
  static_assert(
      std::is_same_v<
          as_const_expression<
              Expression<MultiplyOp<int>, Variable<int, 'x'>, Constant<int>>>,
          Expression<MultiplyOp<int>, Constant<int>, Constant<int>>>);

  static_assert(
      std::is_same_v<
          as_const_expression<Expression<MultiplyOp<int>, Variable<int, 'x'>,
                                         Variable<int, 'y'>>>,
          Expression<MultiplyOp<int>, Constant<int>, Constant<int>>>);

  auto x = 4_vi;
  auto y = 2_vi;
  auto c = 2_ci;
  auto res = x * y + c;
  auto res2 = make_const_variable<'c'>(res);
  ASSERT_EQ(res2, res);
}

TEST(ExpressionTest, SumTest) {
  double a = 1, b = 2, c = 3;
  auto sum_exp = a + b + c;
  ASSERT_EQ(sum_exp, 6);
}

TEST(ExpressionTest, MultiplyTest) {
  auto a = 1_ci;
  auto b = 2_vi;
  auto c = 3_ci;
  auto sum_exp = a * b * c;
  auto d = sum_exp.derivative();
  ASSERT_EQ(sum_exp, 6);
  ASSERT_EQ(d, 3);
}

TEST(ExpressionTest, SubtractTest) {
  auto a = 1_ci;
  auto b = 2_vi;
  auto minus = a - b;
  auto d = minus.derivative();
  ASSERT_EQ(minus, -1);
  ASSERT_EQ(d, -1);
}

TEST(ExpressionTest, DivideTest) {
  auto a = 4.0_vd;
  auto b = 2.0_cd;
  auto divide = a / b;
  auto d = divide.derivative();
  ASSERT_EQ(divide, 2.0);
  ASSERT_EQ(d, 0.5);
}

TEST(ExpressionTest, ExpTest) {
  constexpr auto exp_exp = exp(2.0_cd);
  ASSERT_EQ(exp_exp, std::exp(2.0));
}

TEST(ExpressionTest, ExpSum) {
  constexpr auto target = exp(1_cd + 2_cd);
  ASSERT_EQ(target, std::exp(3.0));
}

TEST(ExpressionTest, ExpDerivative) {
  for (std::size_t i = 1; i < 1000; ++i) {
    auto target = exp(Variable<double, 'x'>{i * 1.0});
    ASSERT_EQ(target.derivative(), target);
  }
}

TEST(ExpressionTest, Combination) {
  double target = exp(1_cd + 2_cd + 3_cd) + 1_cd;
  ASSERT_EQ(target, std::exp(6.0) + 1.0);
}

TEST(ExpressionTest, ConstantDerivative) {
  auto target = Constant<int>(1);
  ASSERT_EQ(target.derivative(), 0);
  ASSERT_EQ(target, 1);
}

TEST(ExpressionTest, VariableDerivative) {
  auto x = Variable<int, 'x'>{5};
  ASSERT_EQ(x.derivative(), 1);
  ASSERT_EQ(x, 5);
}

TEST(ExpressionTest, DerivativeTest) {
  auto x = 4_vi;
  auto expr = x * 2_ci;
  auto derv = expr.derivative();
  ASSERT_EQ(expr, 8);
  ASSERT_EQ(derv, 2);
}

// ===========================================================================
// Algebraic derivative rules
// ===========================================================================

TEST(DerivativeRuleTest, SumRule) {
  // d/dx [f + g] = f' + g'  =>  d/dx [3x + 5] = 3
  auto x = PV(7, 'x');
  auto expr = 3_ci * x + 5_ci;
  ASSERT_EQ(expr.derivative(), 3);
}

TEST(DerivativeRuleTest, ProductRule) {
  // d/dx [x * x] = 2x  at x=4 => 8
  auto x = Variable<int, 'x'>{4};
  auto expr = x * x;
  ASSERT_EQ(expr.derivative(), 8);
}

TEST(DerivativeRuleTest, QuotientRule) {
  // d/dx [x / c] = 1/c  at x=6, c=3 => 1/3
  auto x = Variable<double, 'x'>{6.0};
  auto c = Constant<double>{3.0};
  auto expr = x / c;
  ASSERT_DOUBLE_EQ(expr.derivative(), 1.0 / 3.0);
}

TEST(DerivativeRuleTest, ChainRule_ExpOfLinear) {
  // d/dx [e^(2x)] = 2*e^(2x)  at x=1
  auto x = Variable<double, 'x'>{1.0};
  auto inner = PC(2.0) * x;
  auto expr = exp(inner);
  ASSERT_DOUBLE_EQ(expr.derivative().eval(), 2.0 * std::exp(2.0));
}

TEST(DerivativeRuleTest, ChainRule_SinOfLinear) {
  // d/dx [sin(3x)] = 3*cos(3x)  at x=0
  auto x = Variable<double, 'x'>{0.0};
  auto inner = PC(3.0) * x;
  auto expr = sin(inner);
  ASSERT_DOUBLE_EQ(expr.derivative().eval(), 3.0 * std::cos(0.0));
}

// ===========================================================================
// Variable tests
// ===========================================================================

TEST(VariableTest, GetValue) {
  auto a = 2_vi;
  ASSERT_EQ(a, 2);
}

TEST(VariableTest, Assign) {
  Variable<int, 'a'> a{4};
  a = 2;
  ASSERT_EQ(a, 2);
}

TEST(VariableTest, UdlCompAndAssign) {
  Variable<int, 'a'> a{4};
  auto b = 4_vi;
  ASSERT_EQ(a, b);
}

// ===========================================================================
// Trig tests
// ===========================================================================

TEST(TrigTest, SinTest) {
  auto b = sin(PC(0.5));
  ASSERT_EQ(b, std::sin(0.5));
}

TEST(TrigTest, CosTest) {
  auto b = cos(PV(0.45, 'x'));
  ASSERT_EQ(b, std::cos(0.45));
  ASSERT_DOUBLE_EQ(b.derivative().eval(), -std::sin(0.45));
}

TEST(TrigTest, SinDerivative) {
  auto x = PV(0.7, 'x');
  auto s = sin(x);
  ASSERT_DOUBLE_EQ(s.derivative().eval(), std::cos(0.7));
}

TEST(TrigTest, SinCosIdentity) {
  // sin^2(x) + cos^2(x) == 1
  for (double v : {0.0, 0.5, 1.0, std::numbers::pi / 4}) {
    auto x = Variable<double, 'x'>{v};
    auto s = sin(x);
    auto c = cos(x);
    double lhs = static_cast<double>(s * s) + static_cast<double>(c * c);
    ASSERT_NEAR(lhs, 1.0, 1e-12);
  }
}

TEST(TrigTest, CosDerivativeIsNegSin) {
  // d/dx cos(x) = -sin(x)  at several points
  for (double v : {0.0, 0.3, 1.0, 2.0}) {
    auto x = Variable<double, 'x'>{v};
    ASSERT_DOUBLE_EQ(cos(x).derivative().eval(), -std::sin(v));
  }
}

TEST(TrigTest, ExpDerivativeIsItself) {
  // d/dx e^x = e^x
  for (double v : {-1.0, 0.0, 0.5, 1.5}) {
    auto x = Variable<double, 'x'>{v};
    ASSERT_DOUBLE_EQ(exp(x).derivative().eval(), std::exp(v));
  }
}

// ===========================================================================
// Equation (partial differentiation) tests
// ===========================================================================

TEST(EquationTest, SingleVariable) {
  auto x = PV(4, 'x');
  auto expr = x * 2_ci;
  auto eq = Equation(expr);
  ASSERT_EQ(eq[IDX(1)], 2);
}

TEST(EquationTest, TwoVariables) {
  auto x = PV(4, 'x');
  auto y = PV(2, 'y');
  auto expr = x * y;
  auto eq = Equation(expr);
  ASSERT_EQ(eq[IDX(1)], 2); // df/dx = y = 2
  ASSERT_EQ(eq[IDX(2)], 4); // df/dy = x = 4
}

TEST(EquationTest, LinearCombination) {
  auto x = PV(4, 'x');
  auto y = PV(2, 'y');
  auto expr = PC(1) * x + PC(2) * y;
  auto eq = Equation(expr);
  ASSERT_EQ(eq[IDX(1)], 1);
  ASSERT_EQ(eq[IDX(2)], 2);
}

TEST(EquationTest, DifferenceOfSquares) {
  constexpr auto x = PV(4, 'x');
  constexpr auto y = PV(2, 'y');
  constexpr auto expr = (x + y) * (x - y);
  auto eq = Equation(expr);
  auto [d1, d2] = eq.eval_derivatives();
  ASSERT_EQ(expr, 12); // (4+2)*(4-2) = 12
  ASSERT_EQ(d1, 8);    // 2x = 8
  ASSERT_EQ(d2, -4);   // -2y = -4
}

TEST(EquationTest, ExpEquation) {
  auto x = PV(1.0, 'x');
  auto expr = exp(x);
  auto eq = Equation(expr);
  ASSERT_DOUBLE_EQ(eq.evaluate(), std::exp(1.0));
  ASSERT_DOUBLE_EQ(eq[IDX(1)].eval(), std::exp(1.0)); // d(e^x)/dx = e^x
}

TEST(EquationTest, TrigEquation) {
  auto x = PV(0.5, 'x');
  auto expr = sin(x) * cos(x);
  auto eq = Equation(expr);
  // d/dx [sin(x)cos(x)] = cos^2(x) - sin^2(x) = cos(2x)
  ASSERT_DOUBLE_EQ(eq[IDX(1)].eval(), std::cos(2 * 0.5));
}

TEST(EquationTest, IdxEquivalence) {
  // idx<N>() and IDX(N) must produce the same index object
  auto x = PV(3, 'x');
  auto y = PV(5, 'y');
  auto eq = Equation(x * y);
  ASSERT_EQ(eq[idx<1>()], eq[IDX(1)]);
  ASSERT_EQ(eq[idx<2>()], eq[IDX(2)]);
}

TEST(EquationTest, ThreeVariablePartials) {
  // f(x,y,z) = x*y + y*z  at (2,3,4)
  // df/dx = y = 3, df/dy = x + z = 6, df/dz = y = 3
  auto x = PV(2, 'x');
  auto y = PV(3, 'y');
  auto z = PV(4, 'z');
  auto expr = x * y + y * z;
  auto eq = Equation(expr);
  auto [dx, dy, dz] = eq.eval_derivatives();
  ASSERT_EQ(dx, 3);
  ASSERT_EQ(dy, 6);
  ASSERT_EQ(dz, 3);
}

TEST(EquationTest, UpdateAndReevaluate) {
  // f(x) = x^2,  df/dx = 2x.
  // Both copies of x in the derivative are live variables, so update
  // propagates.
  auto x = Variable<int, 'x'>{3};
  auto expr = x * x;
  auto eq = Equation(expr);

  ASSERT_EQ(eq.evaluate(), 9);
  ASSERT_EQ(eq[IDX(1)].eval(), 6); // 2*3 = 6

  using Syms = Equation<decltype(expr)>::symbols;
  eq.update(Syms{}, std::array{5});
  ASSERT_EQ(eq.evaluate(), 25);
  ASSERT_EQ(eq[IDX(1)].eval(), 10); // 2*5 = 10
}

TEST(EquationTest, MixedTrigExpEquation) {
  // f(x) = e^x * sin(x)  at x=1
  // f'(x) = e^x*(sin(x) + cos(x))
  auto x = PV(1.0, 'x');
  auto expr = exp(x) * sin(x);
  auto eq = Equation(expr);
  double expected = std::exp(1.0) * (std::sin(1.0) + std::cos(1.0));
  ASSERT_DOUBLE_EQ(eq[IDX(1)].eval(), expected);
}

TEST(EquationTest, NumberOfDerivatives) {
  auto x = PV(1, 'x');
  auto y = PV(2, 'y');
  static_assert(Equation<decltype(x * y)>::number_of_derivatives == 2);
  static_assert(Equation<decltype(x * x)>::number_of_derivatives == 1);
}

// ===========================================================================
// Equation — f: ℝⁿ → ℝᵐ  (Jacobian tests)
// ===========================================================================

TEST(EquationTest, Dimensions) {
  auto x = PV(1, 'x');
  auto y = PV(2, 'y');
  // f: ℝ² → ℝ²
  using VE = Equation<decltype(x + y), decltype(x * y)>;
  static_assert(VE::output_dim == 2);
  static_assert(VE::input_dim == 2);
}

TEST(EquationTest, Eval) {
  // f(x,y) = (x + y,  x * y)  at (3, 4)  =>  (7, 12)
  auto x = PV(3, 'x');
  auto y = PV(4, 'y');
  auto ve = Equation(x + y, x * y);
  auto v = ve.evaluate();
  ASSERT_EQ(v[0], 7);
  ASSERT_EQ(v[1], 12);
}

TEST(EquationTest, JacobianLinear) {
  // f(x,y) = (x + y,  x * y)  at (3, 4)
  // J = [[1, 1],
  //      [y, x]] = [[1, 1], [4, 3]]
  auto x = PV(3, 'x');
  auto y = PV(4, 'y');
  auto ve = Equation(x + y, x * y);
  auto J = ve.symbolic_mode_jac();
  ASSERT_EQ(J[0][0], 1); // ∂(x+y)/∂x
  ASSERT_EQ(J[0][1], 1); // ∂(x+y)/∂y
  ASSERT_EQ(J[1][0], 4); // ∂(x*y)/∂x = y = 4
  ASSERT_EQ(J[1][1], 3); // ∂(x*y)/∂y = x = 3
}

TEST(EquationTest, JacobianWithTrig) {
  // f(x,y) = (x*y,  sin(x) + y*y)  at (2.0, 3.0)
  // J = [[y,      x   ],
  //      [cos(x), 2y  ]]
  auto x = PV(2.0, 'x');
  auto y = PV(3.0, 'y');
  auto ve = Equation(x * y, sin(x) + y * y);
  auto J = ve.symbolic_mode_jac();
  ASSERT_DOUBLE_EQ(J[0][0], 3.0);           // ∂(x*y)/∂x = y
  ASSERT_DOUBLE_EQ(J[0][1], 2.0);           // ∂(x*y)/∂y = x
  ASSERT_DOUBLE_EQ(J[1][0], std::cos(2.0)); // ∂(sin(x)+y²)/∂x
  ASSERT_DOUBLE_EQ(J[1][1], 6.0);           // ∂(sin(x)+y²)/∂y = 2y
}

TEST(EquationTest, SingleComponentIsGradient) {
  // Scalar Equation derivatives match a 2-output Jacobian row when
  // both components are the same expression.
  auto x = PV(2.0, 'x');
  auto y = PV(3.0, 'y');
  auto eq = Equation(x * y);
  auto g = eq.eval_derivatives();
  ASSERT_DOUBLE_EQ(g[0], 3.0); // ∂(x*y)/∂x = y
  ASSERT_DOUBLE_EQ(g[1], 2.0); // ∂(x*y)/∂y = x
}

TEST(EquationTest, SymbolUnionAcrossComponents) {
  // f0 depends only on x,  f1 depends only on y.
  // Jacobian should be 2×2 with zeros off the diagonal.
  auto x = PV(4.0, 'x');
  auto y = PV(3.0, 'y');
  auto ve = Equation(x * x, y * y); // (x², y²)
  static_assert(decltype(ve)::input_dim == 2);
  auto J = ve.symbolic_mode_jac();
  ASSERT_DOUBLE_EQ(J[0][0], 8.0); // ∂(x²)/∂x = 2x = 8
  ASSERT_DOUBLE_EQ(J[0][1], 0.0); // ∂(x²)/∂y = 0
  ASSERT_DOUBLE_EQ(J[1][0], 0.0); // ∂(y²)/∂x = 0
  ASSERT_DOUBLE_EQ(J[1][1], 6.0); // ∂(y²)/∂y = 2y = 6
}

TEST(EquationTest, ThreeOutputs) {
  // f(x,y) = (x², x*y, y²)  — Jacobian is 3×2
  auto x = PV(2.0, 'x');
  auto y = PV(5.0, 'y');
  auto ve = Equation(x * x, x * y, y * y);
  static_assert(decltype(ve)::output_dim == 3);
  static_assert(decltype(ve)::input_dim == 2);
  auto J = ve.symbolic_mode_jac();
  ASSERT_DOUBLE_EQ(J[0][0], 4.0);  // 2x
  ASSERT_DOUBLE_EQ(J[0][1], 0.0);  // 0
  ASSERT_DOUBLE_EQ(J[1][0], 5.0);  // y
  ASSERT_DOUBLE_EQ(J[1][1], 2.0);  // x
  ASSERT_DOUBLE_EQ(J[2][0], 0.0);  // 0
  ASSERT_DOUBLE_EQ(J[2][1], 10.0); // 2y
}

TEST(EquationTest, ReverseJacobianAgreesWithSymbolic) {
  auto x = PV(2.0, 'x');
  auto y = PV(3.0, 'y');
  auto z = PV(4.0, 'z');
  auto ve = Equation(x * y, sin(x) + y * z, exp(z));

  auto J_sym = ve.symbolic_mode_jac();
  auto J_rev = ve.reverse_mode_jac();

  for (std::size_t i = 0; i < decltype(ve)::output_dim; ++i)
    for (std::size_t j = 0; j < decltype(ve)::input_dim; ++j)
      ASSERT_DOUBLE_EQ(J_rev[i][j], J_sym[i][j]);
}

TEST(EquationTest, ParallelReverseJacobian_FourOutputs) {
  // f: ℝ³ → ℝ⁴ — four async tasks, verifies no data race across rows
  auto x = PV(1.0, 'x');
  auto y = PV(2.0, 'y');
  auto z = PV(3.0, 'z');
  auto ve = Equation(x * y, y * z, x * z, x * y * z);
  static_assert(decltype(ve)::output_dim == 4);
  static_assert(decltype(ve)::input_dim == 3);

  auto J_sym = ve.symbolic_mode_jac();
  auto J_rev = ve.reverse_mode_jac();

  for (std::size_t i = 0; i < decltype(ve)::output_dim; ++i)
    for (std::size_t j = 0; j < decltype(ve)::input_dim; ++j)
      ASSERT_DOUBLE_EQ(J_rev[i][j], J_sym[i][j]);
}

TEST(EquationTest, ParallelReverseJacobian_FiveOutputsTrigExp) {
  // f: ℝ³ → ℝ⁵ — heavier expressions across more rows
  auto x = PV(0.5, 'x');
  auto y = PV(1.0, 'y');
  auto z = PV(1.5, 'z');
  auto ve = Equation(sin(x) * cos(y), exp(x + y), x * y + y * z,
                     cos(z) * sin(x), exp(x * z) + y * y);
  static_assert(decltype(ve)::output_dim == 5);
  static_assert(decltype(ve)::input_dim == 3);

  auto J_sym = ve.symbolic_mode_jac();
  auto J_rev = ve.reverse_mode_jac();

  for (std::size_t i = 0; i < decltype(ve)::output_dim; ++i)
    for (std::size_t j = 0; j < decltype(ve)::input_dim; ++j)
      ASSERT_NEAR(J_rev[i][j], J_sym[i][j], 1e-12);
}

TEST(EquationTest, ReverseJacobianSingleOutputMatchesGradient) {
  auto x = PV(2.0, 'x');
  auto y = PV(5.0, 'y');
  auto expr = exp(x) * sin(y);
  // Use a 2-component Equation so the vector specialization is selected.
  auto ve = Equation(expr, exp(x) * sin(y));

  auto J_rev = ve.reverse_mode_jac();
  auto g = reverse_mode_grad(expr);

  for (std::size_t j = 0; j < decltype(ve)::input_dim; ++j)
    ASSERT_DOUBLE_EQ(J_rev[0][j], g[j]);
}

// ===========================================================================
// Forward-mode automatic differentiation via dual numbers
// ===========================================================================

TEST(ForwardModeAD, ExpressionStructuredBinding) {
  // Non-Dual: auto [f, df] = expr gives {eval(), derivative().eval()}
  auto x = PV(3.0, 'x');
  auto [f, df] = x * x; // f=9, df=2*3=6
  EXPECT_DOUBLE_EQ(f, 9.0);
  EXPECT_DOUBLE_EQ(df, 6.0);

  auto [g, dg] = sin(PV(0.0, 'x')); // g=sin(0)=0, dg=cos(0)*1=1
  EXPECT_DOUBLE_EQ(g, 0.0);
  EXPECT_DOUBLE_EQ(dg, 1.0);
}

TEST(ForwardModeAD, DualNumericConcept) {
  static_assert(Numeric<Dual<double>>);
  static_assert(Numeric<Dual<float>>);
}

TEST(ForwardModeAD, StructuredBinding) {
  Dual<double> d{3.0, 7.0};
  auto [v, dv] = d;
  EXPECT_DOUBLE_EQ(v, 3.0);
  EXPECT_DOUBLE_EQ(dv, 7.0);
  static_assert(std::tuple_size_v<Dual<double>> == 2);
  static_assert(std::is_same_v<std::tuple_element_t<0, Dual<double>>, double>);
  static_assert(std::is_same_v<std::tuple_element_t<1, Dual<double>>, double>);
}

TEST(ForwardModeAD, BasicArithmetic) {
  // Verify dual number arithmetic rules directly.
  constexpr Dual<double> a{3.0, 1.0};
  constexpr Dual<double> b{2.0, 0.0};
  auto [sum_val, sum_deriv] = a + b;
  EXPECT_DOUBLE_EQ(sum_val, 5.0);
  EXPECT_DOUBLE_EQ(sum_deriv, 1.0);
  auto [prod_val, prod_deriv] = a * b;
  EXPECT_DOUBLE_EQ(prod_val, 6.0);
  EXPECT_DOUBLE_EQ(prod_deriv, 2.0); // 1*2 + 3*0 = 2
  auto [quot_val, quot_deriv] = a / b;
  EXPECT_DOUBLE_EQ(quot_val, 1.5);
  EXPECT_DOUBLE_EQ(quot_deriv, 0.5); // (1*2 - 3*0)/4 = 0.5
}

TEST(ForwardModeAD, PolynomialDerivative) {
  // f(x) = x^2 + x,  f'(x) = 2x + 1
  // At x=3: f=12, f'=7
  Variable<Dual<double>, 'x'> x{Dual<double>{3.0, 1.0}};
  auto [f, df] = (x * x + x).eval();
  EXPECT_DOUBLE_EQ(f, 12.0);
  EXPECT_DOUBLE_EQ(df, 7.0);
}

TEST(ForwardModeAD, PartialDerivativeX) {
  // f(x,y) = x*y,  df/dx = y
  // At (3,4): f=12, df/dx=4
  Variable<Dual<double>, 'x'> x{Dual<double>{3.0, 1.0}};
  Variable<Dual<double>, 'y'> y{Dual<double>{4.0, 0.0}};
  auto [f, df] = (x * y).eval();
  EXPECT_DOUBLE_EQ(f, 12.0);
  EXPECT_DOUBLE_EQ(df, 4.0);
}

TEST(ForwardModeAD, PartialDerivativeY) {
  // f(x,y) = x*y,  df/dy = x
  // At (3,4): f=12, df/dy=3
  Variable<Dual<double>, 'x'> x{Dual<double>{3.0, 0.0}};
  Variable<Dual<double>, 'y'> y{Dual<double>{4.0, 1.0}};
  auto [f, df] = (x * y).eval();
  EXPECT_DOUBLE_EQ(f, 12.0);
  EXPECT_DOUBLE_EQ(df, 3.0);
}

TEST(ForwardModeAD, SinDerivative) {
  // f(x) = sin(x),  f'(x) = cos(x)
  double x0 = std::numbers::pi / 4.0;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto [f, df] = sin(x).eval();
  EXPECT_DOUBLE_EQ(f, std::sin(x0));
  EXPECT_DOUBLE_EQ(df, std::cos(x0));
}

TEST(ForwardModeAD, CosDerivative) {
  // f(x) = cos(x),  f'(x) = -sin(x)
  double x0 = std::numbers::pi / 3.0;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto [f, df] = cos(x).eval();
  EXPECT_DOUBLE_EQ(f, std::cos(x0));
  EXPECT_DOUBLE_EQ(df, -std::sin(x0));
}

TEST(ForwardModeAD, ExpDerivative) {
  // f(x) = exp(x),  f'(x) = exp(x)
  double x0 = 2.0;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto [f, df] = exp(x).eval();
  auto [f2, df2] = exp(x);
  EXPECT_DOUBLE_EQ(f, std::exp(x0));
  EXPECT_DOUBLE_EQ(df, std::exp(x0));
  EXPECT_DOUBLE_EQ(f, f2);
  EXPECT_DOUBLE_EQ(df, df2);
}

TEST(ForwardModeAD, ChainRule) {
  // f(x) = sin(x^2),  f'(x) = 2x*cos(x^2)
  double x0 = 1.0;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto [f, df] = sin(x * x);
  EXPECT_DOUBLE_EQ(f, std::sin(x0 * x0));
  EXPECT_DOUBLE_EQ(df, 2.0 * x0 * std::cos(x0 * x0));
}

TEST(ForwardModeAD, Equivalence) {
  // f(x) = sin(x^2),  f'(x) = 2x*cos(x^2)
  double x0 = 1.0;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto xv = PV(x0, 'x');
  auto l = sin(xv * xv);
  auto [f, df] = sin(x * x);
  auto f2 = l.eval();
  auto df2 = l.derivative().eval();
  EXPECT_DOUBLE_EQ(f, std::sin(x0 * x0));
  EXPECT_DOUBLE_EQ(df, 2.0 * x0 * std::cos(x0 * x0));
  EXPECT_DOUBLE_EQ(df, df2);
  EXPECT_DOUBLE_EQ(f, f2);
}

// ===========================================================================
// Reverse-mode automatic differentiation via backward() /
// reverse_mode_grad()
// ===========================================================================

TEST(ReverseModeAD, SingleVariableLinear) {
  // f(x) = 3*x  at x=5,  df/dx = 3
  auto x = PV(5.0, 'x');
  auto expr = PC(3.0) * x;
  auto g = reverse_mode_grad(expr);
  EXPECT_DOUBLE_EQ(g[0], 3.0);
}

TEST(ReverseModeAD, ProductRule) {
  // f(x) = x*x  at x=4,  df/dx = 2x = 8
  auto x = Variable<double, 'x'>{4.0};
  auto expr = x * x;
  auto g = reverse_mode_grad(expr);
  EXPECT_DOUBLE_EQ(g[0], 8.0);
}

TEST(ReverseModeAD, TwoVariables) {
  // f(x,y) = x*y  at (3,4),  df/dx=4, df/dy=3
  auto x = PV(3.0, 'x');
  auto y = PV(4.0, 'y');
  auto expr = x * y;
  auto g = reverse_mode_grad(expr);
  static_assert(g.size() == 2);
  EXPECT_DOUBLE_EQ(g[0], 4.0); // df/dx = y
  EXPECT_DOUBLE_EQ(g[1], 3.0); // df/dy = x
}

TEST(ReverseModeAD, Sum) {
  // f(x,y) = x + y,  df/dx=1, df/dy=1
  auto x = PV(2.0, 'x');
  auto y = PV(5.0, 'y');
  auto g = reverse_mode_grad(x + y);
  EXPECT_DOUBLE_EQ(g[0], 1.0);
  EXPECT_DOUBLE_EQ(g[1], 1.0);
}

TEST(ReverseModeAD, LinearCombination) {
  // f(x,y) = 2*x + 3*y,  df/dx=2, df/dy=3
  auto x = PV(1.0, 'x');
  auto y = PV(1.0, 'y');
  auto g = reverse_mode_grad(PC(2.0) * x + PC(3.0) * y);
  EXPECT_DOUBLE_EQ(g[0], 2.0);
  EXPECT_DOUBLE_EQ(g[1], 3.0);
}

TEST(ReverseModeAD, Divide) {
  // f(x) = x/c at x=6, c=3,  df/dx = 1/c = 1/3
  auto x = PV(6.0, 'x');
  auto c = PC(3.0);
  auto g = reverse_mode_grad(x / c);
  EXPECT_DOUBLE_EQ(g[0], 1.0 / 3.0);
}

TEST(ReverseModeAD, NegateViaSubtract) {
  // f(x,y) = x - y,  df/dx=1, df/dy=-1
  auto x = PV(5.0, 'x');
  auto y = PV(2.0, 'y');
  auto g = reverse_mode_grad(x - y);
  EXPECT_DOUBLE_EQ(g[0], 1.0);
  EXPECT_DOUBLE_EQ(g[1], -1.0);
}

TEST(ReverseModeAD, SinDerivative) {
  // f(x) = sin(x),  df/dx = cos(x)  at x=1
  auto x = PV(1.0, 'x');
  auto g = reverse_mode_grad(sin(x));
  EXPECT_DOUBLE_EQ(g[0], std::cos(1.0));
}

TEST(ReverseModeAD, CosDerivative) {
  // f(x) = cos(x),  df/dx = -sin(x)  at x=1
  auto x = PV(1.0, 'x');
  auto g = reverse_mode_grad(cos(x));
  EXPECT_DOUBLE_EQ(g[0], -std::sin(1.0));
}

TEST(ReverseModeAD, ExpDerivative) {
  // f(x) = exp(x),  df/dx = exp(x)  at x=2
  auto x = PV(2.0, 'x');
  auto g = reverse_mode_grad(exp(x));
  EXPECT_DOUBLE_EQ(g[0], std::exp(2.0));
}

TEST(ReverseModeAD, ChainRuleSinOfProduct) {
  // f(x,y) = sin(x*y)  at (2,3)
  // df/dx = cos(x*y)*y = cos(6)*3
  // df/dy = cos(x*y)*x = cos(6)*2
  auto x = PV(2.0, 'x');
  auto y = PV(3.0, 'y');
  auto g = reverse_mode_grad(sin(x * y));
  EXPECT_DOUBLE_EQ(g[0], std::cos(6.0) * 3.0);
  EXPECT_DOUBLE_EQ(g[1], std::cos(6.0) * 2.0);
}

TEST(ReverseModeAD, ThreeVariables) {
  // f(x,y,z) = x*y + y*z  at (2,3,4)
  // df/dx=y=3, df/dy=x+z=6, df/dz=y=3
  auto x = PV(2.0, 'x');
  auto y = PV(3.0, 'y');
  auto z = PV(4.0, 'z');
  auto g = reverse_mode_grad(x * y + y * z);
  EXPECT_DOUBLE_EQ(g[0], 3.0);
  EXPECT_DOUBLE_EQ(g[1], 6.0);
  EXPECT_DOUBLE_EQ(g[2], 3.0);
}

// ===========================================================================
// Equation — forward-mode Jacobian via derivative_tensor<1>
// ===========================================================================

TEST(EquationForward, TwoVariables) {
  // f(x,y) = (x*y, x+y) at (3,4) — J = [[4,3],[1,1]]
  Variable<double, 'x'> x{3.0};
  Variable<double, 'y'> y{4.0};
  auto ve = Equation(x * y, x + y);
  auto J = ve.derivative_tensor<1>();
  EXPECT_DOUBLE_EQ(J[0][0], 4.0); // ∂(x*y)/∂x = y = 4
  EXPECT_DOUBLE_EQ(J[0][1], 3.0); // ∂(x*y)/∂y = x = 3
  EXPECT_DOUBLE_EQ(J[1][0], 1.0); // ∂(x+y)/∂x
  EXPECT_DOUBLE_EQ(J[1][1], 1.0); // ∂(x+y)/∂y
}

TEST(EquationForward, AgreesWithSymbolic) {
  // derivative_tensor<1> must agree with symbolic Jacobian.
  double xv = 2.0, yv = 3.0;

  auto xs = PV(xv, 'x');
  auto ys = PV(yv, 'y');
  auto ve_sym = Equation(xs * xs, xs * ys, ys * ys);
  auto J_sym = ve_sym.symbolic_mode_jac();

  Variable<double, 'x'> xd{xv};
  Variable<double, 'y'> yd{yv};
  auto ve_fwd = Equation(xd * xd, xd * yd, yd * yd);
  auto J_fwd = ve_fwd.derivative_tensor<1>();

  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(J_fwd[i][j], J_sym[i][j]);
}

TEST(EquationForward, TrigJacobian) {
  // f(x,y) = (x*y, sin(x) + y*y) at (2,3) — J = [[3,2],[cos(2),6]]
  Variable<double, 'x'> x{2.0};
  Variable<double, 'y'> y{3.0};
  auto ve = Equation(x * y, sin(x) + y * y);
  auto J = ve.derivative_tensor<1>();
  EXPECT_DOUBLE_EQ(J[0][0], 3.0);
  EXPECT_DOUBLE_EQ(J[0][1], 2.0);
  EXPECT_NEAR(J[1][0], std::cos(2.0), 1e-12);
  EXPECT_DOUBLE_EQ(J[1][1], 6.0);
}

TEST(EquationForward, AgreesWithReverse) {
  Variable<double, 'x'> x{2.0};
  Variable<double, 'y'> y{3.0};
  auto ve_fwd = Equation(x * y, sin(x) + y * y);
  auto J_fwd = ve_fwd.derivative_tensor<1>();

  auto xs = PV(2.0, 'x');
  auto ys = PV(3.0, 'y');
  auto ve_rev = Equation(xs * ys, sin(xs) + ys * ys);
  auto J_rev = ve_rev.reverse_mode_jac();

  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_NEAR(J_rev[i][j], J_fwd[i][j], 1e-12);
}

TEST(EquationForward, StateRestoredAfterCall) {
  // After eval_jacobian_forward the expressions should evaluate at the
  // original point (dual parts zeroed out).
  auto x = PDV(3.0, 'x');
  auto y = PDV(4.0, 'y');
  auto k = PDV(4.0, 'y');
  k = 7.0;
  auto ve = make_equation(x * y, x + y);
  auto [v0, v1] = ve.evaluate();
  auto [v01, v02] = v0;
  EXPECT_DOUBLE_EQ(v01, 12.0);                 // x*y = 12
  EXPECT_DOUBLE_EQ(v02, 0.0);                  // dual part zeroed
  EXPECT_DOUBLE_EQ(v1.template get<0>(), 7.0); // x+y = 7
}

TEST(ReverseModeAD, ScalarLiteralCoercion) {
  // reverse_mode_grad(3*x*y + y*z) with plain integer/double literals
  auto x = PDV(2.0, 'x');
  auto y = PDV(3.0, 'y');
  auto z = PDV(4.0, 'z');
  auto expe = 3.0 * x * y + y * z;
  auto g = reverse_mode_grad(expe);
  EXPECT_DOUBLE_EQ(g[0], 9.0);  // df/dx = 3*y = 9
  EXPECT_DOUBLE_EQ(g[1], 10.0); // df/dy = 3*x + z = 10
  EXPECT_DOUBLE_EQ(g[2], 3.0);  // df/dz = y = 3
}

TEST(ReverseModeAD, ScalarOnRight) {
  // expr * scalar and expr + scalar
  auto x = PV(5.0, 'x');
  auto g = reverse_mode_grad(x * 4.0 + 1.0);
  EXPECT_DOUBLE_EQ(g[0], 4.0); // df/dx = 4
}

TEST(ReverseModeAD, AgreesWithForwardMode) {
  // f(x,y) = exp(x) * sin(y)  at (1, pi/4)
  // df/dx = exp(x)*sin(y), df/dy = exp(x)*cos(y)
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  auto x = PV(xv, 'x');
  auto y = PV(yv, 'y');
  auto g = reverse_mode_grad(exp(x) * sin(y));
  EXPECT_DOUBLE_EQ(g[0], std::exp(xv) * std::sin(yv));
  EXPECT_DOUBLE_EQ(g[1], std::exp(xv) * std::cos(yv));
}

// ===========================================================================
// Dual compound assignment operators
// ===========================================================================

TEST(DualCompoundAssign, PlusEq) {
  Dual<double> a{3.0, 1.0}, b{2.0, 0.5};
  a += b;
  EXPECT_DOUBLE_EQ(a.template get<0>(), 5.0);
  EXPECT_DOUBLE_EQ(a.template get<1>(), 1.5);
}

TEST(DualCompoundAssign, MinusEq) {
  Dual<double> a{3.0, 1.0}, b{2.0, 0.5};
  a -= b;
  EXPECT_DOUBLE_EQ(a.template get<0>(), 1.0);
  EXPECT_DOUBLE_EQ(a.template get<1>(), 0.5);
}

TEST(DualCompoundAssign, TimesEq) {
  // (3 + 1e)(2 + 0.5e) = 6 + (1*2 + 3*0.5)e = 6 + 3.5e
  Dual<double> a{3.0, 1.0}, b{2.0, 0.5};
  a *= b;
  EXPECT_DOUBLE_EQ(a.template get<0>(), 6.0);
  EXPECT_DOUBLE_EQ(a.template get<1>(), 3.5);
}

TEST(DualCompoundAssign, DivEq) {
  // (4 + 2e) / (2 + 0e) = 2 + 1e
  Dual<double> a{4.0, 2.0}, b{2.0, 0.0};
  a /= b;
  EXPECT_DOUBLE_EQ(a.template get<0>(), 2.0);
  EXPECT_DOUBLE_EQ(a.template get<1>(), 1.0);
}

TEST(DualCompoundAssign, ScalarAssign) {
  auto k = PDV(4.0, 'x');
  k = 7.0;
  EXPECT_DOUBLE_EQ(static_cast<Dual<double>>(k).template get<0>(), 7.0);
  EXPECT_DOUBLE_EQ(static_cast<Dual<double>>(k).template get<1>(), 0.0);
}

// ===========================================================================
// reverse_mode_gradient on Dual-valued (PDV) expressions
// ===========================================================================

TEST(ReverseModeAD_Dual, SingleVariable) {
  // f(x) = 3*x  at x=5,  df/dx = 3
  auto x = PDV(5.0, 'x');
  auto g = reverse_mode_grad(3.0 * x);
  EXPECT_DOUBLE_EQ(g[0], 3.0);
}

TEST(ReverseModeAD_Dual, TwoVariables) {
  // f(x,y) = x*y  at (3,4),  df/dx=4, df/dy=3
  auto x = PDV(3.0, 'x');
  auto y = PDV(4.0, 'y');
  auto g = reverse_mode_grad(x * y);
  EXPECT_DOUBLE_EQ(g[0], 4.0);
  EXPECT_DOUBLE_EQ(g[1], 3.0);
}

TEST(ReverseModeAD_Dual, ThreeVariables) {
  // f(x,y,z) = 3*x*y + y*z  at (2,3,4)
  // df/dx=9, df/dy=10, df/dz=3
  auto x = PDV(2.0, 'x');
  auto y = PDV(3.0, 'y');
  auto z = PDV(4.0, 'z');
  auto g = reverse_mode_grad(3.0 * x * y + y * z);
  EXPECT_DOUBLE_EQ(g[0], 9.0);
  EXPECT_DOUBLE_EQ(g[1], 10.0);
  EXPECT_DOUBLE_EQ(g[2], 3.0);
}

TEST(ReverseModeAD_Dual, TrigExp) {
  // f(x,y) = exp(x)*sin(y)  at (1, pi/4)
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  auto x = PDV(xv, 'x');
  auto y = PDV(yv, 'y');
  auto g = reverse_mode_grad(exp(x) * sin(y));
  EXPECT_DOUBLE_EQ(g[0], std::exp(xv) * std::sin(yv));
  EXPECT_DOUBLE_EQ(g[1], std::exp(xv) * std::cos(yv));
}

TEST(ReverseModeAD_Dual, AgreesWithPVResult) {
  // PDV and PV reverse mode must give identical scalar gradients
  double xv = 1.3, yv = 0.7;
  auto xp = PV(xv, 'x');
  auto yp = PV(yv, 'y');
  auto xd = PDV(xv, 'x');
  auto yd = PDV(yv, 'y');
  auto gp = reverse_mode_grad(xp * yp + sin(xp) + yp * yp + exp(xp + yp));
  auto gd = reverse_mode_grad(xd * yd + sin(xd) + yd * yd + exp(xd + yd));
  EXPECT_DOUBLE_EQ(gd[0], gp[0]);
  EXPECT_DOUBLE_EQ(gd[1], gp[1]);
}

// ===========================================================================
// Hessian via forward-over-reverse (eval_hessian)
// ===========================================================================

TEST(HessianTest, ForwardOverReverse_FunctionValues) {
  // f(x,y) = (x*y, x*x)  at (2, 3)  =>  f0=6, f1=4
  using D = Dual<double>;
  Variable<D, 'x'> x{D{2.0}};
  Variable<D, 'y'> y{D{3.0}};
  auto ve = Equation(x * y, x * x);
  (void)ve.reverse_mode_hess();
  auto f = ve.evaluate();
  EXPECT_DOUBLE_EQ(f[0].template get<0>(), 6.0);
  EXPECT_DOUBLE_EQ(f[1].template get<0>(), 4.0);
}

TEST(HessianTest, ForwardOverReverse_XY) {
  // H[f0] where f0(x,y) = x*y:
  // ∂²/∂x² = 0, ∂²/∂x∂y = 1, ∂²/∂y∂x = 1, ∂²/∂y² = 0
  using D = Dual<double>;
  Variable<D, 'x'> x{D{2.0}};
  Variable<D, 'y'> y{D{3.0}};
  auto ve = Equation(x * y, x * x);
  auto H = ve.reverse_mode_hess();
  EXPECT_DOUBLE_EQ(H[0][0][0], 0.0); // ∂²(x*y)/∂x²
  EXPECT_DOUBLE_EQ(H[0][0][1], 1.0); // ∂²(x*y)/∂x∂y
  EXPECT_DOUBLE_EQ(H[0][1][0], 1.0); // ∂²(x*y)/∂y∂x
  EXPECT_DOUBLE_EQ(H[0][1][1], 0.0); // ∂²(x*y)/∂y²
}

TEST(HessianTest, ForwardOverReverse_Quadratic) {
  // H[f1] where f1(x,y) = x²:
  // ∂²/∂x² = 2, all others = 0
  using D = Dual<double>;
  Variable<D, 'x'> x{D{2.0}};
  Variable<D, 'y'> y{D{3.0}};
  auto ve = Equation(x * y, x * x);
  auto H = ve.reverse_mode_hess();
  EXPECT_DOUBLE_EQ(H[1][0][0], 2.0);
  EXPECT_DOUBLE_EQ(H[1][0][1], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1][0], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1][1], 0.0);
}

TEST(HessianTest, ForwardOverReverse_WithValues) {
  // Same as above but via the values-accepting overload (sets point first).
  using D = Dual<double>;
  Variable<D, 'x'> x{D{0.0}};
  Variable<D, 'y'> y{D{0.0}};
  auto ve = Equation(x * y, x * x);
  std::array<double, 2> pt{2.0, 3.0};
  auto H = ve.reverse_mode_hess(pt);
  auto f = ve.evaluate();
  EXPECT_DOUBLE_EQ(f[0].template get<0>(), 6.0);
  EXPECT_DOUBLE_EQ(H[0][0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[1][0][0], 2.0);
}

TEST(HessianTest, ForwardOverReverse_TrigFunction) {
  // f(x,y) = (sin(x)*y, x+y)
  // H[f0]: ∂²(y*sin(x))/∂x² = -y*sin(x), ∂²/∂x∂y = cos(x), ∂²/∂y² = 0
  double xv = 1.0, yv = 2.0;
  using D = Dual<double>;
  Variable<D, 'x'> x{D{xv}};
  Variable<D, 'y'> y{D{yv}};
  auto ve = Equation(sin(x) * y, x + y);
  auto H = ve.reverse_mode_hess();
  EXPECT_NEAR(H[0][0][0], -yv * std::sin(xv), 1e-12); // -y*sin(x)
  EXPECT_NEAR(H[0][0][1], std::cos(xv), 1e-12);       // cos(x)
  EXPECT_NEAR(H[0][1][0], std::cos(xv), 1e-12);       // symmetric
  EXPECT_NEAR(H[0][1][1], 0.0, 1e-12);
}

TEST(HessianTest, ForwardOverReverse_Symmetric) {
  // Hessian must be symmetric for smooth functions.
  // f(x,y) = (exp(x*y),)  — we need 2 outputs so wrap it
  double xv = 0.5, yv = 1.5;
  using D = Dual<double>;
  Variable<D, 'x'> x{D{xv}};
  Variable<D, 'y'> y{D{yv}};
  auto ve = Equation(exp(x * y), x + y);
  auto H = ve.reverse_mode_hess();
  EXPECT_NEAR(H[0][0][1], H[0][1][0], 1e-12);
}

// ===========================================================================
// Equation derivative_tensor<2> — forward-mode Hessian, plain scalar variables
// ===========================================================================

TEST(HessianForwardTest, FunctionValues) {
  // f(x,y) = (x*y, x*x) at (2, 3) => f0=6, f1=4
  Variable<double, 'x'> x{2.0};
  Variable<double, 'y'> y{3.0};
  auto ve = Equation(x * y, x * x);
  auto f = ve.evaluate();
  EXPECT_DOUBLE_EQ(f[0], 6.0);
  EXPECT_DOUBLE_EQ(f[1], 4.0);
}

TEST(HessianForwardTest, XY) {
  // H[f0] where f0(x,y) = x*y — H = [[0,1],[1,0]]
  Variable<double, 'x'> x{2.0};
  Variable<double, 'y'> y{3.0};
  auto ve = Equation(x * y, x * x);
  auto H = ve.derivative_tensor<2>();
  EXPECT_DOUBLE_EQ(H[0][0][0], 0.0);
  EXPECT_DOUBLE_EQ(H[0][0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[0][1][0], 1.0);
  EXPECT_DOUBLE_EQ(H[0][1][1], 0.0);
}

TEST(HessianForwardTest, Quadratic) {
  // H[f1] where f1(x,y) = x² — H = [[2,0],[0,0]]
  Variable<double, 'x'> x{2.0};
  Variable<double, 'y'> y{3.0};
  auto ve = Equation(x * y, x * x);
  auto H = ve.derivative_tensor<2>();
  EXPECT_DOUBLE_EQ(H[1][0][0], 2.0);
  EXPECT_DOUBLE_EQ(H[1][0][1], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1][0], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1][1], 0.0);
}

TEST(HessianForwardTest, AgreesWithForwardOverReverse) {
  // derivative_tensor<2> must match reverse-mode Hessian for f(x,y) = (x*y,
  // x*x).
  double xv = 1.5, yv = 2.5;

  using D = Dual<double>;
  Variable<D, 'x'> xr{D{xv}};
  Variable<D, 'y'> yr{D{yv}};
  auto ve_rev = Equation(xr * yr, xr * xr);
  auto H_rev = ve_rev.reverse_mode_hess();

  Variable<double, 'x'> xf{xv};
  Variable<double, 'y'> yf{yv};
  auto ve_fwd = Equation(xf * yf, xf * xf);
  auto H_fwd = ve_fwd.derivative_tensor<2>();

  for (int k = 0; k < 2; ++k)
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        EXPECT_NEAR(H_fwd[k][i][j], H_rev[k][i][j], 1e-12);
}

TEST(HessianForwardTest, WithValues) {
  // Values-accepting overload: variables at (0,0), evaluated at (2,3).
  Variable<double, 'x'> x{0.0};
  Variable<double, 'y'> y{0.0};
  auto ve = Equation(x * y, x * x);
  auto H = ve.derivative_tensor<2>(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(H[0][0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[1][0][0], 2.0);
}

// ===========================================================================
// Scalar Hessian free functions (gradient.hpp)
// ===========================================================================

TEST(ScalarHessianTest, ReverseMode_XY) {
  // f(x,y) = x*y  =>  H = [[0, 1], [1, 0]]
  using D = Dual<double>;
  Variable<D, 'x'> x{D{2.0}};
  Variable<D, 'y'> y{D{3.0}};
  auto expr = x * y;
  auto H = reverse_mode_hess(expr, std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(H[0][0], 0.0);
  EXPECT_DOUBLE_EQ(H[0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[1][0], 1.0);
  EXPECT_DOUBLE_EQ(H[1][1], 0.0);
}

TEST(ScalarHessianTest, ReverseMode_QuadraticForm) {
  // f(x,y) = x*x + 2*y*y  =>  H = [[2, 0], [0, 4]]
  using D = Dual<double>;
  Variable<D, 'x'> x{D{1.0}};
  Variable<D, 'y'> y{D{1.0}};
  auto expr = x * x + PC(D{2.0}) * y * y;
  auto H = reverse_mode_hess(expr, std::array{1.0, 1.0});
  EXPECT_DOUBLE_EQ(H[0][0], 2.0);
  EXPECT_DOUBLE_EQ(H[0][1], 0.0);
  EXPECT_DOUBLE_EQ(H[1][0], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1], 4.0);
}

TEST(ScalarHessianTest, ReverseMode_Symmetric) {
  // H must be symmetric for smooth f.
  using D = Dual<double>;
  Variable<D, 'x'> x{D{0.5}};
  Variable<D, 'y'> y{D{1.5}};
  auto expr = exp(x * y);
  auto H = reverse_mode_hess(expr, std::array{0.5, 1.5});
  EXPECT_NEAR(H[0][1], H[1][0], 1e-12);
}

TEST(ScalarHessianTest, ForwardMode_XY) {
  // f(x,y) = x*y  =>  H = [[0, 1], [1, 0]] — plain double variables
  Variable<double, 'x'> x{2.0};
  Variable<double, 'y'> y{3.0};
  auto expr = x * y;
  auto H = derivative_tensor<2>(expr, std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(H[0][0], 0.0);
  EXPECT_DOUBLE_EQ(H[0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[1][0], 1.0);
  EXPECT_DOUBLE_EQ(H[1][1], 0.0);
}

TEST(ScalarHessianTest, ForwardMode_QuadraticForm) {
  // f(x,y) = x*x + 2*y*y  =>  H = [[2, 0], [0, 4]]
  Variable<double, 'x'> x{1.0};
  Variable<double, 'y'> y{1.0};
  auto expr = x * x + PC(2.0) * y * y;
  auto H = derivative_tensor<2>(expr, std::array{1.0, 1.0});
  EXPECT_DOUBLE_EQ(H[0][0], 2.0);
  EXPECT_DOUBLE_EQ(H[0][1], 0.0);
  EXPECT_DOUBLE_EQ(H[1][0], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1], 4.0);
}

TEST(ScalarHessianTest, ReverseMode_NoValues) {
  // No-values overload reads current variable state via collect().
  using D = Dual<double>;
  Variable<D, 'x'> x{D{2.0}};
  Variable<D, 'y'> y{D{3.0}};
  auto expr = x * y;
  auto H = reverse_mode_hess(expr);
  EXPECT_DOUBLE_EQ(H[0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[1][0], 1.0);
  EXPECT_DOUBLE_EQ(H[0][0], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1], 0.0);
}

TEST(ScalarHessianTest, ForwardMode_NoValues) {
  Variable<double, 'x'> x{2.0};
  Variable<double, 'y'> y{3.0};
  auto expr = x * y;
  auto H = derivative_tensor<2>(expr);
  EXPECT_DOUBLE_EQ(H[0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[1][0], 1.0);
  EXPECT_DOUBLE_EQ(H[0][0], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1], 0.0);
}

TEST(ScalarHessianTest, ForwardAgreesWithReverse) {
  // derivative_tensor<2> must agree with reverse-mode Hessian on exp(x*y).
  double xv = 0.5, yv = 1.5;

  using D = Dual<double>;
  Variable<D, 'x'> xr{D{xv}};
  Variable<D, 'y'> yr{D{yv}};
  auto expr_r = exp(xr * yr);
  auto H_rev = reverse_mode_hess(expr_r, std::array{xv, yv});

  Variable<double, 'x'> xf{xv};
  Variable<double, 'y'> yf{yv};
  auto expr_f = exp(xf * yf);
  auto H_fwd = derivative_tensor<2>(expr_f, std::array{xv, yv});

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      EXPECT_NEAR(H_fwd[i][j], H_rev[i][j], 1e-12);
}

// ===========================================================================
// DerivativeTensorTest — derivative_tensor<Order> free function and Equation
// ===========================================================================

TEST(DerivativeTensorTest, Order1_ScalarVariable) {
  // Plain scalar Variable<double>, f(x) = x^3, f'(3) = 27
  Variable<double, 'x'> x{3.0};
  auto expr = x * x * x;
  auto T1 = derivative_tensor<1>(expr, std::array{3.0});
  EXPECT_NEAR(T1[0], 27.0, 1e-12);
}

TEST(DerivativeTensorTest, Order1_MatchesGradient) {
  // derivative_tensor<1> should match reverse-mode gradient.
  double xv = 1.5, yv = 2.5;
  Variable<double, 'x'> x{xv};
  Variable<double, 'y'> y{yv};
  auto expr = x * x + x * y; // df/dx = 2x+y, df/dy = x

  auto T1 = derivative_tensor<1>(expr, std::array{xv, yv});
  EXPECT_NEAR(T1[0], 2 * xv + yv, 1e-12);
  EXPECT_NEAR(T1[1], xv, 1e-12);
}

TEST(DerivativeTensorTest, Order2_ScalarVariable) {
  // Plain double: f(x) = x^3, f''(2) = 12
  Variable<double, 'x'> x{2.0};
  auto expr = x * x * x;
  auto T2 = derivative_tensor<2>(expr, std::array{2.0});
  EXPECT_NEAR(T2[0][0], 12.0, 1e-12);
}

TEST(DerivativeTensorTest, Order2_MatchesHessian) {
  // derivative_tensor<2> must agree with reverse-mode Hessian.
  double xv = 0.7, yv = 1.3;
  using D = Dual<double>;
  Variable<D, 'x'> xr{D{xv}};
  Variable<D, 'y'> yr{D{yv}};
  auto expr_r = exp(xr * yr);
  auto H_rev = reverse_mode_hess(expr_r, std::array{xv, yv});

  Variable<double, 'x'> xf{xv};
  Variable<double, 'y'> yf{yv};
  auto expr_f = exp(xf * yf);
  auto H_fwd = derivative_tensor<2>(expr_f, std::array{xv, yv});

  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_NEAR(H_fwd[i][j], H_rev[i][j], 1e-12);
}

TEST(DerivativeTensorTest, Order3_Polynomial) {
  // f(x) = x^4, f'''(x) = 24x, f'''(2) = 48
  Variable<double, 'x'> x{2.0};
  auto expr = x * x * x * x;
  auto T3 = derivative_tensor<3>(expr, std::array{2.0});
  EXPECT_NEAR(T3[0][0][0], 48.0, 1e-9);
}

TEST(DerivativeTensorTest, NoArgReadsCurrentValues) {
  // No-arg overload reads current values from variables.
  const double x0 = std::numbers::pi / 4.0;
  Variable<double, 'x'> x{x0};
  auto expr = sin(x);
  auto T2 = derivative_tensor<2>(expr);
  EXPECT_NEAR(T2[0][0], -std::sin(x0), 1e-12);
}

TEST(DerivativeTensorTest, MixedPartials_Symmetric) {
  // Hessian must be symmetric for smooth f.
  double xv = 0.5, yv = 1.5;
  Variable<double, 'x'> x{xv};
  Variable<double, 'y'> y{yv};
  auto expr = exp(x * y);
  auto H = derivative_tensor<2>(expr, std::array{xv, yv});
  EXPECT_NEAR(H[0][1], H[1][0], 1e-12);
}

TEST(DerivativeTensorTest, Equation_Order1_IsJacobian) {
  // derivative_tensor<1> for Equation must match reverse-mode Jacobian.
  double xv = 1.0, yv = 2.0;
  Variable<double, 'x'> xf{xv};
  Variable<double, 'y'> yf{yv};
  auto ve_fwd = Equation(xf * yf, xf * xf);
  auto J_fwd = ve_fwd.derivative_tensor<1>();

  auto xr = PV(xv, 'x');
  auto yr = PV(yv, 'y');
  auto ve_rev = Equation(xr * yr, xr * xr);
  auto J_rev = ve_rev.reverse_mode_jac();

  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_NEAR(J_fwd[i][j], J_rev[i][j], 1e-12);
}

TEST(DerivativeTensorTest, Equation_Order2_IsHessianStack) {
  // derivative_tensor<2> shape [2,2,2] must match per-output reverse Hessians.
  double xv = 1.5, yv = 2.5;
  Variable<double, 'x'> xf{xv};
  Variable<double, 'y'> yf{yv};
  auto ve_fwd = Equation(xf * yf, xf * xf);
  auto H_fwd = ve_fwd.derivative_tensor<2>(std::array{xv, yv});
  using D = Dual<double>;
  Variable<D, 'x'> xr{D{xv}};
  Variable<D, 'y'> yr{D{yv}};
  auto ve_rev = Equation(xr * yr, xr * xr);
  auto H_rev = ve_rev.reverse_mode_hess();

  for (std::size_t k = 0; k < 2; ++k)
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 2; ++j)
        EXPECT_NEAR(H_fwd[k][i][j], H_rev[k][i][j], 1e-12);
}

TEST(DerivativeTensorTest, Equation_Order3_TrigPolynomial) {
  // f(x) = sin(x)*x^2, f'''(x) = -sin(x)*x^2 - ... validated at x=1
  // Use finite differences as ground truth for 3rd order.
  // Actually use a pure polynomial: f(x) = x^4, f'''(x) = 24x, f'''(1) = 24
  Variable<double, 'x'> x{1.0};
  auto ve = Equation(x * x * x * x);
  auto T3 = ve.derivative_tensor<3>(std::array{1.0});
  EXPECT_NEAR(T3[0][0][0][0], 24.0, 1e-9);
}

// ===========================================================================
// Tutorial tests — based on autodiff.github.io/tutorials/
// Each test section maps to one tutorial page.
// ===========================================================================

// ---------------------------------------------------------------------------
// Forward Tutorial 1 — Single-variable function
// f(x) = 1 + x + x² + 1/x + log(x)
// ---------------------------------------------------------------------------

TEST(TutorialForward, SingleVar_SymbolicValueAndDerivative) {
  double x0 = 2.0;
  auto x = PV(x0, 'x');
  auto f = PC(1.0) + x + x * x + PC(1.0) / x + log(x);
  EXPECT_NEAR(f.eval(), 1.0 + x0 + x0 * x0 + 1.0 / x0 + std::log(x0), 1e-12);
  // f'(x) = 1 + 2x - 1/x² + 1/x
  double expected_df = 1.0 + 2.0 * x0 - 1.0 / (x0 * x0) + 1.0 / x0;
  EXPECT_NEAR(f.derivative().eval(), expected_df, 1e-12);
}

TEST(TutorialForward, SingleVar_DualNumbers) {
  // Same function via forward-mode dual numbers (seed = 1)
  double x0 = 2.0;
  Variable<Dual<double>, 'x'> x{Dual<double>{x0, 1.0}};
  auto f = 1.0 + x + x * x + 1.0 / x + log(x);
  auto [fv, df] = f.eval();
  EXPECT_NEAR(fv, 1.0 + x0 + x0 * x0 + 1.0 / x0 + std::log(x0), 1e-12);
  EXPECT_NEAR(df, 1.0 + 2.0 * x0 - 1.0 / (x0 * x0) + 1.0 / x0, 1e-12);
}

TEST(TutorialForward, SingleVar_ReverseMode) {
  double x0 = 2.0;
  auto x = PV(x0, 'x');
  auto f = PC(1.0) + x + x * x + PC(1.0) / x + log(x);
  auto g = reverse_mode_grad(f);
  EXPECT_NEAR(g[0], 1.0 + 2.0 * x0 - 1.0 / (x0 * x0) + 1.0 / x0, 1e-12);
}

// ---------------------------------------------------------------------------
// Forward Tutorial 3 / Reverse Tutorial 2 — Multi-variable function
// f(x,y,z) = 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z)
// ---------------------------------------------------------------------------

TEST(TutorialMultiVar, Value) {
  double xv = 1.0, yv = 2.0, zv = 3.0;
  auto x = PV(xv, 'x');
  auto y = PV(yv, 'y');
  auto z = PV(zv, 'z');
  auto f = PC(1.0) + x + y + z + x * y + y * z + x * z + x * y * z +
           exp(x / y + y / z);
  double expected = 1.0 + xv + yv + zv + xv * yv + yv * zv + xv * zv +
                    xv * yv * zv + std::exp(xv / yv + yv / zv);
  EXPECT_NEAR(f.eval(), expected, 1e-12);
}

TEST(TutorialMultiVar, SymbolicPartials) {
  double xv = 1.0, yv = 2.0, zv = 3.0;
  auto x = PV(xv, 'x');
  auto y = PV(yv, 'y');
  auto z = PV(zv, 'z');
  auto f = PC(1.0) + x + y + z + x * y + y * z + x * z + x * y * z +
           exp(x / y + y / z);
  auto eq = Equation(f);
  auto [dx, dy, dz] = eq.eval_derivatives();
  double e = std::exp(xv / yv + yv / zv);
  // df/dx = 1 + y + z + y*z + exp(.)/y
  EXPECT_NEAR(dx, 1.0 + yv + zv + yv * zv + e / yv, 1e-10);
  // df/dy = 1 + x + z + x*z + exp(.)*(−x/y² + 1/z)
  EXPECT_NEAR(dy, 1.0 + xv + zv + xv * zv + e * (-xv / (yv * yv) + 1.0 / zv),
              1e-10);
  // df/dz = 1 + y + x + x*y + exp(.)*(−y/z²)
  EXPECT_NEAR(dz, 1.0 + yv + xv + xv * yv + e * (-yv / (zv * zv)), 1e-10);
}

TEST(TutorialMultiVar, ReverseGradient) {
  double xv = 1.0, yv = 2.0, zv = 3.0;
  auto x = PV(xv, 'x');
  auto y = PV(yv, 'y');
  auto z = PV(zv, 'z');
  auto f = PC(1.0) + x + y + z + x * y + y * z + x * z + x * y * z +
           exp(x / y + y / z);
  auto g = reverse_mode_grad(f);
  double e = std::exp(xv / yv + yv / zv);
  EXPECT_NEAR(g[0], 1.0 + yv + zv + yv * zv + e / yv, 1e-10);
  EXPECT_NEAR(g[1], 1.0 + xv + zv + xv * zv + e * (-xv / (yv * yv) + 1.0 / zv),
              1e-10);
  EXPECT_NEAR(g[2], 1.0 + yv + xv + xv * yv + e * (-yv / (zv * zv)), 1e-10);
}

TEST(TutorialMultiVar, ForwardReverseGradientAgree) {
  double xv = 1.0, yv = 2.0, zv = 3.0;
  // Forward mode via derivative_tensor
  Variable<double, 'x'> xf{xv};
  Variable<double, 'y'> yf{yv};
  Variable<double, 'z'> zf{zv};
  auto f_fwd = 1.0 + xf + yf + zf + xf * yf + yf * zf + xf * zf + xf * yf * zf +
               exp(xf / yf + yf / zf);
  auto T1 = derivative_tensor<1>(f_fwd, std::array{xv, yv, zv});
  // Reverse mode
  auto xr = PV(xv, 'x');
  auto yr = PV(yv, 'y');
  auto zr = PV(zv, 'z');
  auto f_rev = PC(1.0) + xr + yr + zr + xr * yr + yr * zr + xr * zr +
               xr * yr * zr + exp(xr / yr + yr / zr);
  auto g = reverse_mode_grad(f_rev);
  EXPECT_NEAR(T1[0], g[0], 1e-10);
  EXPECT_NEAR(T1[1], g[1], 1e-10);
  EXPECT_NEAR(T1[2], g[2], 1e-10);
}

// ---------------------------------------------------------------------------
// Forward Tutorial 6 — Gradient vector of scalar function
// ---------------------------------------------------------------------------

TEST(TutorialGradient, ForwardModeDerivativeTensor) {
  // f(x,y) = sin(x)*cos(y) + exp(x*y)  at (1, 0.5)
  double xv = 1.0, yv = 0.5;
  Variable<double, 'x'> x{xv};
  Variable<double, 'y'> y{yv};
  auto f = sin(x) * cos(y) + exp(x * y);
  auto g = derivative_tensor<1>(f, std::array{xv, yv});
  EXPECT_NEAR(g[0], std::cos(xv) * std::cos(yv) + yv * std::exp(xv * yv),
              1e-12);
  EXPECT_NEAR(g[1], -std::sin(xv) * std::sin(yv) + xv * std::exp(xv * yv),
              1e-12);
}

TEST(TutorialGradient, ForwardReverseAgree) {
  double xv = 1.0, yv = 0.5;
  Variable<double, 'x'> xf{xv};
  Variable<double, 'y'> yf{yv};
  auto T1 = derivative_tensor<1>(sin(xf) * cos(yf) + exp(xf * yf),
                                 std::array{xv, yv});
  auto xr = PV(xv, 'x');
  auto yr = PV(yv, 'y');
  auto g = reverse_mode_grad(sin(xr) * cos(yr) + exp(xr * yr));
  EXPECT_NEAR(T1[0], g[0], 1e-12);
  EXPECT_NEAR(T1[1], g[1], 1e-12);
}

// ---------------------------------------------------------------------------
// Forward Tutorial 11 — Higher-order cross derivatives (dual4th equivalent)
// Uses derivative_tensor<4>
// ---------------------------------------------------------------------------

TEST(TutorialHigherOrder, FourthOrder_SinX) {
  // f(x) = sin(x): f''''(x) = sin(x)
  double x0 = std::numbers::pi / 4.0;
  Variable<double, 'x'> x{x0};
  auto T4 = derivative_tensor<4>(sin(x), std::array{x0});
  EXPECT_NEAR(T4[0][0][0][0], std::sin(x0), 1e-9);
}

TEST(TutorialHigherOrder, FourthOrder_ExpX) {
  // f(x) = exp(x): all derivatives = exp(x)
  double x0 = 0.7;
  Variable<double, 'x'> x{x0};
  double ev = std::exp(x0);
  EXPECT_NEAR(derivative_tensor<2>(exp(x), std::array{x0})[0][0], ev, 1e-12);
  EXPECT_NEAR(derivative_tensor<3>(exp(x), std::array{x0})[0][0][0], ev, 1e-12);
  EXPECT_NEAR(derivative_tensor<4>(exp(x), std::array{x0})[0][0][0][0], ev,
              1e-9);
}

TEST(TutorialHigherOrder, FourthOrder_AllCrossPartialsOfSinXplusY) {
  // f(x,y) = sin(x+y): all 4th-order partials equal sin(x+y)
  double xv = 1.0, yv = 1.0;
  Variable<double, 'x'> x{xv};
  Variable<double, 'y'> y{yv};
  auto T4 = derivative_tensor<4>(sin(x + y), std::array{xv, yv});
  double expected = std::sin(xv + yv);
  for (int i : {0, 1})
    for (int j : {0, 1})
      for (int k : {0, 1})
        for (int l : {0, 1})
          EXPECT_NEAR(T4[i][j][k][l], expected, 1e-8);
}

TEST(TutorialHigherOrder, ThirdOrder_MixedPartial) {
  // f(x,y) = exp(x)*y²:  d³f/dx²dy = 2y*exp(x)  at (1,2)
  double xv = 1.0, yv = 2.0;
  Variable<double, 'x'> x{xv};
  Variable<double, 'y'> y{yv};
  auto T3 = derivative_tensor<3>(exp(x) * y * y, std::array{xv, yv});
  EXPECT_NEAR(T3[0][0][1], 2.0 * yv * std::exp(xv), 1e-10);
  EXPECT_NEAR(T3[1][1][1], 0.0, 1e-10); // d³(exp(x)*y²)/dy³ = 0
}

// ---------------------------------------------------------------------------
// Forward Tutorial 12/13 — Directional derivatives (scalar and vector)
// ---------------------------------------------------------------------------

TEST(TutorialDirectional, FirstOrder_DualSeeding) {
  // f(x,y) = exp(x)*sin(y), direction u = (1/√2, 1/√2) at (1, π/4)
  // Df·u = exp(1)*sin(π/4)/√2 + exp(1)*cos(π/4)/√2 = exp(1)
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  double u = 1.0 / std::sqrt(2.0);
  Variable<Dual<double>, 'x'> x{Dual<double>{xv, u}};
  Variable<Dual<double>, 'y'> y{Dual<double>{yv, u}};
  auto [fv, dfdu] = (exp(x) * sin(y)).eval();
  EXPECT_NEAR(fv, std::exp(xv) * std::sin(yv), 1e-12);
  EXPECT_NEAR(dfdu, std::exp(xv), 1e-12);
}

TEST(TutorialDirectional, FirstOrder_ViaGradientDot) {
  // Same result computed by dotting derivative_tensor<1> with direction
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  double u = 1.0 / std::sqrt(2.0);
  Variable<double, 'x'> x{xv};
  Variable<double, 'y'> y{yv};
  auto T1 = derivative_tensor<1>(exp(x) * sin(y), std::array{xv, yv});
  EXPECT_NEAR(T1[0] * u + T1[1] * u, std::exp(xv), 1e-12);
}

TEST(TutorialDirectional, SecondOrder_HessianContraction) {
  // D²f(u) = uᵀ H u for f(x,y)=exp(x)*sin(y), u=(1/√2,1/√2) at (1,π/4)
  // Expected = exp(1)*cos(π/4) = exp(1)/√2
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  double u = 1.0 / std::sqrt(2.0);
  Variable<double, 'x'> x{xv};
  Variable<double, 'y'> y{yv};
  auto H = derivative_tensor<2>(exp(x) * sin(y), std::array{xv, yv});
  double d2fdu2 =
      H[0][0] * u * u + H[0][1] * u * u + H[1][0] * u * u + H[1][1] * u * u;
  EXPECT_NEAR(d2fdu2, std::exp(xv) * std::cos(yv), 1e-12);
}

TEST(TutorialDirectional, VectorFunction_JacobianTimesDirection) {
  // f(x,y) = (exp(x)*sin(y), x*y), direction u = (1/√2, 1/√2) at (1, π/4)
  // Jf*u component-0 = exp(1), component-1 = (x+y)/√2
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  double u = 1.0 / std::sqrt(2.0);
  Variable<double, 'x'> x{xv};
  Variable<double, 'y'> y{yv};
  auto ve = Equation(exp(x) * sin(y), x * y);
  auto J = ve.derivative_tensor<1>(std::array{xv, yv});
  double df0du = J[0][0] * u + J[0][1] * u;
  double df1du = J[1][0] * u + J[1][1] * u;
  EXPECT_NEAR(df0du, std::exp(xv), 1e-12);
  EXPECT_NEAR(df1du, (xv + yv) / std::sqrt(2.0), 1e-12);
}

// ---------------------------------------------------------------------------
// Forward Tutorial 14/15 — Taylor series (scalar and vector)
// ---------------------------------------------------------------------------

TEST(TutorialTaylor, ScalarSin_FourthOrderAccuracy) {
  // Taylor expansion of sin(x) around x₀=π/4 up to 4th order evaluated at x₀+h.
  // Residual is O(h^5).
  double x0 = std::numbers::pi / 4.0;
  double h = 0.05;
  Variable<double, 'x'> x{x0};
  auto f = sin(x);
  double f0 = std::sin(x0);
  auto T1 = derivative_tensor<1>(f, std::array{x0});
  auto T2 = derivative_tensor<2>(f, std::array{x0});
  auto T3 = derivative_tensor<3>(f, std::array{x0});
  auto T4 = derivative_tensor<4>(f, std::array{x0});
  double taylor = f0 + T1[0] * h + T2[0][0] * h * h / 2.0 +
                  T3[0][0][0] * h * h * h / 6.0 +
                  T4[0][0][0][0] * h * h * h * h / 24.0;
  EXPECT_NEAR(taylor, std::sin(x0 + h), 1e-7);
}

TEST(TutorialTaylor, VectorFunction_SecondOrderAccuracy) {
  // f(x,y) = (sin(x)*cos(y), exp(x+y)) at (0,0); evaluate Taylor at (h,h).
  // Component-1 = exp(2h): Taylor to order 2 gives 1+2h+2h², residual O(h³).
  double xv = 0.0, yv = 0.0, h = 0.1;
  Variable<double, 'x'> x{xv};
  Variable<double, 'y'> y{yv};
  auto ve = Equation(sin(x) * cos(y), exp(x + y));
  auto f0 = ve.evaluate();
  auto J = ve.derivative_tensor<1>(std::array{xv, yv});
  auto H = ve.derivative_tensor<2>(std::array{xv, yv});
  // Component 1: f0[1] + (J[1][0]+J[1][1])*h +
  // 0.5*(H[1][0][0]+2H[1][0][1]+H[1][1][1])*h²
  double taylor1 = f0[1] + (J[1][0] + J[1][1]) * h +
                   0.5 * (H[1][0][0] + 2.0 * H[1][0][1] + H[1][1][1]) * h * h;
  EXPECT_NEAR(taylor1, std::exp(2.0 * h), 2e-3);
}

// ---------------------------------------------------------------------------
// Reverse Tutorial 3 — Functions with conditionals / re-evaluation via update
// ---------------------------------------------------------------------------

TEST(TutorialReverseConditionals, AbsGradientChangesSign) {
  // abs(x) is piecewise: gradient = +1 for x>0, −1 for x<0
  EXPECT_DOUBLE_EQ(reverse_mode_grad(abs(PV(3.0, 'x')))[0], 1.0);
  EXPECT_DOUBLE_EQ(reverse_mode_grad(abs(PV(-3.0, 'x')))[0], -1.0);
}

TEST(TutorialReverseConditionals, UpdateReevaluatesDerivatives) {
  // f(x) = x² + sin(x): update x from 2→5 and verify derivative re-evaluates.
  // Symbolic jacobian re-evaluates live variables after update().
  Variable<double, 'x'> x{2.0};
  auto eq = Equation(x * x + sin(x));
  EXPECT_NEAR(eq.evaluate(), 4.0 + std::sin(2.0), 1e-12);
  EXPECT_NEAR(eq.eval_derivatives()[0], 4.0 + std::cos(2.0),
              1e-12); // 2x + cos(x)

  using Syms = decltype(eq)::symbols;
  eq.update(Syms{}, std::array{5.0}); // x←5
  EXPECT_NEAR(eq.evaluate(), 25.0 + std::sin(5.0), 1e-12);
  EXPECT_NEAR(eq.eval_derivatives()[0], 10.0 + std::cos(5.0), 1e-12);
}

// ---------------------------------------------------------------------------
// Reverse Tutorial 4 — Functions with parameters (Constants)
// ---------------------------------------------------------------------------

TEST(TutorialReverseParams, SingleParameter) {
  // f(x; p) = p * sin(x): only x is a Variable; df/dx = p*cos(x)
  double xv = 1.0, p = 3.0;
  auto x = PV(xv, 'x');
  auto g = reverse_mode_grad(PC(p) * sin(x));
  EXPECT_NEAR(g[0], p * std::cos(xv), 1e-12);
}

TEST(TutorialReverseParams, MultiVarWithParameter) {
  // f(x, y; p) = p * x * y: df/dx = p*y, df/dy = p*x
  double xv = 2.0, yv = 3.0, p = 4.0;
  auto x = PV(xv, 'x');
  auto y = PV(yv, 'y');
  auto g = reverse_mode_grad(PC(p) * x * y);
  EXPECT_NEAR(g[0], p * yv, 1e-12);
  EXPECT_NEAR(g[1], p * xv, 1e-12);
}

// ---------------------------------------------------------------------------
// Reverse Tutorial 6 — Hessian matrix
// ---------------------------------------------------------------------------

TEST(TutorialReverseHessian, QuadraticForm) {
  // f(x,y) = x² + x*y + y²: H = [[2, 1], [1, 2]] (constant, independent of
  // point)
  using D = Dual<double>;
  Variable<D, 'x'> x{D{2.0}};
  Variable<D, 'y'> y{D{3.0}};
  auto f = x * x + x * y + y * y;
  auto H = reverse_mode_hess(f, std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(H[0][0], 2.0);
  EXPECT_DOUBLE_EQ(H[0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[1][0], 1.0);
  EXPECT_DOUBLE_EQ(H[1][1], 2.0);
}

TEST(TutorialReverseHessian, TrigFunction) {
  // f(x,y) = sin(x)*cos(y): H = [[-sin(x)cos(y), -cos(x)sin(y)], same
  // symmetric]
  using D = Dual<double>;
  double xv = std::numbers::pi / 4.0, yv = std::numbers::pi / 6.0;
  Variable<D, 'x'> x{D{xv}};
  Variable<D, 'y'> y{D{yv}};
  auto f = sin(x) * cos(y);
  auto H = reverse_mode_hess(f, std::array{xv, yv});
  EXPECT_NEAR(H[0][0], -std::sin(xv) * std::cos(yv), 1e-12);
  EXPECT_NEAR(H[0][1], -std::cos(xv) * std::sin(yv), 1e-12);
  EXPECT_NEAR(H[1][0], -std::cos(xv) * std::sin(yv), 1e-12);
  EXPECT_NEAR(H[1][1], -std::sin(xv) * std::cos(yv), 1e-12);
}

TEST(TutorialReverseHessian, GradientFromSameExpression) {
  // After reverse_mode_hess restores the point, reverse_mode_grad gives ∇f
  using D = Dual<double>;
  double xv = 2.0, yv = 3.0;
  Variable<D, 'x'> x{D{xv}};
  Variable<D, 'y'> y{D{yv}};
  auto f = x * x + x * y + y * y;
  (void)reverse_mode_hess(f, std::array{xv, yv}); // restores to (xv, yv)
  auto g = reverse_mode_grad(f);
  EXPECT_NEAR(g[0], 2.0 * xv + yv, 1e-12); // df/dx = 2x + y
  EXPECT_NEAR(g[1], xv + 2.0 * yv, 1e-12); // df/dy = x + 2y
}

// ---------------------------------------------------------------------------
// Reverse Tutorial 7/8 — Higher-order derivatives
// ---------------------------------------------------------------------------

TEST(TutorialReverseHigherOrder, SingleVar_SecondAndThird) {
  // f(x) = sin(x): f''(x) = -sin(x), f'''(x) = -cos(x)
  double x0 = 1.0;
  Variable<double, 'x'> x{x0};
  EXPECT_NEAR(derivative_tensor<2>(sin(x), std::array{x0})[0][0], -std::sin(x0),
              1e-12);
  EXPECT_NEAR(derivative_tensor<3>(sin(x), std::array{x0})[0][0][0],
              -std::cos(x0), 1e-12);
}

TEST(TutorialReverseHigherOrder, MultiVar_Hessian) {
  // f(x,y) = x²*y + y³: H[0][0]=2y, H[0][1]=H[1][0]=2x, H[1][1]=6y
  using D = Dual<double>;
  double xv = 2.0, yv = 3.0;
  Variable<D, 'x'> x{D{xv}};
  Variable<D, 'y'> y{D{yv}};
  auto f = x * x * y + y * y * y;
  auto H = reverse_mode_hess(f, std::array{xv, yv});
  EXPECT_NEAR(H[0][0], 2.0 * yv, 1e-12);
  EXPECT_NEAR(H[0][1], 2.0 * xv, 1e-12);
  EXPECT_NEAR(H[1][0], 2.0 * xv, 1e-12);
  EXPECT_NEAR(H[1][1], 6.0 * yv, 1e-12);
}

TEST(TutorialReverseHigherOrder, ForwardReverseHessianAgree) {
  // derivative_tensor<2> and reverse_mode_hess must agree on sin(x)*exp(y)
  double xv = 0.8, yv = 1.2;
  Variable<double, 'x'> xf{xv};
  Variable<double, 'y'> yf{yv};
  auto f_fwd = sin(xf) * exp(yf);
  auto H_fwd = derivative_tensor<2>(f_fwd, std::array{xv, yv});
  using D = Dual<double>;
  Variable<D, 'x'> xr{D{xv}};
  Variable<D, 'y'> yr{D{yv}};
  auto f_rev = sin(xr) * exp(yr);
  auto H_rev = reverse_mode_hess(f_rev, std::array{xv, yv});
  for (int i : {0, 1})
    for (int j : {0, 1})
      EXPECT_NEAR(H_fwd[i][j], H_rev[i][j], 1e-12);
}
