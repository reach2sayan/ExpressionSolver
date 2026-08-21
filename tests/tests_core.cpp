#include "tests_common.hpp"

TEST(MathFunctionTest, TanEvalAndDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(tan(x).eval(x0), std::tan(x0));
  ASSERT_DOUBLE_EQ(tan(x).derivative().eval(x0),
                   1.0 / (std::cos(x0) * std::cos(x0)));
}

TEST(MathFunctionTest, LogEvalAndDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(log(x).eval(x0), std::log(x0));
  ASSERT_DOUBLE_EQ(log(x).derivative().eval(x0), 1.0 / x0);
}

TEST(MathFunctionTest, SqrtEvalAndDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(sqrt(x).eval(x0), std::sqrt(x0));
  ASSERT_DOUBLE_EQ(sqrt(x).derivative().eval(x0), 0.5 / std::sqrt(x0));
}

TEST(MathFunctionTest, AsinDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(asin(x).eval(x0), std::asin(x0));
  ASSERT_DOUBLE_EQ(asin(x).derivative().eval(x0),
                   1.0 / std::sqrt(1.0 - x0 * x0));
}

TEST(MathFunctionTest, AcosDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(acos(x).eval(x0), std::acos(x0));
  ASSERT_DOUBLE_EQ(acos(x).derivative().eval(x0),
                   -1.0 / std::sqrt(1.0 - x0 * x0));
}

TEST(MathFunctionTest, AtanDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(atan(x).eval(x0), std::atan(x0));
  ASSERT_DOUBLE_EQ(atan(x).derivative().eval(x0), 1.0 / (1.0 + x0 * x0));
}

TEST(MathFunctionTest, SinhDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(sinh(x).eval(x0), std::sinh(x0));
  ASSERT_DOUBLE_EQ(sinh(x).derivative().eval(x0), std::cosh(x0));
}

TEST(MathFunctionTest, CoshDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(cosh(x).eval(x0), std::cosh(x0));
  ASSERT_DOUBLE_EQ(cosh(x).derivative().eval(x0), std::sinh(x0));
}

TEST(MathFunctionTest, TanhDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  ASSERT_DOUBLE_EQ(tanh(x).eval(x0), std::tanh(x0));
  double c = std::cosh(x0);
  ASSERT_DOUBLE_EQ(tanh(x).derivative().eval(x0), 1.0 / (c * c));
}

TEST(MathFunctionTest, AbsEval) {
  ASSERT_DOUBLE_EQ(abs(var<"x">).eval(3.0), 3.0);
  ASSERT_DOUBLE_EQ(abs(var<"x">).eval(-3.0), 3.0);
  ASSERT_DOUBLE_EQ(abs(var<"x">).eval(0.0), 0.0);
}

TEST(MathFunctionTest, ChainRuleTan) {
  double x0 = 0.4;
  auto x = var_of<"x">(x0);
  auto expr = tan(constant(2.0) * x);
  ASSERT_DOUBLE_EQ(expr.derivative().eval(x0),
                   2.0 / (std::cos(2.0 * x0) * std::cos(2.0 * x0)));
}

TEST(MathFunctionTest, ChainRuleLog) {
  double x0 = 0.8;
  auto x = var_of<"x">(x0);
  auto expr = log(x * x);
  ASSERT_DOUBLE_EQ(expr.derivative().eval(x0), 2.0 / x0);
}

TEST(MathFunctionTest, ChainRuleSqrt) {
  double x0 = 1.0;
  auto x = var_of<"x">(x0);
  auto expr = sqrt(sin(x));
  double expected = std::cos(x0) / (2.0 * std::sqrt(std::sin(x0)));
  ASSERT_NEAR(expr.derivative().eval(x0), expected, 1e-12);
}

TEST(MathFunctionTest, PythagoreanIdentity) {
  for (double v : {0.0, 0.5, 1.0, 2.0}) {
    auto x = var_of<"x">(v);
    auto expr = sin(x) * sin(x) + cos(x) * cos(x);
    ASSERT_NEAR(expr.eval(v), 1.0, 1e-12);
    ASSERT_NEAR(expr.derivative().eval(v), 0.0, 1e-12);
  }
}

TEST(MathFunctionTest, HyperbolicIdentity) {
  for (double v : {0.0, 0.5, 1.0, 2.0}) {
    auto x = var_of<"x">(v);
    auto expr = cosh(x) * cosh(x) - sinh(x) * sinh(x);
    ASSERT_NEAR(expr.eval(v), 1.0, 1e-12);
    ASSERT_NEAR(expr.derivative().eval(v), 0.0, 1e-12);
  }
}

TEST(MathFunctionTest, ExpLogIdentity) {
  for (double v : {0.3, 0.5, 1.0, 2.0}) {
    auto x = var_of<"x">(v);
    auto expr = exp(log(x));
    ASSERT_NEAR(expr.eval(v), v, 1e-12);
    ASSERT_NEAR(expr.derivative().eval(v), 1.0, 1e-12);
  }
}

TEST(MathFunctionTest, QuotientSelfIsConstant) {
  for (double v : {1.0, 2.0, 5.0}) {
    auto x = var_of<"x">(v);
    auto expr = x / x;
    ASSERT_NEAR(expr.eval(v), 1.0, 1e-12);
    ASSERT_NEAR(expr.derivative().eval(v), 0.0, 1e-12);
  }
}

TEST(MathFunctionTest, TanEqualsRatio) {
  double x0 = 0.7;
  auto x1 = var_of<"x">(x0);
  auto x2 = var_of<"x">(x0);
  ASSERT_NEAR(tan(x1).derivative().eval(x0),
              (sin(x2) / cos(x2)).derivative().eval(x0), 1e-12);
}

// ===========================================================================
// Reverse-mode gradients for new math functions
// ===========================================================================

TEST(ReverseModeAD, TanDerivative) {
  auto x = var<"x">;
  auto g = Equation{tan(x)}.gradient(0.5);
  ASSERT_DOUBLE_EQ(g[0], 1.0 / (std::cos(0.5) * std::cos(0.5)));
}

TEST(ReverseModeAD, LogDerivative) {
  auto x = var<"x">;
  auto g = Equation{log(x)}.gradient(0.5);
  ASSERT_DOUBLE_EQ(g[0], 2.0);
}

TEST(ReverseModeAD, SqrtDerivative) {
  auto x = var<"x">;
  auto g = Equation{sqrt(x)}.gradient(4.0);
  ASSERT_DOUBLE_EQ(g[0], 0.25); // 0.5/sqrt(4) = 0.25
}

TEST(ReverseModeAD, AsinDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  auto g = Equation{asin(x)}.gradient(x0);
  ASSERT_DOUBLE_EQ(g[0], 1.0 / std::sqrt(1.0 - x0 * x0));
}

TEST(ReverseModeAD, AcosDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  auto g = Equation{acos(x)}.gradient(x0);
  ASSERT_DOUBLE_EQ(g[0], -1.0 / std::sqrt(1.0 - x0 * x0));
}

TEST(ReverseModeAD, AtanDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  auto g = Equation{atan(x)}.gradient(x0);
  ASSERT_DOUBLE_EQ(g[0], 1.0 / (1.0 + x0 * x0));
}

TEST(ReverseModeAD, SinhDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  auto g = Equation{sinh(x)}.gradient(x0);
  ASSERT_DOUBLE_EQ(g[0], std::cosh(x0));
}

TEST(ReverseModeAD, CoshDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  auto g = Equation{cosh(x)}.gradient(x0);
  ASSERT_DOUBLE_EQ(g[0], std::sinh(x0));
}

TEST(ReverseModeAD, TanhDerivative) {
  double x0 = 0.5;
  auto x = var_of<"x">(x0);
  auto g = Equation{tanh(x)}.gradient(x0);
  double c = std::cosh(x0);
  ASSERT_DOUBLE_EQ(g[0], 1.0 / (c * c));
}

TEST(ReverseModeAD, AbsDerivativePositive) {
  auto x = var<"x">;
  ASSERT_DOUBLE_EQ(Equation{abs(x)}.gradient(1.0)[0], 1.0);
}

TEST(ReverseModeAD, AbsDerivativeNegative) {
  auto x = var<"x">;
  ASSERT_DOUBLE_EQ(Equation{abs(x)}.gradient(-1.0)[0], -1.0);
}

TEST(ReverseModeAD, AbsDerivativeAtZero) {
  auto x = var<"x">;
  ASSERT_DOUBLE_EQ(Equation{abs(x)}.gradient(0.0)[0], 0.0);
}

// ===========================================================================
// Forward-mode (Dual) for new math functions
// ===========================================================================

TEST(ForwardModeAD, TanDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = tan(x).eval(Dual<double>{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, std::tan(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / (std::cos(x0) * std::cos(x0)));
}

TEST(ForwardModeAD, LogDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = log(x).eval(Dual<double>{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, std::log(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / x0);
}

TEST(ForwardModeAD, SqrtDerivative) {
  double x0 = 4.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = sqrt(x).eval(Dual<double>{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, 2.0);
  ASSERT_DOUBLE_EQ(df, 0.25);
}

TEST(ForwardModeAD, AsinDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = asin(x).eval(Dual<double>{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, std::asin(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / std::sqrt(1.0 - x0 * x0));
}

TEST(ForwardModeAD, AcosDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = acos(x).eval(Dual<double>{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, std::acos(x0));
  ASSERT_DOUBLE_EQ(df, -1.0 / std::sqrt(1.0 - x0 * x0));
}

TEST(ForwardModeAD, AtanDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = atan(x).eval(Dual<double>{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, std::atan(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / (1.0 + x0 * x0));
}

TEST(ForwardModeAD, SinhDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = sinh(x).eval(Dual<double>{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, std::sinh(x0));
  ASSERT_DOUBLE_EQ(df, std::cosh(x0));
}

TEST(ForwardModeAD, CoshDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = cosh(x).eval(Dual<double>{x0, 1.0});
  ASSERT_DOUBLE_EQ(f, std::cosh(x0));
  ASSERT_DOUBLE_EQ(df, std::sinh(x0));
}

TEST(ForwardModeAD, TanhDerivative) {
  double x0 = 0.5;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = tanh(x).eval(Dual<double>{x0, 1.0});
  double c = std::cosh(x0);
  ASSERT_DOUBLE_EQ(f, std::tanh(x0));
  ASSERT_DOUBLE_EQ(df, 1.0 / (c * c));
}

TEST(ForwardModeAD, AbsDerivativePositive) {
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = abs(x).eval(Dual<double>{2.0, 1.0});
  ASSERT_DOUBLE_EQ(f, 2.0);
  ASSERT_DOUBLE_EQ(df, 1.0);
}

TEST(ForwardModeAD, AbsDerivativeNegative) {
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = abs(x).eval(Dual<double>{-2.0, 1.0});
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
  static_assert(CExpression<Variable<double, ddx::impl::FixedString{"x"}>>);
  using SumExpr =
      decltype(std::declval<Variable<double, ddx::impl::FixedString{"x"}>>() +
               std::declval<Constant<double>>());
  static_assert(CExpression<SumExpr>);
  static_assert(!CExpression<int>);
  static_assert(!CExpression<double>);
}

TEST(ConceptTest, AnOpSatisfied) {
  static_assert(COperation<SumOp<double>>);
  static_assert(COperation<MultiplyOp<float>>);
  static_assert(COperation<SineOp<double>>);
  static_assert(COperation<CosineOp<double>>);
  static_assert(COperation<ExpOp<double>>);
  static_assert(COperation<NegateOp<int>>);
  static_assert(COperation<DivideOp<double>>);
}

// ===========================================================================
// Symbol extraction
// ===========================================================================

TEST(SymbolTest, SingleVariable) {
  using E = Variable<double, ddx::impl::FixedString{"x"}>;
  using Syms = extract_symbols_from_expr_t<E>;
  static_assert(ddx::impl::mp::mp_size<Syms>::value == 1);
  static_assert(
      std::is_same_v<ddx::impl::mp::mp_at_c<Syms, 0>,
                     ddx::impl::symbol_type<ddx::impl::FixedString{"x"}>>);
}

TEST(SymbolTest, TwoVariables) {
  using E =
      decltype(std::declval<Variable<double, ddx::impl::FixedString{"x"}>>() *
               std::declval<Variable<double, ddx::impl::FixedString{"y"}>>());
  using Syms = extract_symbols_from_expr_t<E>;
  static_assert(ddx::impl::mp::mp_size<Syms>::value == 2);
  // Symbols are sorted lexicographically: "x" < "y"
  static_assert(
      std::is_same_v<ddx::impl::mp::mp_at_c<Syms, 0>,
                     ddx::impl::symbol_type<ddx::impl::FixedString{"x"}>>);
  static_assert(
      std::is_same_v<ddx::impl::mp::mp_at_c<Syms, 1>,
                     ddx::impl::symbol_type<ddx::impl::FixedString{"y"}>>);
}

TEST(SymbolTest, DuplicateSymbolsDeduplicated) {
  using E =
      decltype(std::declval<Variable<double, ddx::impl::FixedString{"x"}>>() *
               std::declval<Variable<double, ddx::impl::FixedString{"x"}>>());
  using Syms = extract_symbols_from_expr_t<E>;
  static_assert(ddx::impl::mp::mp_size<Syms>::value == 1);
}

TEST(SymbolTest, ThreeVariables) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto z = var<"z">;
  auto expr = x + y + z;
  using Syms = extract_symbols_from_expr_t<decltype(expr)>;
  static_assert(ddx::impl::mp::mp_size<Syms>::value == 3);
}

// The canonical order is what turns a label into a slot index, so an expression
// whose symbols appear out of order, more than once, and across several
// subtrees has to come back alphabetical and deduplicated -- every gradient is
// returned in this order, and a sort that reordered them would silently return
// the right numbers against the wrong names.
TEST(SymbolTest, CanonicalOrderIsAlphabetical) {
  auto expr = (var<"z"> * var<"a">)+(var<"m"> * var<"z">)+var<"b">;
  using Syms = extract_symbols_from_expr_t<decltype(expr)>;
  using ddx::impl::FixedString;
  static_assert(ddx::impl::mp::mp_size<Syms>::value == 4);
  static_assert(
      std::is_same_v<Syms, ddx::impl::mp::mp_list<
                               ddx::impl::symbol_type<FixedString{"a"}>,
                               ddx::impl::symbol_type<FixedString{"b"}>,
                               ddx::impl::symbol_type<FixedString{"m"}>,
                               ddx::impl::symbol_type<FixedString{"z"}>>>,
      "canonical symbol order must stay alphabetical and deduplicated");
}

// ===========================================================================
// Expression / operator tests
// ===========================================================================

TEST(ExpressionTest, StaticTests) {
  static_assert(std::is_same_v<
                as_const_expression<Expression<
                    MultiplyOp<int>, Variable<int, ddx::impl::FixedString{"x"}>,
                    Constant<int>>>,
                Expression<MultiplyOp<int>,
                           Variable<int, ddx::impl::FixedString{"x"}, true>,
                           Constant<int>>>);

  static_assert(std::is_same_v<
                as_const_expression<Expression<
                    MultiplyOp<int>, Variable<int, ddx::impl::FixedString{"x"}>,
                    Variable<int, ddx::impl::FixedString{"y"}>>>,
                Expression<MultiplyOp<int>,
                           Variable<int, ddx::impl::FixedString{"x"}, true>,
                           Variable<int, ddx::impl::FixedString{"y"}, true>>>);

  auto x = 4_vi;
  auto y = 2_vi;
  auto c = 2_ci;
  auto res = x * y + c;
  auto res2 = make_const_variable<ddx::impl::FixedString{"c"}>(res);
  // Freezing changes only the derivative, so the value is unchanged.
  ASSERT_EQ(res2.eval(4), res.eval(4));
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
  ASSERT_EQ(sum_exp.eval(2), 6);
  ASSERT_EQ(d.eval(), 3); // 1*b*3 folds to the constant 3: no symbols left
}

TEST(ExpressionTest, SubtractTest) {
  auto a = 1_ci;
  auto b = 2_vi;
  auto minus = a - b;
  auto d = minus.derivative();
  ASSERT_EQ(minus.eval(2), -1);
  ASSERT_EQ(d.eval(), -1); // derivative of a constant-only tree
}

TEST(ExpressionTest, DivideTest) {
  auto a = 4.0_vd;
  auto b = 2.0_cd;
  auto divide = a / b;
  auto d = divide.derivative();
  ASSERT_EQ(divide.eval(4.0), 2.0);
  ASSERT_EQ(d.eval(), 0.5); // d(a/2)/da folds to the constant 0.5
}

TEST(ExpressionTest, ExpTest) {
  constexpr auto exp_exp = exp(2.0_cd);
  ASSERT_EQ(exp_exp.eval(), std::exp(2.0));
}

TEST(ExpressionTest, ExpSum) {
  constexpr auto target = exp(1_cd + 2_cd);
  ASSERT_EQ(target.eval(), std::exp(3.0));
}

TEST(ExpressionTest, ExpDerivative) {
  for (std::size_t i = 1; i < 1000; ++i) {
    Variable<double, ddx::impl::FixedString{"x"}> xv;
    auto target = exp(xv);
    ASSERT_EQ(target.derivative().eval(i * 1.0), target.eval(i * 1.0));
  }
}

TEST(ExpressionTest, Combination) {
  double target = (exp(1_cd + 2_cd + 3_cd) + 1_cd).eval();
  ASSERT_EQ(target, std::exp(6.0) + 1.0);
}

TEST(ExpressionTest, ConstantDerivative) {
  auto target = Constant<int>(1);
  ASSERT_EQ(target.derivative(), 0);
  ASSERT_EQ(target, 1);
}

TEST(ExpressionTest, VariableDerivative) {
  Variable<int, ddx::impl::FixedString{"x"}> x;
  ASSERT_EQ(x.derivative().eval(), 1);
  ASSERT_EQ(x.eval(5), 5);
}

TEST(ExpressionTest, DerivativeTest) {
  auto x = 4_vi;
  auto expr = x * 2_ci;
  auto derv = expr.derivative();
  ASSERT_EQ(expr.eval(4), 8);
  ASSERT_EQ(derv.eval(), 2); // d(x*2)/dx folds to the constant 2
}

// ===========================================================================
// Algebraic derivative rules
// ===========================================================================

TEST(DerivativeRuleTest, SumRule) {
  auto x = var<"x", int>;
  auto expr = 3_ci * x + 5_ci;
  ASSERT_EQ(expr.derivative().eval(), 3); // folds to the constant 3
}

TEST(DerivativeRuleTest, ProductRule) {
  Variable<int, ddx::impl::FixedString{"x"}> x;
  auto expr = x * x;
  ASSERT_EQ(expr.derivative().eval(4), 8);
}

TEST(DerivativeRuleTest, QuotientRule) {
  Variable<double, ddx::impl::FixedString{"x"}> x;
  auto c = Constant<double>{3.0};
  auto expr = x / c;
  // 1/c is constant in x, so the folded derivative names no symbols.
  ASSERT_DOUBLE_EQ(expr.derivative().eval(), 1.0 / 3.0);
}

TEST(DerivativeRuleTest, ChainRule_ExpOfLinear) {
  Variable<double, ddx::impl::FixedString{"x"}> x;
  auto inner = constant(2.0) * x;
  auto expr = exp(inner);
  ASSERT_DOUBLE_EQ(expr.derivative().eval(1.0), 2.0 * std::exp(2.0));
}

TEST(DerivativeRuleTest, ChainRule_SinOfLinear) {
  Variable<double, ddx::impl::FixedString{"x"}> x;
  auto inner = constant(3.0) * x;
  auto expr = sin(inner);
  ASSERT_DOUBLE_EQ(expr.derivative().eval(0.0), 3.0 * std::cos(0.0));
}

// ===========================================================================
// Variable tests
// ===========================================================================

TEST(VariableTest, GetValue) {
  auto a = 2_vi;
  ASSERT_EQ(a.eval(2), 2);
}

TEST(VariableTest, Assign) {
  // A variable holds no value, so "assignment" is supplying a point.
  Variable<int, FixedString{"a"}> a;
  ASSERT_EQ(a.eval(2), 2);
  ASSERT_EQ(a.eval(5), 5);
}

TEST(VariableTest, UdlCompAndAssign) {
  Variable<int, FixedString{"a"}> a;
  auto b = 4_vi;
  ASSERT_EQ(a.eval(4), b.eval(4));
}

// ===========================================================================
// Trig tests
// ===========================================================================

TEST(TrigTest, SinTest) {
  auto b = sin(constant(0.5));
  ASSERT_EQ(b.eval(), std::sin(0.5)); // no symbols
}

TEST(TrigTest, CosTest) {
  auto b = cos(var<"x">);
  ASSERT_DOUBLE_EQ(b.eval(0.45), std::cos(0.45));
  ASSERT_DOUBLE_EQ(b.derivative().eval(0.45), -std::sin(0.45));
}

TEST(TrigTest, SinDerivative) {
  auto x = var<"x">;
  auto s = sin(x);
  ASSERT_DOUBLE_EQ(s.derivative().eval(0.7), std::cos(0.7));
}

TEST(TrigTest, SinCosIdentity) {
  for (double v : {0.0, 0.5, 1.0, std::numbers::pi / 4}) {
    Variable<double, ddx::impl::FixedString{"x"}> x;
    auto s = sin(x);
    auto c = cos(x);
    double lhs = (s * s).eval(v) + (c * c).eval(v);
    ASSERT_NEAR(lhs, 1.0, 1e-12);
  }
}

TEST(TrigTest, CosDerivativeIsNegSin) {
  for (double v : {0.0, 0.3, 1.0, 2.0}) {
    Variable<double, ddx::impl::FixedString{"x"}> x;
    ASSERT_DOUBLE_EQ(cos(x).derivative().eval(v), -std::sin(v));
  }
}

TEST(TrigTest, ExpDerivativeIsItself) {
  for (double v : {-1.0, 0.0, 0.5, 1.5}) {
    Variable<double, ddx::impl::FixedString{"x"}> x;
    ASSERT_DOUBLE_EQ(exp(x).derivative().eval(v), std::exp(v));
  }
}

// ===========================================================================
// Equation (partial differentiation) tests
// ===========================================================================

TEST(EquationTest, SingleVariable) {
  auto x = var<"x", int>;
  auto expr = x * 2_ci;
  auto eq = Equation(expr);
  ASSERT_EQ(eq[idx<1>()].eval(), 2); // d(2x)/dx folds to the constant 2
}

TEST(EquationTest, TwoVariables) {
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  auto expr = x * y;
  auto eq = Equation(expr);
  // The folded partials name one symbol each, so each takes just that value.
  ASSERT_EQ(eq[idx<1>()].eval(2), 2); // df/dx = y = 2
  ASSERT_EQ(eq[idx<2>()].eval(4), 4); // df/dy = x = 4
}

TEST(EquationTest, LinearCombination) {
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  auto expr = constant(1) * x + constant(2) * y;
  auto eq = Equation(expr);
  // Both partials fold to constants: no symbols left to supply.
  ASSERT_EQ(eq[idx<1>()].eval(), 1);
  ASSERT_EQ(eq[idx<2>()].eval(), 2);
}

TEST(EquationTest, DifferenceOfSquares) {
  constexpr auto x = var<"x", int>;
  constexpr auto y = var<"y", int>;
  constexpr auto expr = (x + y) * (x - y);
  auto eq = Equation(expr);
  auto [d1, d2] =
      eq.template gradient<ddx::DiffMode::Symbolic>(std::array{4, 2});
  ASSERT_EQ(expr.eval(4, 2), 12); // (4+2)*(4-2) = 12
  ASSERT_EQ(d1, 8);               // 2x = 8
  ASSERT_EQ(d2, -4);              // -2y = -4
}

TEST(EquationTest, ExpEquation) {
  auto x = var<"x">;
  auto expr = exp(x);
  auto eq = Equation(expr);
  ASSERT_DOUBLE_EQ(eq.evaluate(std::array{1.0}), std::exp(1.0));
  ASSERT_DOUBLE_EQ(eq[idx<1>()].eval(1.0), std::exp(1.0)); // d(e^x)/dx = e^x
}

TEST(EquationTest, TrigEquation) {
  auto x = var<"x">;
  auto expr = sin(x) * cos(x);
  auto eq = Equation(expr);
  ASSERT_DOUBLE_EQ(eq[idx<1>()].eval(0.5), std::cos(2 * 0.5));
}

TEST(EquationTest, IdxEquivalence) {
  // eq[idx<N>()] is the subscript spelling of eq.get<N>(); both must select the
  // same slot.
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  auto eq = Equation(x * y);
  ASSERT_EQ(eq[idx<1>()].eval(5), eq.get<1>().eval(5)); // df/dx = y
  ASSERT_EQ(eq[idx<2>()].eval(3), eq.get<2>().eval(3)); // df/dy = x
}

TEST(EquationTest, ThreeVariablePartials) {
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  auto z = var<"z", int>;
  auto expr = x * y + y * z;
  auto eq = Equation(expr);
  auto [dx, dy, dz] =
      eq.template gradient<ddx::DiffMode::Symbolic>(std::array{2, 3, 4});
  ASSERT_EQ(dx, 3);
  ASSERT_EQ(dy, 6);
  ASSERT_EQ(dz, 3);
}

TEST(EquationTest, UpdateAndReevaluate) {
  // Both copies of x in the derivative are live variables, so update
  // propagates.
  Variable<int, ddx::impl::FixedString{"x"}> x;
  auto expr = x * x;
  auto eq = Equation(expr);

  ASSERT_EQ(eq.evaluate(std::array{3}), 9);
  ASSERT_EQ(eq[idx<1>()].eval(3), 6); // 2*3 = 6

  ASSERT_EQ(eq.evaluate(std::array{5}), 25);
  ASSERT_EQ(eq[idx<1>()].eval(5), 10); // 2*5 = 10
}

TEST(EquationTest, MixedTrigExpEquation) {
  auto x = var<"x">;
  auto expr = exp(x) * sin(x);
  auto eq = Equation(expr);
  double expected = std::exp(1.0) * (std::sin(1.0) + std::cos(1.0));
  ASSERT_DOUBLE_EQ(eq[idx<1>()].eval(1.0), expected);
}

TEST(EquationTest, NumberOfDerivatives) {
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  static_assert(Equation<decltype(x * y)>::number_of_derivatives == 2);
  static_assert(Equation<decltype(x * x)>::number_of_derivatives == 1);
}

// ===========================================================================
// Equation — f: ℝⁿ → ℝᵐ  (Jacobian tests)
// ===========================================================================

TEST(EquationTest, Dimensions) {
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  using VE = Equation<decltype(x + y), decltype(x * y)>;
  static_assert(VE::output_dim == 2);
  static_assert(VE::input_dim == 2);
}

TEST(EquationTest, Eval) {
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  auto ve = Equation(x + y, x * y);
  auto v = ve.evaluate(std::array{3, 4});
  ASSERT_EQ(v[0], 7);
  ASSERT_EQ(v[1], 12);
}

TEST(EquationTest, JacobianLinear) {
  auto x = var<"x", int>;
  auto y = var<"y", int>;
  auto ve = Equation(x + y, x * y);
  auto J = ve.template jacobian<ddx::DiffMode::Symbolic>(std::array{3, 4});
  ASSERT_EQ(J[0][0], 1); // ∂(x+y)/∂x
  ASSERT_EQ(J[0][1], 1); // ∂(x+y)/∂y
  ASSERT_EQ(J[1][0], 4); // ∂(x*y)/∂x = y = 4
  ASSERT_EQ(J[1][1], 3); // ∂(x*y)/∂y = x = 3
}

TEST(EquationTest, JacobianWithTrig) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto ve = Equation(x * y, sin(x) + y * y);
  auto J = ve.template jacobian<ddx::DiffMode::Symbolic>(std::array{2.0, 3.0});
  ASSERT_DOUBLE_EQ(J[0][0], 3.0);           // ∂(x*y)/∂x = y
  ASSERT_DOUBLE_EQ(J[0][1], 2.0);           // ∂(x*y)/∂y = x
  ASSERT_DOUBLE_EQ(J[1][0], std::cos(2.0)); // ∂(sin(x)+y²)/∂x
  ASSERT_DOUBLE_EQ(J[1][1], 6.0);           // ∂(sin(x)+y²)/∂y = 2y
}

TEST(EquationTest, SingleComponentIsGradient) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto eq = Equation(x * y);
  auto g = eq.template gradient<ddx::DiffMode::Symbolic>(std::array{2.0, 3.0});
  ASSERT_DOUBLE_EQ(g[0], 3.0); // ∂(x*y)/∂x = y
  ASSERT_DOUBLE_EQ(g[1], 2.0); // ∂(x*y)/∂y = x
}

TEST(EquationTest, SymbolUnionAcrossComponents) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto ve = Equation(x * x, y * y); // (x², y²)
  static_assert(decltype(ve)::input_dim == 2);
  auto J = ve.template jacobian<ddx::DiffMode::Symbolic>(std::array{4.0, 3.0});
  ASSERT_DOUBLE_EQ(J[0][0], 8.0); // ∂(x²)/∂x = 2x = 8
  ASSERT_DOUBLE_EQ(J[0][1], 0.0); // ∂(x²)/∂y = 0
  ASSERT_DOUBLE_EQ(J[1][0], 0.0); // ∂(y²)/∂x = 0
  ASSERT_DOUBLE_EQ(J[1][1], 6.0); // ∂(y²)/∂y = 2y = 6
}

TEST(EquationTest, ThreeOutputs) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto ve = Equation(x * x, x * y, y * y);
  static_assert(decltype(ve)::output_dim == 3);
  static_assert(decltype(ve)::input_dim == 2);
  auto J = ve.template jacobian<ddx::DiffMode::Symbolic>(std::array{2.0, 5.0});
  ASSERT_DOUBLE_EQ(J[0][0], 4.0);  // 2x
  ASSERT_DOUBLE_EQ(J[0][1], 0.0);  // 0
  ASSERT_DOUBLE_EQ(J[1][0], 5.0);  // y
  ASSERT_DOUBLE_EQ(J[1][1], 2.0);  // x
  ASSERT_DOUBLE_EQ(J[2][0], 0.0);  // 0
  ASSERT_DOUBLE_EQ(J[2][1], 10.0); // 2y
}

TEST(EquationTest, ReverseJacobianAgreesWithSymbolic) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto z = var<"z">;
  auto ve = Equation(x * y, sin(x) + y * z, exp(z));

  auto J_sym =
      ve.template jacobian<ddx::DiffMode::Symbolic>(std::array{2.0, 3.0, 4.0});
  auto J_rev = ve.jacobian(std::array{2.0, 3.0, 4.0});

  for (std::size_t i = 0; i < decltype(ve)::output_dim; ++i)
    for (std::size_t j = 0; j < decltype(ve)::input_dim; ++j)
      ASSERT_DOUBLE_EQ(J_rev[i][j], J_sym[i][j]);
}

TEST(EquationTest, ParallelReverseJacobian_FourOutputs) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto z = var<"z">;
  auto ve = Equation(x * y, y * z, x * z, x * y * z);
  static_assert(decltype(ve)::output_dim == 4);
  static_assert(decltype(ve)::input_dim == 3);

  auto J_sym =
      ve.template jacobian<ddx::DiffMode::Symbolic>(std::array{1.0, 2.0, 3.0});
  auto J_rev = ve.jacobian(std::array{1.0, 2.0, 3.0});

  for (std::size_t i = 0; i < decltype(ve)::output_dim; ++i)
    for (std::size_t j = 0; j < decltype(ve)::input_dim; ++j)
      ASSERT_DOUBLE_EQ(J_rev[i][j], J_sym[i][j]);
}

TEST(EquationTest, ParallelReverseJacobian_FiveOutputsTrigExp) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto z = var<"z">;
  auto ve = Equation(sin(x) * cos(y), exp(x + y), x * y + y * z,
                     cos(z) * sin(x), exp(x * z) + y * y);
  static_assert(decltype(ve)::output_dim == 5);
  static_assert(decltype(ve)::input_dim == 3);

  auto J_sym =
      ve.template jacobian<ddx::DiffMode::Symbolic>(std::array{0.5, 1.0, 1.5});
  auto J_rev = ve.jacobian(std::array{0.5, 1.0, 1.5});

  for (std::size_t i = 0; i < decltype(ve)::output_dim; ++i)
    for (std::size_t j = 0; j < decltype(ve)::input_dim; ++j)
      ASSERT_NEAR(J_rev[i][j], J_sym[i][j], 1e-12);
}

TEST(EquationTest, ReverseJacobianSingleOutputMatchesGradient) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto expr = exp(x) * sin(y);
  // Use a 2-component Equation so the vector specialization is selected.
  auto ve = Equation(expr, exp(x) * sin(y));

  auto J_rev = ve.jacobian(std::array{2.0, 5.0});
  auto g = Equation{expr}.gradient(2.0, 5.0);

  for (std::size_t j = 0; j < decltype(ve)::input_dim; ++j)
    ASSERT_DOUBLE_EQ(J_rev[0][j], g[j]);
}

// ===========================================================================
// Forward-mode automatic differentiation via dual numbers
// ===========================================================================

TEST(ForwardModeAD, ExpressionStructuredBinding) {
  // A bare expression carries no point, so value and derivative are taken
  // explicitly rather than by structured binding.
  auto x = var<"x">;
  const auto e = x * x;
  EXPECT_DOUBLE_EQ(e.eval(3.0), 9.0);
  EXPECT_DOUBLE_EQ(e.derivative().eval(3.0), 6.0);

  const auto s = sin(var<"x">);
  EXPECT_DOUBLE_EQ(s.eval(0.0), 0.0);
  EXPECT_DOUBLE_EQ(s.derivative().eval(0.0), 1.0);
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

TEST(ForwardModeAD, DualScalarArithmetic) {
  // A bare scalar promotes to a zero-derivative Dual, so the derivative part
  // of `d` is preserved (or scaled, for * and /) and the value part shifts.
  constexpr Dual<double> d{3.0, 1.0};

  auto [add_v, add_d] = d + 2.0;
  EXPECT_DOUBLE_EQ(add_v, 5.0);
  EXPECT_DOUBLE_EQ(add_d, 1.0);
  auto [radd_v, radd_d] = 2.0 + d;
  EXPECT_DOUBLE_EQ(radd_v, 5.0);
  EXPECT_DOUBLE_EQ(radd_d, 1.0);

  auto [sub_v, sub_d] = d - 2.0;
  EXPECT_DOUBLE_EQ(sub_v, 1.0);
  EXPECT_DOUBLE_EQ(sub_d, 1.0);
  auto [rsub_v, rsub_d] = 2.0 - d;
  EXPECT_DOUBLE_EQ(rsub_v, -1.0);
  EXPECT_DOUBLE_EQ(rsub_d, -1.0);

  auto [mul_v, mul_d] = 2.0 * d;
  EXPECT_DOUBLE_EQ(mul_v, 6.0);
  EXPECT_DOUBLE_EQ(mul_d, 2.0); // scalar scales the derivative

  auto [div_v, div_d] = d / 2.0;
  EXPECT_DOUBLE_EQ(div_v, 1.5);
  EXPECT_DOUBLE_EQ(div_d, 0.5);

  Dual<double> acc{3.0, 1.0};
  acc += 2.0;
  EXPECT_DOUBLE_EQ(acc.template get<0>(), 5.0);
  EXPECT_DOUBLE_EQ(acc.template get<1>(), 1.0);
}

TEST(ForwardModeAD, ScalarPromotionDeepDual) {
  // `expr + scalar` must embed a zero derivative at every nesting level: a
  // single static_cast would need two chained explicit conversions here.
  using DD = Dual<Dual<double>>;
  Variable<DD, ddx::impl::FixedString{"x"}> x;
  auto expr = x + 2.0;
  DD result = expr.eval(DD{Dual<double>{2.0, 0.0}, Dual<double>{0.0, 0.0}});
  EXPECT_DOUBLE_EQ(get_real_part<2>(result), 4.0); // peel both Dual<> layers

  auto y = var<"y">;
  EXPECT_DOUBLE_EQ((y + 2.0).eval(3.0), 5.0);
}

TEST(ForwardModeAD, PolynomialDerivative) {
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = (x * x + x).eval(Dual<double>{3.0, 1.0});
  EXPECT_DOUBLE_EQ(f, 12.0);
  EXPECT_DOUBLE_EQ(df, 7.0);
}

TEST(ForwardModeAD, PartialDerivativeX) {
  Variable<Dual<double>, FixedString{"x"}> x;
  Variable<Dual<double>, FixedString{"y"}> y;
  auto [f, df] = (x * y).eval(Dual<double>{3.0, 1.0}, Dual<double>{4.0, 0.0});
  EXPECT_DOUBLE_EQ(f, 12.0);
  EXPECT_DOUBLE_EQ(df, 4.0);
}

TEST(ForwardModeAD, PartialDerivativeY) {
  Variable<Dual<double>, FixedString{"x"}> x;
  Variable<Dual<double>, FixedString{"y"}> y;
  auto [f, df] = (x * y).eval(Dual<double>{3.0, 0.0}, Dual<double>{4.0, 1.0});
  EXPECT_DOUBLE_EQ(f, 12.0);
  EXPECT_DOUBLE_EQ(df, 3.0);
}

TEST(ForwardModeAD, SinDerivative) {
  double x0 = std::numbers::pi / 4.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = sin(x).eval(Dual<double>{x0, 1.0});
  EXPECT_DOUBLE_EQ(f, std::sin(x0));
  EXPECT_DOUBLE_EQ(df, std::cos(x0));
}

TEST(ForwardModeAD, CosDerivative) {
  double x0 = std::numbers::pi / 3.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = cos(x).eval(Dual<double>{x0, 1.0});
  EXPECT_DOUBLE_EQ(f, std::cos(x0));
  EXPECT_DOUBLE_EQ(df, -std::sin(x0));
}

TEST(ForwardModeAD, ExpDerivative) {
  double x0 = 2.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = exp(x).eval(Dual<double>{x0, 1.0});
  EXPECT_DOUBLE_EQ(f, std::exp(x0));
  EXPECT_DOUBLE_EQ(df, std::exp(x0));
}

TEST(ForwardModeAD, ChainRule) {
  double x0 = 1.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto [f, df] = sin(x * x).eval(Dual<double>{x0, 1.0});
  EXPECT_DOUBLE_EQ(f, std::sin(x0 * x0));
  EXPECT_DOUBLE_EQ(df, 2.0 * x0 * std::cos(x0 * x0));
}

TEST(ForwardModeAD, Equivalence) {
  double x0 = 1.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto xv = var_of<"x">(x0);
  auto l = sin(xv * xv);
  auto [f, df] = sin(x * x).eval(Dual<double>{x0, 1.0});
  auto f2 = l.eval(x0);
  auto df2 = l.derivative().eval(x0);
  EXPECT_DOUBLE_EQ(f, std::sin(x0 * x0));
  EXPECT_DOUBLE_EQ(df, 2.0 * x0 * std::cos(x0 * x0));
  EXPECT_DOUBLE_EQ(df, df2);
  EXPECT_DOUBLE_EQ(f, f2);
}

// ===========================================================================
// Reverse-mode automatic differentiation via backward() /
// Equation{}.gradient()
// ===========================================================================

TEST(ReverseModeAD, SingleVariableLinear) {
  auto x = var<"x">;
  auto expr = constant(3.0) * x;
  auto g = Equation{expr}.gradient(5.0);
  EXPECT_DOUBLE_EQ(g[0], 3.0);
}

TEST(ReverseModeAD, ProductRule) {
  Variable<double, ddx::impl::FixedString{"x"}> x;
  auto expr = x * x;
  auto g = Equation{expr}.gradient(std::array{4.0});
  EXPECT_DOUBLE_EQ(g[0], 8.0);
}

TEST(ReverseModeAD, TwoVariables) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto expr = x * y;
  auto g = Equation{expr}.gradient(3.0, 4.0);
  static_assert(g.size() == 2);
  EXPECT_DOUBLE_EQ(g[0], 4.0); // df/dx = y
  EXPECT_DOUBLE_EQ(g[1], 3.0); // df/dy = x
}

TEST(ReverseModeAD, Sum) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto g = Equation{x + y}.gradient(2.0, 5.0);
  EXPECT_DOUBLE_EQ(g[0], 1.0);
  EXPECT_DOUBLE_EQ(g[1], 1.0);
}

TEST(ReverseModeAD, LinearCombination) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto g = Equation{constant(2.0) * x + constant(3.0) * y}.gradient(1.0, 1.0);
  EXPECT_DOUBLE_EQ(g[0], 2.0);
  EXPECT_DOUBLE_EQ(g[1], 3.0);
}

TEST(ReverseModeAD, Divide) {
  auto x = var<"x">;
  auto c = constant(3.0);
  auto g = Equation{x / c}.gradient(6.0);
  EXPECT_DOUBLE_EQ(g[0], 1.0 / 3.0);
}

TEST(ReverseModeAD, NegateViaSubtract) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto g = Equation{x - y}.gradient(5.0, 2.0);
  EXPECT_DOUBLE_EQ(g[0], 1.0);
  EXPECT_DOUBLE_EQ(g[1], -1.0);
}

TEST(ReverseModeAD, SinDerivative) {
  auto x = var<"x">;
  auto g = Equation{sin(x)}.gradient(1.0);
  EXPECT_DOUBLE_EQ(g[0], std::cos(1.0));
}

TEST(ReverseModeAD, CosDerivative) {
  auto x = var<"x">;
  auto g = Equation{cos(x)}.gradient(1.0);
  EXPECT_DOUBLE_EQ(g[0], -std::sin(1.0));
}

TEST(ReverseModeAD, ExpDerivative) {
  auto x = var<"x">;
  auto g = Equation{exp(x)}.gradient(2.0);
  EXPECT_DOUBLE_EQ(g[0], std::exp(2.0));
}

TEST(ReverseModeAD, ChainRuleSinOfProduct) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto g = Equation{sin(x * y)}.gradient(2.0, 3.0);
  EXPECT_DOUBLE_EQ(g[0], std::cos(6.0) * 3.0);
  EXPECT_DOUBLE_EQ(g[1], std::cos(6.0) * 2.0);
}

TEST(ReverseModeAD, ThreeVariables) {
  auto x = var<"x">;
  auto y = var<"y">;
  auto z = var<"z">;
  auto g = Equation{x * y + y * z}.gradient(2.0, 3.0, 4.0);
  EXPECT_DOUBLE_EQ(g[0], 3.0);
  EXPECT_DOUBLE_EQ(g[1], 6.0);
  EXPECT_DOUBLE_EQ(g[2], 3.0);
}

// ===========================================================================
// Equation — forward-mode Jacobian via derivative_tensor<1>
// ===========================================================================

TEST(EquationForward, TwoVariables) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(x * y, x + y);
  auto J = ve.derivative_tensor<1>(std::array{3.0, 4.0});
  EXPECT_DOUBLE_EQ(J[0][0], 4.0); // ∂(x*y)/∂x = y = 4
  EXPECT_DOUBLE_EQ(J[0][1], 3.0); // ∂(x*y)/∂y = x = 3
  EXPECT_DOUBLE_EQ(J[1][0], 1.0); // ∂(x+y)/∂x
  EXPECT_DOUBLE_EQ(J[1][1], 1.0); // ∂(x+y)/∂y
}

TEST(EquationForward, AgreesWithSymbolic) {
  double xv = 2.0, yv = 3.0;

  auto xs = var_of<"x">(xv);
  auto ys = var_of<"y">(yv);
  auto ve_sym = Equation(xs * xs, xs * ys, ys * ys);
  auto J_sym =
      ve_sym.template jacobian<ddx::DiffMode::Symbolic>(std::array{xv, yv});

  Variable<double, FixedString{"x"}> xd;
  Variable<double, FixedString{"y"}> yd;
  auto ve_fwd = Equation(xd * xd, xd * yd, yd * yd);
  auto J_fwd = ve_fwd.derivative_tensor<1>(std::array{xv, yv});

  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_DOUBLE_EQ(J_fwd[i][j], J_sym[i][j]);
}

TEST(EquationForward, TrigJacobian) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(x * y, sin(x) + y * y);
  auto J = ve.derivative_tensor<1>(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(J[0][0], 3.0);
  EXPECT_DOUBLE_EQ(J[0][1], 2.0);
  EXPECT_NEAR(J[1][0], std::cos(2.0), 1e-12);
  EXPECT_DOUBLE_EQ(J[1][1], 6.0);
}

TEST(EquationForward, AgreesWithReverse) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve_fwd = Equation(x * y, sin(x) + y * y);
  auto J_fwd = ve_fwd.derivative_tensor<1>(std::array{2.0, 3.0});

  auto xs = var<"x">;
  auto ys = var<"y">;
  auto ve_rev = Equation(xs * ys, sin(xs) + ys * ys);
  auto J_rev = ve_rev.jacobian(std::array{2.0, 3.0});

  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_NEAR(J_rev[i][j], J_fwd[i][j], 1e-12);
}

TEST(ReverseModeAD, ScalarLiteralCoercion) {
  auto x = var<"x", dual>;
  auto y = var<"y", dual>;
  auto z = var<"z", dual>;
  auto expe = 3.0 * x * y + y * z;
  auto g = Equation{expe}.gradient(
      Dual<double>{2.0, 0.0}, Dual<double>{3.0, 0.0}, Dual<double>{4.0, 0.0});
  EXPECT_DOUBLE_EQ(g[0], 9.0);  // df/dx = 3*y = 9
  EXPECT_DOUBLE_EQ(g[1], 10.0); // df/dy = 3*x + z = 10
  EXPECT_DOUBLE_EQ(g[2], 3.0);  // df/dz = y = 3
}

TEST(ReverseModeAD, ScalarOnRight) {
  auto x = var<"x">;
  auto g = Equation{x * 4.0 + 1.0}.gradient(5.0);
  EXPECT_DOUBLE_EQ(g[0], 4.0); // df/dx = 4
}

TEST(ReverseModeAD, AgreesWithForwardMode) {
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto g = Equation{exp(x) * sin(y)}.gradient(xv, yv);
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
  Dual<double> a{3.0, 1.0}, b{2.0, 0.5};
  a *= b;
  EXPECT_DOUBLE_EQ(a.template get<0>(), 6.0);
  EXPECT_DOUBLE_EQ(a.template get<1>(), 3.5);
}

TEST(DualCompoundAssign, DivEq) {
  Dual<double> a{4.0, 2.0}, b{2.0, 0.0};
  a /= b;
  EXPECT_DOUBLE_EQ(a.template get<0>(), 2.0);
  EXPECT_DOUBLE_EQ(a.template get<1>(), 1.0);
}

// ===========================================================================
// reverse_mode_gradient on Dual-valued (PDV) expressions
// ===========================================================================

TEST(ReverseModeAD_Dual, SingleVariable) {
  auto x = var<"x", dual>;
  auto g = Equation{3.0 * x}.gradient(Dual<double>{5.0, 0.0});
  EXPECT_DOUBLE_EQ(g[0], 3.0);
}

TEST(ReverseModeAD_Dual, TwoVariables) {
  auto x = var<"x", dual>;
  auto y = var<"y", dual>;
  auto g =
      Equation{x * y}.gradient(Dual<double>{3.0, 0.0}, Dual<double>{4.0, 0.0});
  EXPECT_DOUBLE_EQ(g[0], 4.0);
  EXPECT_DOUBLE_EQ(g[1], 3.0);
}

TEST(ReverseModeAD_Dual, ThreeVariables) {
  auto x = var<"x", dual>;
  auto y = var<"y", dual>;
  auto z = var<"z", dual>;
  auto g = Equation{3.0 * x * y + y * z}.gradient(
      Dual<double>{2.0, 0.0}, Dual<double>{3.0, 0.0}, Dual<double>{4.0, 0.0});
  EXPECT_DOUBLE_EQ(g[0], 9.0);
  EXPECT_DOUBLE_EQ(g[1], 10.0);
  EXPECT_DOUBLE_EQ(g[2], 3.0);
}

TEST(ReverseModeAD_Dual, TrigExp) {
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  auto x = dual_var_of<"x">(xv);
  auto y = dual_var_of<"y">(yv);
  auto g = Equation{exp(x) * sin(y)}.gradient(Dual<double>{xv, 0.0},
                                              Dual<double>{yv, 0.0});
  EXPECT_DOUBLE_EQ(g[0], std::exp(xv) * std::sin(yv));
  EXPECT_DOUBLE_EQ(g[1], std::exp(xv) * std::cos(yv));
}

TEST(ReverseModeAD_Dual, AgreesWithPVResult) {
  double xv = 1.3, yv = 0.7;
  auto xp = var_of<"x">(xv);
  auto yp = var_of<"y">(yv);
  auto xd = dual_var_of<"x">(xv);
  auto yd = dual_var_of<"y">(yv);
  auto gp =
      Equation{xp * yp + sin(xp) + yp * yp + exp(xp + yp)}.gradient(xv, yv);
  auto gd = Equation{xd * yd + sin(xd) + yd * yd + exp(xd + yd)}.gradient(
      Dual<double>{xv, 0.0}, Dual<double>{yv, 0.0});
  EXPECT_DOUBLE_EQ(gd[0], gp[0]);
  EXPECT_DOUBLE_EQ(gd[1], gp[1]);
}

// ===========================================================================
// Hessian via forward-over-reverse (eval_hessian)
// ===========================================================================

TEST(HessianTest, ForwardOverReverse_FunctionValues) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  const std::array<D, 2> pt{D{2.0}, D{3.0}};
  (void)ve.hessian(std::array{2.0, 3.0});
  auto f = ve.evaluate(pt);
  EXPECT_DOUBLE_EQ(f[0].template get<0>(), 6.0);
  EXPECT_DOUBLE_EQ(f[1].template get<0>(), 4.0);
}

TEST(HessianTest, ForwardOverReverse_XY) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  auto H = ve.hessian(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(H[0][0][0], 0.0); // ∂²(x*y)/∂x²
  EXPECT_DOUBLE_EQ(H[0][0][1], 1.0); // ∂²(x*y)/∂x∂y
  EXPECT_DOUBLE_EQ(H[0][1][0], 1.0); // ∂²(x*y)/∂y∂x
  EXPECT_DOUBLE_EQ(H[0][1][1], 0.0); // ∂²(x*y)/∂y²
}

TEST(HessianTest, ForwardOverReverse_Quadratic) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  auto H = ve.hessian(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(H[1][0][0], 2.0);
  EXPECT_DOUBLE_EQ(H[1][0][1], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1][0], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1][1], 0.0);
}

TEST(HessianTest, ForwardOverReverse_WithValues) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  std::array<double, 2> pt{2.0, 3.0};
  auto H = ve.hessian(pt);
  auto f = ve.evaluate(std::array{D{2.0}, D{3.0}});
  EXPECT_DOUBLE_EQ(f[0].template get<0>(), 6.0);
  EXPECT_DOUBLE_EQ(H[0][0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[1][0][0], 2.0);
}

TEST(HessianTest, ForwardOverReverse_TrigFunction) {
  double xv = 1.0, yv = 2.0;
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto ve = Equation(sin(x) * y, x + y);
  auto H = ve.hessian(std::array{xv, yv});
  EXPECT_NEAR(H[0][0][0], -yv * std::sin(xv), 1e-12); // -y*sin(x)
  EXPECT_NEAR(H[0][0][1], std::cos(xv), 1e-12);       // cos(x)
  EXPECT_NEAR(H[0][1][0], std::cos(xv), 1e-12);       // symmetric
  EXPECT_NEAR(H[0][1][1], 0.0, 1e-12);
}

TEST(HessianTest, ForwardOverReverse_Symmetric) {
  // Two outputs, so the vector specialisation is selected.
  double xv = 0.5, yv = 1.5;
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto ve = Equation(exp(x * y), x + y);
  auto H = ve.hessian(std::array{xv, yv});
  EXPECT_NEAR(H[0][0][1], H[0][1][0], 1e-12);
}

// ===========================================================================
// Equation derivative_tensor<2> — forward-mode Hessian, plain scalar variables
// ===========================================================================

TEST(HessianForwardTest, FunctionValues) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  auto f = ve.evaluate(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(f[0], 6.0);
  EXPECT_DOUBLE_EQ(f[1], 4.0);
}

TEST(HessianForwardTest, XY) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  auto H = ve.derivative_tensor<2>(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(H[0][0][0], 0.0);
  EXPECT_DOUBLE_EQ(H[0][0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[0][1][0], 1.0);
  EXPECT_DOUBLE_EQ(H[0][1][1], 0.0);
}

TEST(HessianForwardTest, Quadratic) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  auto H = ve.derivative_tensor<2>(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(H[1][0][0], 2.0);
  EXPECT_DOUBLE_EQ(H[1][0][1], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1][0], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1][1], 0.0);
}

TEST(HessianForwardTest, AgreesWithForwardOverReverse) {
  double xv = 1.5, yv = 2.5;

  using D = Dual<double>;
  Variable<D, FixedString{"x"}> xr;
  Variable<D, FixedString{"y"}> yr;
  auto ve_rev = Equation(xr * yr, xr * xr);
  auto H_rev = ve_rev.hessian(std::array{xv, yv});

  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  auto ve_fwd = Equation(xf * yf, xf * xf);
  auto H_fwd = ve_fwd.derivative_tensor<2>(std::array{xv, yv});

  for (int k = 0; k < 2; ++k)
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        EXPECT_NEAR(H_fwd[k][i][j], H_rev[k][i][j], 1e-12);
}

TEST(HessianForwardTest, WithValues) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(x * y, x * x);
  auto H = ve.derivative_tensor<2>(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(H[0][0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[1][0][0], 2.0);
}

// ===========================================================================
// Scalar Hessian free functions (gradient.hpp)
// ===========================================================================

TEST(ScalarHessianTest, ReverseMode_XY) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto expr = x * y;
  auto H = Equation{expr}.hessian(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(H[0][0], 0.0);
  EXPECT_DOUBLE_EQ(H[0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[1][0], 1.0);
  EXPECT_DOUBLE_EQ(H[1][1], 0.0);
}

TEST(ScalarHessianTest, ReverseMode_QuadraticForm) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto expr = x * x + constant(D{2.0}) * y * y;
  auto H = Equation{expr}.hessian(std::array{1.0, 1.0});
  EXPECT_DOUBLE_EQ(H[0][0], 2.0);
  EXPECT_DOUBLE_EQ(H[0][1], 0.0);
  EXPECT_DOUBLE_EQ(H[1][0], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1], 4.0);
}

TEST(ScalarHessianTest, ReverseMode_Symmetric) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto expr = exp(x * y);
  auto H = Equation{expr}.hessian(std::array{0.5, 1.5});
  EXPECT_NEAR(H[0][1], H[1][0], 1e-12);
}

TEST(ScalarHessianTest, ForwardMode_XY) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto expr = x * y;
  auto H = Equation{expr}.template derivative_tensor<2>(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(H[0][0], 0.0);
  EXPECT_DOUBLE_EQ(H[0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[1][0], 1.0);
  EXPECT_DOUBLE_EQ(H[1][1], 0.0);
}

TEST(ScalarHessianTest, ForwardMode_QuadraticForm) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto expr = x * x + constant(2.0) * y * y;
  auto H = Equation{expr}.template derivative_tensor<2>(std::array{1.0, 1.0});
  EXPECT_DOUBLE_EQ(H[0][0], 2.0);
  EXPECT_DOUBLE_EQ(H[0][1], 0.0);
  EXPECT_DOUBLE_EQ(H[1][0], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1], 4.0);
}

TEST(ScalarHessianTest, ReverseMode_NoValues) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto expr = x * y;
  auto H = Equation{expr}.hessian(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(H[0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[1][0], 1.0);
  EXPECT_DOUBLE_EQ(H[0][0], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1], 0.0);
}

TEST(ScalarHessianTest, ForwardMode_AtAPoint) {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto expr = x * y;
  auto H = Equation{expr}.template derivative_tensor<2>(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(H[0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[1][0], 1.0);
  EXPECT_DOUBLE_EQ(H[0][0], 0.0);
  EXPECT_DOUBLE_EQ(H[1][1], 0.0);
}

TEST(ScalarHessianTest, ForwardAgreesWithReverse) {
  double xv = 0.5, yv = 1.5;

  using D = Dual<double>;
  Variable<D, FixedString{"x"}> xr;
  Variable<D, FixedString{"y"}> yr;
  auto expr_r = exp(xr * yr);
  auto H_rev = Equation{expr_r}.hessian(std::array{xv, yv});

  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  auto expr_f = exp(xf * yf);
  auto H_fwd =
      Equation{expr_f}.template derivative_tensor<2>(std::array{xv, yv});

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      EXPECT_NEAR(H_fwd[i][j], H_rev[i][j], 1e-12);
}

// ===========================================================================
// DerivativeTensorTest — derivative_tensor<Order> free function and Equation
// ===========================================================================

TEST(DerivativeTensorTest, Order1_ScalarVariable) {
  Variable<double, FixedString{"x"}> x;
  auto expr = x * x * x;
  auto T1 = Equation{expr}.template derivative_tensor<1>(std::array{3.0});
  EXPECT_NEAR(T1[0], 27.0, 1e-12);
}

TEST(DerivativeTensorTest, Order1_MatchesGradient) {
  double xv = 1.5, yv = 2.5;
  (void)xv;
  (void)yv;
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto expr = x * x + x * y; // df/dx = 2x+y, df/dy = x

  auto T1 = Equation{expr}.template derivative_tensor<1>(std::array{xv, yv});
  EXPECT_NEAR(T1[0], 2 * xv + yv, 1e-12);
  EXPECT_NEAR(T1[1], xv, 1e-12);
}

TEST(DerivativeTensorTest, Order2_ScalarVariable) {
  Variable<double, FixedString{"x"}> x;
  auto expr = x * x * x;
  auto T2 = Equation{expr}.template derivative_tensor<2>(std::array{2.0});
  EXPECT_NEAR(T2[0][0], 12.0, 1e-12);
}

TEST(DerivativeTensorTest, Order2_MatchesHessian) {
  double xv = 0.7, yv = 1.3;
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> xr;
  Variable<D, FixedString{"y"}> yr;
  auto expr_r = exp(xr * yr);
  auto H_rev = Equation{expr_r}.hessian(std::array{xv, yv});

  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  auto expr_f = exp(xf * yf);
  auto H_fwd =
      Equation{expr_f}.template derivative_tensor<2>(std::array{xv, yv});

  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_NEAR(H_fwd[i][j], H_rev[i][j], 1e-12);
}

TEST(DerivativeTensorTest, Order3_Polynomial) {
  Variable<double, FixedString{"x"}> x;
  auto expr = x * x * x * x;
  auto T3 = Equation{expr}.template derivative_tensor<3>(std::array{2.0});
  EXPECT_NEAR(T3[0][0][0], 48.0, 1e-9);
}

TEST(DerivativeTensorTest, SecondOrderAtAPoint) {
  const double x0 = std::numbers::pi / 4.0;
  Variable<double, FixedString{"x"}> x;
  auto expr = sin(x);
  auto T2 = Equation{expr}.template derivative_tensor<2>(std::array{x0});
  EXPECT_NEAR(T2[0][0], -std::sin(x0), 1e-12);
}

TEST(DerivativeTensorTest, MixedPartials_Symmetric) {
  double xv = 0.5, yv = 1.5;
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto expr = exp(x * y);
  auto H = Equation{expr}.template derivative_tensor<2>(std::array{xv, yv});
  EXPECT_NEAR(H[0][1], H[1][0], 1e-12);
}

TEST(DerivativeTensorTest, Equation_Order1_IsJacobian) {
  double xv = 1.0, yv = 2.0;
  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  auto ve_fwd = Equation(xf * yf, xf * xf);
  auto J_fwd = ve_fwd.derivative_tensor<1>(std::array{xv, yv});

  auto xr = var_of<"x">(xv);
  auto yr = var_of<"y">(yv);
  auto ve_rev = Equation(xr * yr, xr * xr);
  auto J_rev = ve_rev.jacobian(std::array{xv, yv});

  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_NEAR(J_fwd[i][j], J_rev[i][j], 1e-12);
}

TEST(DerivativeTensorTest, Equation_Order2_IsHessianStack) {
  double xv = 1.5, yv = 2.5;
  (void)xv;
  (void)yv;
  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  auto ve_fwd = Equation(xf * yf, xf * xf);
  auto H_fwd = ve_fwd.derivative_tensor<2>(std::array{xv, yv});
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> xr;
  Variable<D, FixedString{"y"}> yr;
  auto ve_rev = Equation(xr * yr, xr * xr);
  auto H_rev = ve_rev.hessian(std::array{2.0, 3.0});

  for (std::size_t k = 0; k < 2; ++k)
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 2; ++j)
        EXPECT_NEAR(H_fwd[k][i][j], H_rev[k][i][j], 1e-12);
}

TEST(DerivativeTensorTest, Equation_Order3_TrigPolynomial) {
  Variable<double, FixedString{"x"}> x;
  auto ve = Equation(x * x * x * x);
  auto T3 = ve.derivative_tensor<3>(std::array{1.0});
  EXPECT_NEAR(T3[0][0][0], 24.0, 1e-9); // one output: no leading axis
}

// ===========================================================================
// Tutorial tests — the standard worked examples for a forward-mode AD library.
// Each test section maps to one tutorial page.
// ===========================================================================

// ---------------------------------------------------------------------------
// Forward Tutorial 1 — Single-variable function
// f(x) = 1 + x + x² + 1/x + log(x)
// ---------------------------------------------------------------------------

TEST(TutorialForward, SingleVar_SymbolicValueAndDerivative) {
  double x0 = 2.0;
  auto x = var_of<"x">(x0);
  auto f = constant(1.0) + x + x * x + constant(1.0) / x + log(x);
  EXPECT_NEAR(f.eval(x0), 1.0 + x0 + x0 * x0 + 1.0 / x0 + std::log(x0), 1e-12);
  double expected_df = 1.0 + 2.0 * x0 - 1.0 / (x0 * x0) + 1.0 / x0;
  EXPECT_NEAR(f.derivative().eval(x0), expected_df, 1e-12);
}

TEST(TutorialForward, SingleVar_DualNumbers) {
  double x0 = 2.0;
  Variable<Dual<double>, FixedString{"x"}> x;
  auto f = 1.0 + x + x * x + 1.0 / x + log(x);
  auto [fv, df] = f.eval(Dual<double>{x0, 1.0});
  EXPECT_NEAR(fv, 1.0 + x0 + x0 * x0 + 1.0 / x0 + std::log(x0), 1e-12);
  EXPECT_NEAR(df, 1.0 + 2.0 * x0 - 1.0 / (x0 * x0) + 1.0 / x0, 1e-12);
}

TEST(TutorialForward, SingleVar_ReverseMode) {
  double x0 = 2.0;
  auto x = var_of<"x">(x0);
  auto f = constant(1.0) + x + x * x + constant(1.0) / x + log(x);
  auto g = Equation{f}.gradient(x0);
  EXPECT_NEAR(g[0], 1.0 + 2.0 * x0 - 1.0 / (x0 * x0) + 1.0 / x0, 1e-12);
}

// ---------------------------------------------------------------------------
// Forward Tutorial 3 / Reverse Tutorial 2 — Multi-variable function
// f(x,y,z) = 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z)
// ---------------------------------------------------------------------------

TEST(TutorialMultiVar, Value) {
  double xv = 1.0, yv = 2.0, zv = 3.0;
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto z = var_of<"z">(zv);
  auto f = constant(1.0) + x + y + z + x * y + y * z + x * z + x * y * z +
           exp(x / y + y / z);
  double expected = 1.0 + xv + yv + zv + xv * yv + yv * zv + xv * zv +
                    xv * yv * zv + std::exp(xv / yv + yv / zv);
  EXPECT_NEAR(f.eval(xv, yv, zv), expected, 1e-12);
}

TEST(TutorialMultiVar, SymbolicPartials) {
  double xv = 1.0, yv = 2.0, zv = 3.0;
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto z = var_of<"z">(zv);
  auto f = constant(1.0) + x + y + z + x * y + y * z + x * z + x * y * z +
           exp(x / y + y / z);
  auto eq = Equation(f);
  auto [dx, dy, dz] =
      eq.template gradient<ddx::DiffMode::Symbolic>(std::array{xv, yv, zv});
  double e = std::exp(xv / yv + yv / zv);
  EXPECT_NEAR(dx, 1.0 + yv + zv + yv * zv + e / yv, 1e-10);
  EXPECT_NEAR(dy, 1.0 + xv + zv + xv * zv + e * (-xv / (yv * yv) + 1.0 / zv),
              1e-10);
  EXPECT_NEAR(dz, 1.0 + yv + xv + xv * yv + e * (-yv / (zv * zv)), 1e-10);
}

TEST(TutorialMultiVar, ReverseGradient) {
  double xv = 1.0, yv = 2.0, zv = 3.0;
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto z = var_of<"z">(zv);
  auto f = constant(1.0) + x + y + z + x * y + y * z + x * z + x * y * z +
           exp(x / y + y / z);
  auto g = Equation{f}.gradient(xv, yv, zv);
  double e = std::exp(xv / yv + yv / zv);
  EXPECT_NEAR(g[0], 1.0 + yv + zv + yv * zv + e / yv, 1e-10);
  EXPECT_NEAR(g[1], 1.0 + xv + zv + xv * zv + e * (-xv / (yv * yv) + 1.0 / zv),
              1e-10);
  EXPECT_NEAR(g[2], 1.0 + yv + xv + xv * yv + e * (-yv / (zv * zv)), 1e-10);
}

TEST(TutorialMultiVar, ForwardReverseGradientAgree) {
  double xv = 1.0, yv = 2.0, zv = 3.0;
  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  Variable<double, FixedString{"z"}> zf;
  auto f_fwd = 1.0 + xf + yf + zf + xf * yf + yf * zf + xf * zf + xf * yf * zf +
               exp(xf / yf + yf / zf);
  auto T1 =
      Equation{f_fwd}.template derivative_tensor<1>(std::array{xv, yv, zv});
  auto xr = var_of<"x">(xv);
  auto yr = var_of<"y">(yv);
  auto zr = var_of<"z">(zv);
  auto f_rev = constant(1.0) + xr + yr + zr + xr * yr + yr * zr + xr * zr +
               xr * yr * zr + exp(xr / yr + yr / zr);
  auto g = Equation{f_rev}.gradient(xv, yv, zv);
  EXPECT_NEAR(T1[0], g[0], 1e-10);
  EXPECT_NEAR(T1[1], g[1], 1e-10);
  EXPECT_NEAR(T1[2], g[2], 1e-10);
}

// ---------------------------------------------------------------------------
// Forward Tutorial 6 — Gradient vector of scalar function
// ---------------------------------------------------------------------------

TEST(TutorialGradient, ForwardModeDerivativeTensor) {
  double xv = 1.0, yv = 0.5;
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto f = sin(x) * cos(y) + exp(x * y);
  auto g = Equation{f}.template derivative_tensor<1>(std::array{xv, yv});
  EXPECT_NEAR(g[0], std::cos(xv) * std::cos(yv) + yv * std::exp(xv * yv),
              1e-12);
  EXPECT_NEAR(g[1], -std::sin(xv) * std::sin(yv) + xv * std::exp(xv * yv),
              1e-12);
}

TEST(TutorialGradient, ForwardReverseAgree) {
  double xv = 1.0, yv = 0.5;
  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  auto T1 =
      Equation{sin(xf) * cos(yf) + exp(xf * yf)}.template derivative_tensor<1>(
          std::array{xv, yv});
  auto xr = var_of<"x">(xv);
  auto yr = var_of<"y">(yv);
  auto g = Equation{sin(xr) * cos(yr) + exp(xr * yr)}.gradient(xv, yv);
  EXPECT_NEAR(T1[0], g[0], 1e-12);
  EXPECT_NEAR(T1[1], g[1], 1e-12);
}

// ---------------------------------------------------------------------------
// Forward Tutorial 11 — Higher-order cross derivatives (dual4th equivalent)
// Uses derivative_tensor<4>
// ---------------------------------------------------------------------------

TEST(TutorialHigherOrder, FourthOrder_SinX) {
  double x0 = std::numbers::pi / 4.0;
  Variable<double, FixedString{"x"}> x;
  auto T4 = Equation{sin(x)}.template derivative_tensor<4>(std::array{x0});
  EXPECT_NEAR(T4[0][0][0][0], std::sin(x0), 1e-9);
}

TEST(TutorialHigherOrder, FourthOrder_ExpX) {
  double x0 = 0.7;
  Variable<double, FixedString{"x"}> x;
  double ev = std::exp(x0);
  EXPECT_NEAR(
      Equation{exp(x)}.template derivative_tensor<2>(std::array{x0})[0][0], ev,
      1e-12);
  EXPECT_NEAR(
      Equation{exp(x)}.template derivative_tensor<3>(std::array{x0})[0][0][0],
      ev, 1e-12);
  EXPECT_NEAR(Equation{exp(x)}.template derivative_tensor<4>(
                  std::array{x0})[0][0][0][0],
              ev, 1e-9);
}

TEST(TutorialHigherOrder, FourthOrder_AllCrossPartialsOfSinXplusY) {
  double xv = 1.0, yv = 1.0;
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto T4 =
      Equation{sin(x + y)}.template derivative_tensor<4>(std::array{xv, yv});
  double expected = std::sin(xv + yv);
  for (int i : {0, 1})
    for (int j : {0, 1})
      for (int k : {0, 1})
        for (int l : {0, 1})
          EXPECT_NEAR(T4[i][j][k][l], expected, 1e-8);
}

TEST(TutorialHigherOrder, ThirdOrder_MixedPartial) {
  double xv = 1.0, yv = 2.0;
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto T3 = Equation{exp(x) * y * y}.template derivative_tensor<3>(
      std::array{xv, yv});
  EXPECT_NEAR(T3[0][0][1], 2.0 * yv * std::exp(xv), 1e-10);
  EXPECT_NEAR(T3[1][1][1], 0.0, 1e-10); // d³(exp(x)*y²)/dy³ = 0
}

// ---------------------------------------------------------------------------
// Forward Tutorial 12/13 — Directional derivatives (scalar and vector)
// ---------------------------------------------------------------------------

TEST(TutorialDirectional, FirstOrder_DualSeeding) {
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  double u = 1.0 / std::sqrt(2.0);
  Variable<Dual<double>, FixedString{"x"}> x;
  Variable<Dual<double>, FixedString{"y"}> y;
  auto [fv, dfdu] =
      (exp(x) * sin(y)).eval(Dual<double>{xv, u}, Dual<double>{yv, u});
  EXPECT_NEAR(fv, std::exp(xv) * std::sin(yv), 1e-12);
  EXPECT_NEAR(dfdu, std::exp(xv), 1e-12);
}

TEST(TutorialDirectional, FirstOrder_ViaGradientDot) {
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  double u = 1.0 / std::sqrt(2.0);
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto T1 = Equation{exp(x) * sin(y)}.template derivative_tensor<1>(
      std::array{xv, yv});
  EXPECT_NEAR(T1[0] * u + T1[1] * u, std::exp(xv), 1e-12);
}

TEST(TutorialDirectional, SecondOrder_HessianContraction) {
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  double u = 1.0 / std::sqrt(2.0);
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto H = Equation{exp(x) * sin(y)}.template derivative_tensor<2>(
      std::array{xv, yv});
  double d2fdu2 =
      H[0][0] * u * u + H[0][1] * u * u + H[1][0] * u * u + H[1][1] * u * u;
  EXPECT_NEAR(d2fdu2, std::exp(xv) * std::cos(yv), 1e-12);
}

TEST(TutorialDirectional, VectorFunction_JacobianTimesDirection) {
  double xv = 1.0, yv = std::numbers::pi / 4.0;
  double u = 1.0 / std::sqrt(2.0);
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
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
  double x0 = std::numbers::pi / 4.0;
  double h = 0.05;
  Variable<double, FixedString{"x"}> x;
  auto f = sin(x);
  double f0 = std::sin(x0);
  auto T1 = Equation{f}.template derivative_tensor<1>(std::array{x0});
  auto T2 = Equation{f}.template derivative_tensor<2>(std::array{x0});
  auto T3 = Equation{f}.template derivative_tensor<3>(std::array{x0});
  auto T4 = Equation{f}.template derivative_tensor<4>(std::array{x0});
  double taylor = f0 + T1[0] * h + T2[0][0] * h * h / 2.0 +
                  T3[0][0][0] * h * h * h / 6.0 +
                  T4[0][0][0][0] * h * h * h * h / 24.0;
  EXPECT_NEAR(taylor, std::sin(x0 + h), 1e-7);
}

TEST(TutorialTaylor, VectorFunction_SecondOrderAccuracy) {
  double xv = 0.0, yv = 0.0, h = 0.1;
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto ve = Equation(sin(x) * cos(y), exp(x + y));
  auto f0 = ve.evaluate(std::array{xv, yv});
  auto J = ve.derivative_tensor<1>(std::array{xv, yv});
  auto H = ve.derivative_tensor<2>(std::array{xv, yv});
  double taylor1 = f0[1] + (J[1][0] + J[1][1]) * h +
                   0.5 * (H[1][0][0] + 2.0 * H[1][0][1] + H[1][1][1]) * h * h;
  EXPECT_NEAR(taylor1, std::exp(2.0 * h), 2e-3);
}

// ---------------------------------------------------------------------------
// Reverse Tutorial 3 — Functions with conditionals / re-evaluation via update
// ---------------------------------------------------------------------------

TEST(TutorialReverseConditionals, AbsGradientChangesSign) {
  EXPECT_DOUBLE_EQ(Equation{abs(var<"x">)}.gradient(3.0)[0], 1.0);
  EXPECT_DOUBLE_EQ(Equation{abs(var<"x">)}.gradient(-3.0)[0], -1.0);
}

TEST(TutorialReverseConditionals, UpdateReevaluatesDerivatives) {
  Variable<double, FixedString{"x"}> x;
  auto eq = Equation(x * x + sin(x));
  EXPECT_NEAR(eq.evaluate(std::array{2.0}), 4.0 + std::sin(2.0), 1e-12);
  EXPECT_NEAR(eq.template gradient<ddx::DiffMode::Symbolic>(std::array{2.0})[0],
              4.0 + std::cos(2.0),
              1e-12); // 2x + cos(x)
  EXPECT_NEAR(eq.evaluate(std::array{5.0}), 25.0 + std::sin(5.0), 1e-12);
  EXPECT_NEAR(eq.template gradient<ddx::DiffMode::Symbolic>(std::array{5.0})[0],
              10.0 + std::cos(5.0), 1e-12);
}

// ---------------------------------------------------------------------------
// Reverse Tutorial 4 — Functions with parameters (Constants)
// ---------------------------------------------------------------------------

TEST(TutorialReverseParams, SingleParameter) {
  double xv = 1.0, p = 3.0;
  auto x = var_of<"x">(xv);
  auto g = Equation{constant(p) * sin(x)}.gradient(xv);
  EXPECT_NEAR(g[0], p * std::cos(xv), 1e-12);
}

TEST(TutorialReverseParams, MultiVarWithParameter) {
  double xv = 2.0, yv = 3.0, p = 4.0;
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto g = Equation{constant(p) * x * y}.gradient(xv, yv);
  EXPECT_NEAR(g[0], p * yv, 1e-12);
  EXPECT_NEAR(g[1], p * xv, 1e-12);
}

// ---------------------------------------------------------------------------
// Reverse Tutorial 6 — Hessian matrix
// ---------------------------------------------------------------------------

TEST(TutorialReverseHessian, QuadraticForm) {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto f = x * x + x * y + y * y;
  auto H = Equation{f}.hessian(std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(H[0][0], 2.0);
  EXPECT_DOUBLE_EQ(H[0][1], 1.0);
  EXPECT_DOUBLE_EQ(H[1][0], 1.0);
  EXPECT_DOUBLE_EQ(H[1][1], 2.0);
}

TEST(TutorialReverseHessian, TrigFunction) {
  using D = Dual<double>;
  double xv = std::numbers::pi / 4.0, yv = std::numbers::pi / 6.0;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto f = sin(x) * cos(y);
  auto H = Equation{f}.hessian(std::array{xv, yv});
  EXPECT_NEAR(H[0][0], -std::sin(xv) * std::cos(yv), 1e-12);
  EXPECT_NEAR(H[0][1], -std::cos(xv) * std::sin(yv), 1e-12);
  EXPECT_NEAR(H[1][0], -std::cos(xv) * std::sin(yv), 1e-12);
  EXPECT_NEAR(H[1][1], -std::sin(xv) * std::cos(yv), 1e-12);
}

TEST(TutorialReverseHessian, GradientFromSameExpression) {
  using D = Dual<double>;
  double xv = 2.0, yv = 3.0;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto f = x * x + x * y + y * y;
  auto g = Equation{f}.gradient(D{xv}, D{yv});
  EXPECT_NEAR(g[0], 2.0 * xv + yv, 1e-12); // df/dx = 2x + y
  EXPECT_NEAR(g[1], xv + 2.0 * yv, 1e-12); // df/dy = x + 2y
}

// ---------------------------------------------------------------------------
// Reverse Tutorial 7/8 — Higher-order derivatives
// ---------------------------------------------------------------------------

TEST(TutorialReverseHigherOrder, SingleVar_SecondAndThird) {
  double x0 = 1.0;
  Variable<double, FixedString{"x"}> x;
  EXPECT_NEAR(
      Equation{sin(x)}.template derivative_tensor<2>(std::array{x0})[0][0],
      -std::sin(x0), 1e-12);
  EXPECT_NEAR(
      Equation{sin(x)}.template derivative_tensor<3>(std::array{x0})[0][0][0],
      -std::cos(x0), 1e-12);
}

TEST(TutorialReverseHigherOrder, MultiVar_Hessian) {
  using D = Dual<double>;
  double xv = 2.0, yv = 3.0;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  auto f = x * x * y + y * y * y;
  auto H = Equation{f}.hessian(std::array{xv, yv});
  EXPECT_NEAR(H[0][0], 2.0 * yv, 1e-12);
  EXPECT_NEAR(H[0][1], 2.0 * xv, 1e-12);
  EXPECT_NEAR(H[1][0], 2.0 * xv, 1e-12);
  EXPECT_NEAR(H[1][1], 6.0 * yv, 1e-12);
}

TEST(TutorialReverseHigherOrder, ForwardReverseHessianAgree) {
  double xv = 0.8, yv = 1.2;
  Variable<double, FixedString{"x"}> xf;
  Variable<double, FixedString{"y"}> yf;
  auto f_fwd = sin(xf) * exp(yf);
  auto H_fwd =
      Equation{f_fwd}.template derivative_tensor<2>(std::array{xv, yv});
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> xr;
  Variable<D, FixedString{"y"}> yr;
  auto f_rev = sin(xr) * exp(yr);
  auto H_rev = Equation{f_rev}.hessian(std::array{xv, yv});
  for (int i : {0, 1})
    for (int j : {0, 1})
      EXPECT_NEAR(H_fwd[i][j], H_rev[i][j], 1e-12);
}

// ===========================================================================
// The freestanding scalar contract + forward-mode drivers.
// These cover the surface a generic templated f(const Scalar*) needs to be
// differentiated the way a second-order dual + hessian(...) is used downstream.
// ===========================================================================

TEST(DualScalarContract, ValAndToDouble) {
  const dual2nd a = embed_constant<double, 2>(3.5);
  EXPECT_DOUBLE_EQ(val(a), 3.5);
  EXPECT_DOUBLE_EQ(to_double(a), 3.5);
  EXPECT_DOUBLE_EQ(val(2.0), 2.0); // identity on a plain scalar
}

TEST(DualScalarContract, ScalarMixingAtDepth) {
  // x = 2, seeded so the outer first-derivative slot tracks d/dx.
  const dual2nd x{Dual<double>{2.0, 1.0}, Dual<double>{1.0, 0.0}};
  // Bare doubles mix at depth 2.
  const dual2nd f = 3.0 * x + x * 2.0 - 1.0;
  EXPECT_DOUBLE_EQ(val(f), 9.0);
  EXPECT_DOUBLE_EQ(f.get<1>().get<0>(), 5.0);
}

TEST(DualScalarContract, PowMaxMinAndComparisons) {
  const dual2nd x = embed_constant<double, 2>(2.0);
  EXPECT_DOUBLE_EQ(val(pow(x, 3.0)), 8.0); // scalar exponent
  EXPECT_DOUBLE_EQ(val(pow(x, x)), 4.0);   // dual exponent
  EXPECT_DOUBLE_EQ(val(max(x, 5.0)), 5.0);
  EXPECT_DOUBLE_EQ(val(min(x, 5.0)), 2.0);
  EXPECT_TRUE(x < 3.0);
  EXPECT_TRUE(x <= 2.0);
  EXPECT_FALSE(x > 2.0);
  EXPECT_TRUE(x == 2.0);
  EXPECT_TRUE(x != 1.0);
}

TEST(DualScalarContract, ImplicitConstantFromScalarAtDepth) {
  // The S{0} / S c = 1.0 idiom must work for nested duals as a zero-derivative
  // constant (generic numeric code written against a plain scalar relies on
  // it).
  const dual2nd a{0};    // brace-init from int
  const dual2nd b = 2.5; // copy-init needs a non-explicit ctor
  const std::vector<dual2nd> v(3, dual2nd{1}); // vector fill from a scalar
  EXPECT_DOUBLE_EQ(val(a), 0.0);
  EXPECT_DOUBLE_EQ(val(b), 2.5);
  EXPECT_DOUBLE_EQ(val(v[0]), 1.0);
  EXPECT_DOUBLE_EQ(b.get<1>().get<0>(), 0.0);
  EXPECT_DOUBLE_EQ(b.get<0>().get<1>(), 0.0);
  EXPECT_DOUBLE_EQ(b.get<1>().get<1>(), 0.0);
}

TEST(ForwardDriver, GradientAndHessianCrossTerm) {
  auto f = [](const auto *x) {
    return x[0] * x[0] * x[1] + x[1] * x[1] * x[1];
  };
  const std::array<double, 2> x{2.0, 3.0};
  const std::span<const double> xs{x.data(), x.size()};

  const auto H = ddx::impl::hessian(f, xs);
  EXPECT_DOUBLE_EQ(val_of(H), 39.0); // 4*3 + 27
  ASSERT_EQ(hess_n(H), 2u);
  EXPECT_DOUBLE_EQ(grad_at(H, 0), 12.0);   // 2 x0 x1
  EXPECT_DOUBLE_EQ(grad_at(H, 1), 31.0);   // x0^2 + 3 x1^2
  EXPECT_DOUBLE_EQ(hess_at(H, 0, 0), 6.0); // 2 x1
  EXPECT_DOUBLE_EQ(hess_at(H, 0, 1), 4.0); // 2 x0
  EXPECT_DOUBLE_EQ(hess_at(H, 1, 0), 4.0);
  EXPECT_DOUBLE_EQ(hess_at(H, 1, 1), 18.0); // 6 x1

  const auto g = ddx::impl::gradient(f, xs);
  EXPECT_DOUBLE_EQ(g[0], 12.0);
  EXPECT_DOUBLE_EQ(g[1], 31.0);
}

TEST(ForwardDriver, IdealMixingHessianMatchesClosedForm) {
  // Mirrors the downstream CALPHAD oracle: G = R T (y0 ln y0 + y1 ln y1).
  const double R = 8.31446261815324;
  const double T = 1000.0;
  auto f = [R, T](const auto *y) {
    using std::log; // ADL still picks ddx::impl::log for the dual argument
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
  EXPECT_NEAR(hess_at(H, 0, 1), 0.0, 1e-6); // no cross term
}

// ===========================================================================
// scoped_value — the RAII seed toggle every driver's sweep loop is built on.
// ===========================================================================

TEST(ScopedValue, SeedsOnEntryAndRestoresOnExit) {
  double slot = 7.5;
  {
    const auto seed = scoped_seed<1.0>(slot);
    EXPECT_DOUBLE_EQ(slot, 1.0);
  }
  EXPECT_DOUBLE_EQ(slot, 7.5); // the previous value, not zero
}

TEST(ScopedValue, RestoreIsBitExactFromAnyBase) {
  // Why the guard saves the old bits instead of undoing arithmetically: a
  // `slot += 1` / `slot -= 1` round trip is not the identity in floating
  // point, and this library cares about the last ULP.
  double slot = 0.1;
  {
    const auto seed = scoped_seed<1.0>(slot);
    EXPECT_DOUBLE_EQ(slot, 1.0);
  }
  EXPECT_EQ(slot, 0.1);            // exact, not merely close
  EXPECT_NE(0.1 + 1.0 - 1.0, 0.1); // what the arithmetic undo would have given
}

TEST(ScopedValue, NestedGuardsOverSiblingScalarsOfOneDual) {
  // The hessian shape: the inner- and outer-derivative seeds of the
  // SAME dual2nd are held by two independently scoped guards.
  dual2nd d{Dual<double>{2.0, 0.0}, Dual<double>{0.0, 0.0}};
  {
    const auto inner = scoped_seed<1.0>(d.value().deriv());
    {
      const auto outer = scoped_seed<1.0>(d.deriv().value());
      EXPECT_DOUBLE_EQ(d.value().deriv(), 1.0); // both live at once
      EXPECT_DOUBLE_EQ(d.deriv().value(), 1.0);
    }
    EXPECT_DOUBLE_EQ(d.deriv().value(), 0.0); // outer cleared
    EXPECT_DOUBLE_EQ(d.value().deriv(), 1.0); // inner survives, unclobbered
  }
  EXPECT_DOUBLE_EQ(d.value().deriv(), 0.0);
  EXPECT_DOUBLE_EQ(d.value().value(), 2.0); // the point never moved
}

#ifdef __cpp_exceptions
// Nothing in ddx throws, but a caller's energy may, and the guarantee this
// asserts is the reason scoped_value saves the old value rather than
// recomputing it.  There is no unwinding to assert about under
// -fno-exceptions, so the test is not there either -- the nesting tests around
// it cover the rest.
TEST(ScopedValue, RestoresWhenTheGuardedScopeThrows) {
  // An energy that throws mid-probe must not leave the dof buffer seeded --
  // which a hand-written `slot = 0` reset cannot guarantee.
  double slot = 0.0;
  auto blow_up = [&slot] {
    const auto seed = scoped_seed<1.0>(slot);
    throw std::runtime_error("energy blew up");
  };
  EXPECT_THROW(blow_up(), std::runtime_error);
  EXPECT_DOUBLE_EQ(slot, 0.0);
}
#endif

TEST(ScopedValue, IsPinnedByItsBase) {
  // A guard restores in its destructor, so exactly one of them may exist per
  // slot: moving one would leave a second destructor writing back a value that
  // has been carried off.  The four deletes are `private pinned`, not four
  // lines here -- this is what they are for.
  using Guard = decltype(scoped_seed<1.0>(std::declval<double &>()));
  static_assert(!std::is_copy_constructible_v<Guard>);
  static_assert(!std::is_move_constructible_v<Guard>);
  static_assert(!std::is_copy_assignable_v<Guard>);
  static_assert(!std::is_move_assignable_v<Guard>);
  SUCCEED();
}

TEST(ScopedValue, CarriesNothingForTheNewValue) {
  // The guard is the reference plus the saved scalar: the new value is moved
  // into the slot by the constructor and never stored beside it.
  static_assert(sizeof(decltype(scoped_seed<1.0>(std::declval<double &>()))) ==
                sizeof(double *) + sizeof(double));
  SUCCEED();
}

TEST(ScopedValue, HoldsARuntimeValueToo) {
  // The same guard rt::equation() makes an arena current with -- a pointer
  // slot and a value that is not a constant.
  int a = 1;
  int b = 2;
  int *slot = &a;
  {
    const ddx::impl::scoped_value hold{slot, &b};
    EXPECT_EQ(slot, &b);
    {
      const ddx::impl::scoped_value inner{slot, &a};
      EXPECT_EQ(slot, &a); // nesting, the way equation() nests
    }
    EXPECT_EQ(slot, &b);
  }
  EXPECT_EQ(slot, &a);
}

TEST(ScopedValue, UsableDuringConstantEvaluation) {
  // gradient.hpp's reverse_mode_hessian and equation.hpp's
  // hessian_forward_over_reverse are constexpr, so the guard must be too.
  static constexpr auto probe = []() constexpr {
    double slot = 3.0;
    double seen = 0.0;
    {
      const auto seed = scoped_seed<1.0>(slot);
      seen = slot;
    }
    return std::array<double, 2>{seen, slot};
  }();
  static_assert(probe[0] == 1.0);
  static_assert(probe[1] == 3.0);
  SUCCEED();
}

TEST(ForwardDriver, RepeatedCallsDoNotLeakSeeds) {
  // A seed left behind by one sweep would show up as a wrong second answer.
  auto f = [](const auto *x) {
    return x[0] * x[0] * x[1] + x[1] * x[1] * x[1];
  };
  const std::array<double, 2> x{2.0, 3.0};
  const std::span<const double> xs{x.data(), x.size()};

  const auto g1 = ddx::impl::gradient(f, xs);
  const auto g2 = ddx::impl::gradient(f, xs);
  EXPECT_DOUBLE_EQ(g1[0], g2[0]);
  EXPECT_DOUBLE_EQ(g1[1], g2[1]);

  const auto H1 = ddx::impl::hessian(f, xs);
  const auto H2 = ddx::impl::hessian(f, xs);
  EXPECT_DOUBLE_EQ(val_of(H1), val_of(H2));
  for (std::size_t i = 0; i < hess_n(H1); ++i) {
    EXPECT_DOUBLE_EQ(grad_at(H1, i), grad_at(H2, i));
    for (std::size_t j = 0; j < hess_n(H1); ++j) {
      EXPECT_DOUBLE_EQ(hess_at(H1, i, j), hess_at(H2, i, j));
    }
  }
}

// ===========================================================================
// Map — the heterogeneous compile-time map.  ValueMap (tests_math.cpp) is the
// homogeneous one; these pin what that one cannot do.
// ===========================================================================

namespace {

using ddx::impl::Map;
using ddx::impl::NamedValue;

constexpr auto kMapX = var<"x">;
constexpr auto kMapN = var<"n", int>;

} // namespace

TEST(MapTest, KeepsTheValueTypeOfEachEntry) {
  constexpr auto m = map(named<"n">(3), named<"x">(1.5));
  static_assert(m.size == 2);
  static_assert(std::is_same_v<decltype(m)::value_type_of<"n">, int>);
  static_assert(std::is_same_v<decltype(m)::value_type_of<"x">, double>);
  EXPECT_EQ(m.get<"n">(), 3);
  EXPECT_DOUBLE_EQ(m.get<"x">(), 1.5);
}

TEST(MapTest, SubscriptReadsTheSameSlotAsGet) {
  constexpr auto m = map(named<"n">(3), named<"x">(1.5));
  EXPECT_EQ(m["n"_s], m.get<"n">());
  EXPECT_DOUBLE_EQ(m["x"_s], m.get<"x">());
}

TEST(MapTest, OwnershipFollowsTheMapNotTheKey) {
  auto m = map(named<"x">(1.5));
  static_assert(std::is_same_v<decltype(m.get<"x">()), double &>);
  static_assert(
      std::is_same_v<decltype(std::as_const(m).get<"x">()), const double &>);
  static_assert(
      std::is_same_v<decltype(map(named<"x">(1.5)).get<"x">()), double>);
}

TEST(MapTest, ContainsAnswersForAbsentAndSimilarKeys) {
  constexpr auto m = map(named<"x">(1.5));
  static_assert(m.contains<"x">());
  static_assert(!m.contains<"y">());
  static_assert(!m.contains<"xy">());
}

TEST(MapTest, EverySpellingBuildsTheSameType) {
  constexpr Map by_keyword{named<"n">(3), named<"x">(1.5)};
  constexpr Map by_label{NamedValue{"n"_s, 3}, NamedValue{"x"_s, 1.5}};
  constexpr Map by_variable{NamedValue{kMapN, 3}, NamedValue{kMapX, 1.5}};
  constexpr Map by_tagged_named{named(kMapN, 3), named(kMapX, 1.5)};
  constexpr Map<NamedValue<"n", int>, NamedValue<"x", double>> by_type{{3},
                                                                       {1.5}};

  static_assert(std::is_same_v<decltype(by_keyword), decltype(by_label)>);
  static_assert(std::is_same_v<decltype(by_keyword), decltype(by_variable)>);
  static_assert(
      std::is_same_v<decltype(by_keyword), decltype(by_tagged_named)>);
  static_assert(std::is_same_v<decltype(by_keyword), decltype(by_type)>);
  static_assert(by_keyword == by_label && by_label == by_variable &&
                by_variable == by_tagged_named && by_tagged_named == by_type);
}

TEST(MapTest, KeysAreInEntryOrderNotSorted) {
  constexpr auto m = map(named<"z">(1), named<"a">(2), named<"m">(3));
  static_assert(m.keys()[0] == std::string_view{"z"});
  static_assert(m.keys()[1] == std::string_view{"a"});
  static_assert(m.keys()[2] == std::string_view{"m"});
}

TEST(MapTest, SetWritesTheNamedSlotInPlace) {
  auto m = map(named<"a">(1.0), named<"b">(2.0));
  m.set<"a">(7.0);
  m["b"_s] = 9.0;
  EXPECT_DOUBLE_EQ(m.get<"a">(), 7.0);
  EXPECT_DOUBLE_EQ(m.get<"b">(), 9.0);
}

TEST(MapTest, InsertAndEraseReturnNewMaps) {
  constexpr auto m = map(named<"n">(3), named<"x">(1.5));
  constexpr auto with_y = m.insert(named<"y">('c'));
  static_assert(with_y.size == 3 && with_y.get<"y">() == 'c');
  static_assert(with_y.get<"n">() == 3 && with_y.get<"x">() == 1.5);
  static_assert(m.size == 2, "insert leaves the original alone");

  constexpr auto without_n = with_y.erase<"n">();
  static_assert(without_n.size == 2 && !without_n.contains<"n">());
  static_assert(without_n.get<"x">() == 1.5 && without_n.get<"y">() == 'c');
  static_assert(without_n.keys()[0] == std::string_view{"x"},
                "erase keeps the order of what is left");

  static_assert(m.insert(named<"p">(1), named<"q">(2)).size == 4);
  static_assert(m.insert().size == 2);
}

TEST(MapTest, EraseThenInsertRetypesASlot) {
  constexpr auto m = map(named<"n">(3), named<"x">(1.5));
  constexpr auto retyped = m.erase<"n">().insert(named<"n">(2.5f));
  static_assert(std::is_same_v<decltype(retyped)::value_type_of<"n">, float>);
  static_assert(retyped.get<"n">() == 2.5f);
}

TEST(MapTest, ForEachVisitsEveryEntryInOrder) {
  constexpr auto m = map(named<"n">(3), named<"x">(1.5));
  std::string labels;
  double sum = 0.0;
  m.for_each([&](auto key, const auto &v) {
    labels += key.name;
    sum += static_cast<double>(v);
  });
  EXPECT_EQ(labels, "nx");
  EXPECT_DOUBLE_EQ(sum, 4.5);

  auto mutable_map = map(named<"a">(1.0), named<"b">(2.0));
  mutable_map.for_each([](auto, auto &v) { v *= 10.0; });
  EXPECT_DOUBLE_EQ(mutable_map.get<"a">(), 10.0);
  EXPECT_DOUBLE_EQ(mutable_map.get<"b">(), 20.0);
}

TEST(MapTest, HoldsValuesNoExpressionWouldAccept) {
  constexpr auto m = map(named<"name">(std::string_view{"ddx"}),
                         named<"order">(2u), named<"tol">(1e-9));
  static_assert(m.get<"name">() == std::string_view{"ddx"});
  static_assert(m.get<"order">() == 2u);
  EXPECT_DOUBLE_EQ(m.get<"tol">(), 1e-9);
}

TEST(MapTest, EqualityIsKeysInOrderAndValues) {
  constexpr auto m = map(named<"n">(3), named<"x">(1.5));
  static_assert(m == map(named<"n">(3), named<"x">(1.5)));
  static_assert(!(m == map(named<"n">(4), named<"x">(1.5))));
  static_assert(!std::is_same_v<decltype(m),
                                decltype(map(named<"x">(1.5), named<"n">(3)))>);
}

TEST(MapTest, EmptyMapIsUsable) {
  constexpr Map<> m{};
  static_assert(m.size == 0);
  static_assert(m.keys().empty());
  static_assert(!m.contains<"x">());
  static_assert(m.insert(named<"x">(1.0)).get<"x">() == 1.0);
}

// Relaxing NamedValue to any object type must not disturb its numeric use.
TEST(MapTest, NamedValueStillDrivesTheExpressionSide) {
  constexpr auto x = var<"x">;
  constexpr auto y = var<"y">;
  const auto f = x * y;
  EXPECT_DOUBLE_EQ(f.eval(named<"y">(2.0), named<"x">(4.0)), 8.0);
  EXPECT_DOUBLE_EQ(Equation{f}.gradient(named<"y">(2.0), named<"x">(4.0))[0],
                   2.0);
}
