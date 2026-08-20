#include "tests_common.hpp"


// ===========================================================================
// Expression printing.
//
// Every one of these was unverifiable before expr/format.hpp: printing had no
// test at all, which is how a unary node that printed only its child (sin(x)
// as "x") survived.  The assertions are on std::format, since operator<< is a
// forward to it.
// ===========================================================================

namespace {
constexpr auto px = ddx::var<"x">;
constexpr auto py = ddx::var<"y">;
constexpr auto pz = ddx::var<"z">;
} // namespace

// A unary node must name its operator, not print as its bare child.
TEST(ExpressionPrinting, UnaryNodesNameTheirOperator) {
  EXPECT_EQ(std::format("{}", sin(px)), "sin(x)");
  EXPECT_EQ(std::format("{}", -px), "-x");
  EXPECT_EQ(std::format("{}", abs(px)), "abs(x)");
  EXPECT_EQ(std::format("{}", exp(px * py)), "exp(x * y)");
}

TEST(ExpressionPrinting, ParenthesesOnlyWherePrecedenceDemands) {
  EXPECT_EQ(std::format("{}", px * py + sin(px)), "x * y + sin(x)");
  EXPECT_EQ(std::format("{}", (px + py) * pz), "(x + y) * z");
  EXPECT_EQ(std::format("{}", px / (py / pz)), "x / (y / z)");
  EXPECT_EQ(std::format("{}", px / py / pz), "x / y / z");
  EXPECT_EQ(std::format("{}", -(px + py)), "-(x + y)");
  EXPECT_EQ(std::format("{}", -px * py), "-x * y");
}

// a - b is built as a + (-b); without the peephole every subtraction in the
// library would read "x + -y".
TEST(ExpressionPrinting, SubtractionSurvivesItsLowering) {
  EXPECT_EQ(std::format("{}", px - py), "x - y");
  EXPECT_EQ(std::format("{}", px - (py + pz)), "x - (y + z)");
  EXPECT_EQ(std::format("{}", px - py * pz), "x - y * z");
  EXPECT_EQ(std::format("{}", 2.0 * px * py + sin(px) - pz),
            "2 * x * y + sin(x) - z");
}

TEST(ExpressionPrinting, FunctionStyleOps) {
  EXPECT_EQ(std::format("{}", pow(px, 2.0)), "pow(x, 2)");
  EXPECT_EQ(std::format("{}", atan2(py, px)), "atan2(y, x)");
  EXPECT_EQ(std::format("{}", hypot(px, py)), "hypot(x, y)");
  EXPECT_EQ(std::format("{}", max(px, py + pz)), "max(x, y + z)");
}

// Names are variables and numbers are constants, so the only leaf that needs
// marking is the frozen variable: it prints as a name but differentiates to
// zero, which is what "_c" says.
TEST(ExpressionPrinting, FrozenVariablesPrintAsConstants) {
  EXPECT_EQ(std::format("{}", px * py), "x * y");
  EXPECT_EQ(
      std::format("{}",
                  make_all_constant_except<ddx::impl::FixedString{"x"}>(px * py)),
      "x * y_c");
}

TEST(ExpressionPrinting, ValueSpecReachesEveryNumber) {
  EXPECT_EQ(std::format("{::.3f}", 2.0 * px), "2.000 * x");
  EXPECT_EQ(std::format("{::.1e}", 2.0 * px + 3.0), "2.0e+00 * x + 3.0e+00");
  // The leading ':' is optional -- the whole spec is the value-spec.
  EXPECT_EQ(std::format("{:.3f}", 2.0 * px), "2.000 * x");
}

TEST(ExpressionPrinting, StreamInserterMatchesFormat) {
  const auto e = px * py + sin(px) - pz;
  std::ostringstream oss;
  oss << e;
  EXPECT_EQ(oss.str(), std::format("{}", e));
}

// A dual-valued constant is reachable through PDV, so the leaf printer has to
// cope with a value_type that is not an arithmetic type.
TEST(ExpressionPrinting, DualValuedLeavesPrint) {
  const ddx::impl::Constant<ddx::impl::Dual<double>> c{ddx::impl::Dual<double>{1.5, 2.0}};
  EXPECT_EQ(std::format("{}", c), "1.5+2ε");
  EXPECT_EQ(std::format("{::.2f}", c), "1.50+2.00ε");
}

TEST(ExpressionPrinting, RejectsUnparseableValueSpec) {
  // The spec is parsed at compile time for literals, so a bad one has to be
  // reached through vformat to be observable as a throw.  It is rejected by
  // the value_type's own formatter, which is the whole grammar there is.
  EXPECT_THROW((void)std::vformat("{:q}", std::make_format_args(px)),
               std::format_error);
}

TEST(ExpressionPrinting, EquationPrintsFunctionsAndGradientRows) {
  auto eq = ddx::Equation{px * py};
  const auto text = std::format("{}", eq);
  EXPECT_TRUE(text.starts_with("f0: x * y\n"));
  EXPECT_NE(text.find("grad: "), std::string::npos);
}

// The dual family shares one formatter base (util/fmt.hpp): parse() and the
// number/punctuation primitives are written once, so a spec means the same
// thing in both and the iostream spelling cannot drift from std::format.
TEST(DualPrinting, EachDualRendersItsOwnShape) {
  const ddx::impl::Dual<double> d{1.5, 2.0};
  ddx::impl::TaylorDual<double, 3> t;
  t.c = {1.0, 2.0, 3.0, 4.0};

  EXPECT_EQ(std::format("{}", d), "1.5+2ε");
  EXPECT_EQ(std::format("{}", t), "1+2ε+3ε^2+4ε^3");
}

TEST(DualPrinting, OneSpecReachesEveryPart) {
  const ddx::impl::Dual<double> d{1.5, 2.0};
  ddx::impl::TaylorDual<double, 2> t;
  t.c = {1.0, 2.0, 3.0};

  EXPECT_EQ(std::format("{:.2f}", d), "1.50+2.00ε");
  EXPECT_EQ(std::format("{:.2f}", t), "1.00+2.00ε+3.00ε^2");
}

// Without bracketing, Dual<Dual<double>> reads as one four-term series.
TEST(DualPrinting, NestedCoefficientsAreBracketed) {
  const ddx::impl::Dual<ddx::impl::Dual<double>> d{ddx::impl::Dual<double>{1.0, 2.0},
                                         ddx::impl::Dual<double>{3.0, 4.0}};
  EXPECT_EQ(std::format("{}", d), "(1+2ε)+(3+4ε)ε");
}

TEST(DualPrinting, StreamInserterMatchesFormat) {
  const ddx::impl::Dual<double> d{1.5, 2.0};
  ddx::impl::TaylorDual<double, 2> t;
  t.c = {1.0, 2.0, 3.0};

  std::ostringstream oss;
  oss << d << '|' << t;
  EXPECT_EQ(oss.str(), std::format("{}|{}", d, t));
}

// Every dual is reachable as an expression leaf, which is the reason both have
// to be formattable and not just Dual.
TEST(DualPrinting, AllThreeWorkAsExpressionLeaves) {
  EXPECT_EQ(std::format("{}", ddx::impl::Constant<ddx::impl::Dual<double>>{
                                  ddx::impl::Dual<double>{5.0, 1.0}}),
            "5+1ε");
  EXPECT_EQ(std::format("{}", ddx::impl::Constant<ddx::impl::TaylorDual<double, 2>>{
                                  ddx::impl::TaylorDual<double, 2>{5.0}}),
            "5+0ε+0ε^2");
}
