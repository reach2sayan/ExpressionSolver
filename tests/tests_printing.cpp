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
constexpr auto px = diff::var<"x">;
constexpr auto py = diff::var<"y">;
constexpr auto pz = diff::var<"z">;
} // namespace

// Unary nodes used to print as their bare child -- the operator itself was
// dropped on the floor.  These three are the regression gate.
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
                  make_all_constant_except<diff::FixedString{"x"}>(px * py)),
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
  const diff::Constant<diff::Dual<double>> c{diff::Dual<double>{1.5, 2.0}};
  EXPECT_EQ(std::format("{}", c), "1.5+2e");
  EXPECT_EQ(std::format("{::.2f}", c), "1.50+2.00e");
}

TEST(ExpressionPrinting, RejectsUnparseableValueSpec) {
  // The spec is parsed at compile time for literals, so a bad one has to be
  // reached through vformat to be observable as a throw.  It is rejected by
  // the value_type's own formatter, which is the whole grammar there is.
  EXPECT_THROW((void)std::vformat("{:q}", std::make_format_args(px)),
               std::format_error);
}

TEST(ExpressionPrinting, EquationPrintsFunctionsAndGradientRows) {
  auto eq = diff::Equation{px * py};
  const auto text = std::format("{}", eq);
  EXPECT_TRUE(text.starts_with("f0: x * y\n"));
  EXPECT_NE(text.find("grad: "), std::string::npos);
}

// The dual family shares one formatter base (util/fmt.hpp): parse() and the
// number/punctuation primitives are written once, so a spec means the same
// thing in both and the iostream spelling cannot drift from std::format.
TEST(DualPrinting, EachDualRendersItsOwnShape) {
  const diff::Dual<double> d{1.5, 2.0};
  diff::TaylorDual<double, 3> t;
  t.c = {1.0, 2.0, 3.0, 4.0};

  EXPECT_EQ(std::format("{}", d), "1.5+2e");
  EXPECT_EQ(std::format("{}", t), "1+2e+3e^2+4e^3");
}

TEST(DualPrinting, OneSpecReachesEveryPart) {
  const diff::Dual<double> d{1.5, 2.0};
  diff::TaylorDual<double, 2> t;
  t.c = {1.0, 2.0, 3.0};

  EXPECT_EQ(std::format("{:.2f}", d), "1.50+2.00e");
  EXPECT_EQ(std::format("{:.2f}", t), "1.00+2.00e+3.00e^2");
}

// Without bracketing, Dual<Dual<double>> reads as one four-term series.
TEST(DualPrinting, NestedCoefficientsAreBracketed) {
  const diff::Dual<diff::Dual<double>> d{diff::Dual<double>{1.0, 2.0},
                                         diff::Dual<double>{3.0, 4.0}};
  EXPECT_EQ(std::format("{}", d), "(1+2e)+(3+4e)e");
}

TEST(DualPrinting, StreamInserterMatchesFormat) {
  const diff::Dual<double> d{1.5, 2.0};
  diff::TaylorDual<double, 2> t;
  t.c = {1.0, 2.0, 3.0};

  std::ostringstream oss;
  oss << d << '|' << t;
  EXPECT_EQ(oss.str(), std::format("{}|{}", d, t));
}

// Every dual is reachable as an expression leaf, which is the reason both have
// to be formattable and not just Dual.
TEST(DualPrinting, AllThreeWorkAsExpressionLeaves) {
  EXPECT_EQ(std::format("{}", diff::Constant<diff::Dual<double>>{
                                  diff::Dual<double>{5.0, 1.0}}),
            "5+1e");
  EXPECT_EQ(std::format("{}", diff::Constant<diff::TaylorDual<double, 2>>{
                                  diff::TaylorDual<double, 2>{5.0}}),
            "5+0e+0e^2");
}
