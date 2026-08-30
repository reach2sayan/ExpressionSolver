#include "tests_drivers_fixtures.hpp"

TEST(Ownership, EquationSubtreeAccessorsWorkOnTemporaries) {
  // The nodes are empty types now, so the rvalue overload copies nothing.
  auto x = var<"x">;
  auto y = var<"y">;
  EXPECT_DOUBLE_EQ(Equation(x * y).get<1>().eval(2.0), 2.0);
  EXPECT_DOUBLE_EQ(Equation(x * y)[idx<2>()].eval(4.0), 4.0);
}
// A plain arithmetic lambda is not a CExpression, so it must keep routing to
// the raw-callable branch.
TEST(SeededExprEnergy, RawLambdaIsNotMistakenForAGraph) {
  auto f = [](auto y) {
    using std::log;
    return y[0] * log(y[0]) + y[1] * log(y[1]);
  };
  static_assert(!ddx::impl::CSeededExprEnergy<decltype(f)>,
                "untagged lambda must not be treated as expr-template energy");
  static_assert(!ddx::impl::CExpression<decltype(f)>,
                "a lambda is not an expression graph");
}
