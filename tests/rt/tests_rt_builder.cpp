#include "rt/expr.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

// ===========================================================================
// The runtime builder (rt/builder.hpp, rt/expr.hpp)
//
// The compile-time side folds in the operator factories so a tree is born
// folded and the garbage is never instantiated.  The builder does the same
// thing with ids: a rewrite returns an existing node, so the assertions here
// are on node identity rather than on node count.
// ===========================================================================

namespace {
using ddx::rt::Builder;
using ddx::rt::RTExpression;

TEST(RtBuilder, IdenticalSubtreesInternToOneNode) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = exp(x) * sin(y);
  const std::size_t after_first = b.size();
  const auto g = exp(x) * sin(y);

  EXPECT_EQ(f.id(b), g.id(b));
  EXPECT_EQ(b.size(), after_first);
}

TEST(RtBuilder, CommutativeOperandsCanonicalise) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  EXPECT_EQ((x * y).id(b), (y * x).id(b));
  EXPECT_EQ((x + y).id(b), (y + x).id(b));
  EXPECT_NE((x / y).id(b), (y / x).id(b));
}

TEST(RtBuilder, IdentityRewrites) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");

  EXPECT_EQ((x + RTExpression<>{0}).id(b), x.id(b));
  EXPECT_EQ((x * RTExpression<>{1}).id(b), x.id(b));
  EXPECT_EQ((x / RTExpression<>{1}).id(b), x.id(b));
  EXPECT_EQ((-(-x)).id(b), x.id(b));
  EXPECT_EQ(pow(x, RTExpression<>{1}).id(b), x.id(b));

  EXPECT_TRUE(b.is_constant((x * RTExpression<>{0}).id(b), 0.0));
  EXPECT_TRUE(b.is_constant((RTExpression<>{0} / x).id(b), 0.0));
  EXPECT_TRUE(b.is_constant((x / x).id(b), 1.0));
  EXPECT_TRUE(b.is_constant(pow(x, RTExpression<>{0}).id(b), 1.0));

  // (n/d)*d -> n, the DAG form of the compile-time rule: on interned nodes the
  // denominator match is an id compare.
  EXPECT_EQ(((y / x) * x).id(b), y.id(b));
  EXPECT_EQ((x * (y / x)).id(b), y.id(b));
}

TEST(RtBuilder, LiteralsFoldBeforeReachingTheGraph) {
  Builder<> b;
  const auto x = var(b, "x");
  const std::size_t before = b.size();
  const auto c = RTExpression<>{2} * RTExpression<>{3} + RTExpression<>{1};
  EXPECT_TRUE(c.pending());
  EXPECT_DOUBLE_EQ(c.literal(), 7.0);
  EXPECT_EQ(b.size(), before);

  EXPECT_TRUE(
      b.is_constant((RTExpression<>{2} * RTExpression<>{3}).id(b), 6.0));
  (void)x;
}

TEST(RtBuilder, ConstantOperandsFold) {
  Builder<> b;
  const auto x = var(b, "x");
  EXPECT_TRUE(b.is_constant((lit(b, 2.0) * lit(b, 3.0)).id(b), 6.0));
  EXPECT_TRUE(b.is_constant(sqrt(lit(b, 9.0)).id(b), 3.0));
  (void)x;
}

TEST(RtBuilder, SymbolsInternByName) {
  Builder<> b;
  EXPECT_EQ(var(b, "x").id(b), var(b, "x").id(b));
  EXPECT_NE(var(b, "x").id(b), var(b, "y").id(b));
  ASSERT_EQ(b.symbols().size(), 2u);
  EXPECT_EQ(b.symbols()[0], "x");
}

TEST(RtInterpreter, EvaluatesSharedNodesOnce) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = (x * y) + sin(x * y);
  const std::array<double, 2> pt{0.5, 1.5};
  const double xy = 0.75;
  EXPECT_NEAR(ddx::rt::evaluate(b, f.id(b), pt), xy + std::sin(xy), 1e-15);
}

} // namespace
