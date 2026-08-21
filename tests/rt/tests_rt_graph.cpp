#include "ddx.hpp"
#include "rt/bridge.hpp"
#include "rt/derivative.hpp"
#include "rt/graph.hpp"

#include <gtest/gtest.h>

// ===========================================================================
// The frozen graph (rt/graph.hpp)
//
// Freezing is what makes the graph static: the builder can still be added to,
// a Graph<> cannot.  A CSR row is a set, so the assertions that matter are that
// operand *position* survives the compression and that nothing reachable is
// dropped.
// ===========================================================================

namespace {
using ddx::rt::Graph;
using ddx::rt::NodeId;

TEST(RtGraph, OperandOrderSurvivesCompression) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x / y; // not commutative: the slots must not be swapped
  const auto graph = Graph<>::freeze(b, std::array{f.id(b)});

  const auto [lhs, rhs] = graph.operands(f.id(b));
  EXPECT_EQ(lhs, x.id(b));
  EXPECT_EQ(rhs, y.id(b));
}

TEST(RtGraph, UnaryNodeHasOneOperand) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto f = sin(x);
  const auto graph = Graph<>::freeze(b, std::array{f.id(b)});
  const auto [lhs, rhs] = graph.operands(f.id(b));
  EXPECT_EQ(lhs, x.id(b));
  EXPECT_EQ(rhs, ddx::rt::no_node);
}

TEST(RtGraph, EverythingTheOutputsReachIsLive) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = exp(x) * sin(y);
  const auto dead = cos(x) + y; // built, never an output
  const auto graph = Graph<>::freeze(b, std::array{f.id(b)});

  EXPECT_TRUE(graph.live(f.id(b)));
  EXPECT_TRUE(graph.live(x.id(b)));
  EXPECT_TRUE(graph.live(y.id(b)));
  EXPECT_FALSE(graph.live(dead.id(b)));
  EXPECT_LT(graph.live_count(), graph.size());
}

TEST(RtGraph, CarriesSymbolsAndOutputs) {
  ddx::rt::Builder<> b;
  const auto root = ddx::rt::to_graph(b, ddx::var<"x"> * ddx::var<"y">);
  const auto graph = ddx::rt::GraphBuilder{b}.value(root).gradient().build();

  EXPECT_EQ(graph.symbols().size(), 2u);
  EXPECT_EQ(graph.outputs().size(), 3u); // value plus two partials
}

TEST(RtGraph, IdOrderIsTopological) {
  ddx::rt::Builder<> b;
  const auto root = ddx::rt::to_graph(b, sin(ddx::var<"x"> * ddx::var<"y">) +
                                             exp(ddx::var<"x">));
  const auto graph = ddx::rt::GraphBuilder{b}.value(root).gradient().build();

  // Codegen emits in id order in one pass, which is only correct if every
  // child precedes its parent.
  for (NodeId v = 0; v < graph.size(); ++v) {
    for (const NodeId child : graph.operands(v)) {
      if (child != ddx::rt::no_node) {
        EXPECT_LT(child, v) << "node " << v << " has a forward reference";
      }
    }
  }
}

} // namespace
