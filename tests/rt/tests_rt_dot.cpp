#include "ddx.hpp"
#include "rt/dot.hpp"
#include "rt/graph.hpp"

#include <gtest/gtest.h>

#include <format>
#include <sstream>
#include <string>

// ===========================================================================
// The graph as DOT (rt/dot.hpp)
//
// A debugging view, so what is tested is that it says the true things: the
// symbols by name, the operand order of a division, and which nodes the live
// sweep pruned.
// ===========================================================================

namespace {
using ddx::rt::Graph;

TEST(RtDot, NamesSymbolsAndOperations) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto graph = Graph<>::freeze(b, std::array{(x / y).id(b)});
  const auto dot = ddx::rt::Dot{graph}.str();

  EXPECT_NE(dot.find("digraph"), std::string::npos);
  EXPECT_NE(dot.find("x"), std::string::npos);
  EXPECT_NE(dot.find("y"), std::string::npos);
  EXPECT_NE(dot.find("/"), std::string::npos);
  // The root is an output; a symbol is a leaf.
  EXPECT_NE(dot.find("doubleoctagon"), std::string::npos);
  EXPECT_NE(dot.find("box"), std::string::npos);
}

TEST(RtDot, LabelsTheSlotsOfAnOrderSensitiveNode) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");

  // A division has to say which operand is the denominator...
  const auto div = Graph<>::freeze(b, std::array{(x / y).id(b)});
  const auto div_dot = ddx::rt::Dot{div}.str();
  EXPECT_NE(div_dot.find("label=1]"), std::string::npos);

  // ...and a sum has nothing to say: both slots mean the same thing.
  ddx::rt::Builder<> c;
  const auto p = var(c, "p");
  const auto q = var(c, "q");
  const auto sum = Graph<>::freeze(c, std::array{(p + q).id(c)});
  EXPECT_EQ(ddx::rt::Dot{sum}.str().find("label=1]"), std::string::npos);
}

TEST(RtDot, LiveViewDrawsOnlyWhatCodegenEmits) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto unused = sin(x); // in the arena, but no output reaches it
  const auto graph = Graph<>::freeze(b, std::array{(x * x).id(b)});
  ASSERT_FALSE(graph.live(unused.id(b)));

  const auto dot = ddx::rt::Dot{graph}.str();
  EXPECT_EQ(dot.find("sin"), std::string::npos);
  EXPECT_NE(dot.find("*"), std::string::npos);

  // The filter drops vertices, it does not renumber them: a node keeps the id
  // codegen and every error message call it by.
  EXPECT_NE(dot.find(std::format("{}: *", (x * x).id(b))), std::string::npos);
}

TEST(RtDot, AllShowsWhatTheFreezePruned) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto unused = sin(x);
  const auto graph = Graph<>::freeze(b, std::array{(x * x).id(b)});
  ASSERT_FALSE(graph.live(unused.id(b)));

  const auto dot = ddx::rt::Dot{graph, ddx::rt::Scope::All}.str();
  EXPECT_NE(dot.find("sin"), std::string::npos);
  EXPECT_NE(dot.find("dashed"), std::string::npos);
}

// Streams and formats alike, as an expression and the JIT's IR do.
TEST(RtDot, StreamsAndFormatsAlike) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto graph = Graph<>::freeze(b, std::array{sin(x).id(b)});
  const ddx::rt::Dot dot{graph};

  std::ostringstream os;
  os << dot;
  EXPECT_EQ(os.str(), std::format("{}", dot));
}

} // namespace
