#include "ddx.hpp"
#include "rt/bridge.hpp"
#include "rt/derivative.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

// "Runtime" names when the *structure* is decided, not when the arithmetic
// runs, so a graph over constant-visible values is a constant expression.
// Everything here is a static_assert; the TEST bodies exist so the file
// reports.  The frozen graph is the boundary -- Boost.Graph's containers are
// not constexpr, and neither is the JIT.

namespace {
using ddx::rt::Builder;
using ddx::rt::RTExpression;

consteval double build_and_evaluate() {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * x + y;
  return ddx::rt::evaluate(b, f.id(b), std::array{3.0, 4.0});
}
static_assert(build_and_evaluate() == 13.0);

// The rewrites fold at compile time exactly as they do at run time.
consteval bool rewrites_hold() {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  return (x + RTExpression<>{0}).id(b) == x.id(b) &&
         (x * RTExpression<>{1}).id(b) == x.id(b) && (-(-x)).id(b) == x.id(b) &&
         ((y / x) * x).id(b) == y.id(b) && ((x * y)).id(b) == (y * x).id(b);
}
static_assert(rewrites_hold());

// Interning is a compile-time property too: the same subtree twice is one node.
consteval std::size_t nodes_for_repeated_subtree() {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = exp(x) * sin(y);
  const auto g = exp(x) * sin(y);
  return f.id(b) == g.id(b) ? b.size() : 0;
}
static_assert(nodes_for_repeated_subtree() == 5);

// A Jacobian, swept and evaluated entirely during constant evaluation.
consteval double partial_of_product() {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * y;
  const auto g = ddx::rt::build_jacobian_impl(b, f.id(b));
  return ddx::rt::evaluate_all(b, std::array{3.0, 4.0})[g.partial[0]];
}
static_assert(partial_of_product() == 4.0);

// <cmath> is not constexpr; folding a libm call is a GCC extension and clang
// rejects it outright, so this one sweep is GCC-only.
#if defined(__GNUC__) && !defined(__clang__)
consteval double transcendental_jacobian() {
  Builder<> b;
  const auto x = var(b, "x");
  const auto f = sin(x) * x;
  const auto g = ddx::rt::build_jacobian_impl(b, f.id(b));
  const auto v = ddx::rt::evaluate_all(b, std::array{0.0});
  return v[g.partial[0]]; // d(x sin x)/dx at 0 is sin 0 + 0 cos 0
}
static_assert(transcendental_jacobian() == 0.0);
#endif

// Lowering a typed tree into a graph is a constant expression as well, so the
// two representations can be compared without running anything.
consteval bool bridge_agrees_with_equation() {
  constexpr auto x = ddx::var<"x">;
  constexpr auto y = ddx::var<"y">;
  Builder<> b;
  const auto root = ddx::rt::to_graph(b, x * y + x);
  const double viaGraph =
      ddx::rt::evaluate(b, root.id(b), std::array{3.0, 4.0});
  return viaGraph == ddx::Equation{x * y + x}.evaluate(3.0, 4.0);
}
static_assert(bridge_agrees_with_equation());

TEST(RtConstexpr, EverythingAboveIsAStaticAssert) {
  EXPECT_DOUBLE_EQ(build_and_evaluate(), 13.0);
  EXPECT_DOUBLE_EQ(partial_of_product(), 4.0);
  EXPECT_EQ(nodes_for_repeated_subtree(), 5u);
}

} // namespace
