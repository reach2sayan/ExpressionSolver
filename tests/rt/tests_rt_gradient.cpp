#include "ddx.hpp"
#include "rt/bridge.hpp"
#include "rt/derivative.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

// ===========================================================================
// Runtime differentiation (rt/derivative.hpp)
//
// The sweep is checked against ddx itself: to_graph lowers a compile-time
// expression into the graph, so the same function can be differentiated by
// Equation and by the sweep and the two must agree.  That makes these tests
// comparisons against code already covered elsewhere, rather than against
// hand-computed constants.
// ===========================================================================

namespace {

// Value and every partial, graph against Equation.  Symbols are matched by
// name: Equation reports partials in canonical alphabetical order, the builder
// numbers them in the order they were first seen.
template <ddx::impl::CExpression E, std::size_t N>
void expect_agrees_with_ddx(const E &e, std::array<double, N> pt) {
  ddx::rt::Builder<> b;
  const auto root = ddx::rt::to_graph(b, e);
  const auto g = ddx::rt::gradient(b, root.id(b));
  const auto values = ddx::rt::evaluate_all(b, std::span<const double>{pt});

  const double expected = std::apply(
      [&](auto... a) { return ddx::Equation{e}.evaluate(a...); }, pt);
  EXPECT_NEAR(values[g.value], expected,
              1e-12 * std::max(1.0, std::abs(expected)));

  const auto expected_grad = std::apply(
      [&](auto... a) { return ddx::Equation{e}.gradient(a...); }, pt);

  auto sorted = b.symbols();
  std::ranges::sort(sorted);
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    const auto it = std::ranges::find(b.symbols(), sorted[i]);
    ASSERT_NE(it, b.symbols().end());
    const auto slot = static_cast<std::size_t>(it - b.symbols().begin());
    const double want = expected_grad[i];
    EXPECT_NEAR(values[g.partial[slot]], want,
                1e-12 * std::max(1.0, std::abs(want)))
        << "partial d/d" << sorted[i];
  }
}

constexpr auto x = ddx::var<"x">;
constexpr auto y = ddx::var<"y">;
constexpr auto z = ddx::var<"z">;

TEST(RtGradient, Arithmetic) {
  expect_agrees_with_ddx(x * y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(x + y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(x - y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(x / y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(-x * y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(x * x * x + y * y - x * y, std::array{1.3, 2.1});
}

TEST(RtGradient, BinaryFunctions) {
  expect_agrees_with_ddx(pow(x, y), std::array{1.7, 2.3});
  expect_agrees_with_ddx(atan2(x, y), std::array{1.3, 2.1});
  expect_agrees_with_ddx(hypot(x, y), std::array{1.3, 2.1});
}

TEST(RtGradient, Transcendentals) {
  expect_agrees_with_ddx(exp(x) * sin(y), std::array{0.7, 1.1});
  expect_agrees_with_ddx(log(x) * sqrt(y), std::array{1.3, 2.1});
  expect_agrees_with_ddx(tanh(x) + cbrt(y), std::array{0.4, 2.1});
  expect_agrees_with_ddx(erf(x) * atanh(y), std::array{0.4, 0.3});
  expect_agrees_with_ddx(asin(x) + acos(y), std::array{0.4, 0.3});
  expect_agrees_with_ddx(acosh(x) + asinh(y), std::array{1.9, 0.6});
  expect_agrees_with_ddx(sinh(x) * cosh(y), std::array{0.4, 0.6});
  expect_agrees_with_ddx(tan(x) + log10(y), std::array{0.4, 2.6});
  expect_agrees_with_ddx(cos(x) / exp(y), std::array{0.4, 0.6});
}

TEST(RtGradient, SharedAndNested) {
  expect_agrees_with_ddx((x * y) * (x * y) + sin(x * y), std::array{0.3, 0.7});
  expect_agrees_with_ddx(sin(cos(exp(x * y))), std::array{0.3, 0.7});
  expect_agrees_with_ddx(exp(x * y) + log(z) / tanh(x),
                         std::array{0.3, 0.7, 1.4});
}

// max, min and abs have no entry in the compile-time derivative table, so they
// are checked against central differences on the interpreter instead.
TEST(RtGradient, SelectingOpsAgainstFiniteDifferences) {
  using ddx::rt::Builder;
  using ddx::rt::RTExpression;
  const auto check = [](auto build, std::array<double, 2> pt) {
    Builder<> b;
    const auto vx = var(b, "x");
    const auto vy = var(b, "y");
    const auto f = build(vx, vy);
    const auto g = ddx::rt::gradient(b, f.id(b));
    const auto values = ddx::rt::evaluate_all(b, std::span<const double>{pt});
    for (std::size_t i = 0; i < 2; ++i) {
      const double h = 1e-6;
      auto hi = pt, lo = pt;
      hi[i] += h;
      lo[i] -= h;
      const double fd = (ddx::rt::evaluate(b, g.value, hi) -
                         ddx::rt::evaluate(b, g.value, lo)) /
                        (2 * h);
      EXPECT_NEAR(values[g.partial[i]], fd, 1e-5 * std::max(1.0, std::abs(fd)));
    }
  };
  check([](RTExpression<> a, RTExpression<> b2) { return max(a, b2); },
        {1.3, 2.1});
  check([](RTExpression<> a, RTExpression<> b2) { return min(a, b2); },
        {1.3, 2.1});
  check([](RTExpression<> a, RTExpression<> b2) { return abs(a * b2); },
        {1.3, 2.1});
  check([](RTExpression<> a, RTExpression<> b2) { return abs(a * b2); },
        {-1.3, 2.1});
}

TEST(RtGradient, DerivFromValueReusesThePrimalNode) {
  // exp's derivative is the primal, so the sweep must not build a second exp.
  ddx::rt::Builder<> b;
  const auto vx = var(b, "x");
  const auto f = exp(vx);
  const auto g = ddx::rt::gradient(b, f.id(b));
  EXPECT_EQ(g.partial[0], f.id(b));
}

TEST(RtGradient, UnusedSymbolHasZeroPartial) {
  ddx::rt::Builder<> b;
  const auto vx = var(b, "x");
  const auto vy = var(b, "y");
  const auto f = vx * vx;
  const auto g = ddx::rt::gradient(b, f.id(b));
  ASSERT_EQ(g.partial.size(), 2u);
  EXPECT_TRUE(b.is_constant(g.partial[1], 0.0));
  (void)vy;
}

} // namespace
