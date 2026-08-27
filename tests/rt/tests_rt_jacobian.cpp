#include "ddx.hpp"
#include "rt/bridge.hpp"
#include "rt/derivative.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <string>
#include <utility>
#include <vector>

// to_graph lowers a compile-time expression into the graph, so the same
// function can be differentiated by Equation and by the sweep, and the two
// must agree.

namespace {

// An overflow is an answer, and the two engines have to reach the same one:
// EXPECT_NEAR cannot say that, since inf - inf is NaN.
void expect_close(double got, double want) {
  if (std::isfinite(want) && std::isfinite(got)) {
    EXPECT_NEAR(got, want, 1e-12 * std::max(1.0, std::abs(want)));
  } else {
    EXPECT_EQ(std::isnan(got), std::isnan(want));
    if (!std::isnan(want)) {
      EXPECT_EQ(got, want);
    }
  }
}

// Symbols are matched by name: Equation reports partials alphabetically, the
// builder numbers them in first-seen order.
template <ddx::impl::CExpression E, std::size_t N>
void expect_agrees_with_ddx(const E &e, std::array<double, N> pt) {
  ddx::rt::Builder<> b;
  const auto root = ddx::rt::to_graph(b, e);
  const auto g = ddx::rt::build_jacobian_impl(b, root.id(b));
  const auto values = ddx::rt::evaluate_all(b, std::span<const double>{pt});

  const double expected = std::apply(
      [&](auto... a) { return ddx::Equation{e}.evaluate(a...); }, pt);
  expect_close(values[g.value], expected);

  const auto expected_grad = std::apply(
      [&](auto... a) { return ddx::Equation{e}.jacobian(a...); }, pt);

  // Both sides number their symbols alphabetically, so slot i is partial i.
  for (std::size_t i = 0; i < b.symbols().size(); ++i) {
    SCOPED_TRACE("partial d/d" + std::string{b.symbols()[i]});
    expect_close(values[g.partial[i]], expected_grad[i]);
  }
}

constexpr auto x = ddx::var<"x">;
constexpr auto y = ddx::var<"y">;
constexpr auto z = ddx::var<"z">;

TEST(RtJacobian, Arithmetic) {
  expect_agrees_with_ddx(x * y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(x + y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(x - y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(x / y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(-x * y, std::array{1.3, 2.1});
  expect_agrees_with_ddx(x * x * x + y * y - x * y, std::array{1.3, 2.1});
}

TEST(RtJacobian, BinaryFunctions) {
  expect_agrees_with_ddx(pow(x, y), std::array{1.7, 2.3});
  expect_agrees_with_ddx(atan2(x, y), std::array{1.3, 2.1});
  expect_agrees_with_ddx(hypot(x, y), std::array{1.3, 2.1});
}

TEST(RtJacobian, Transcendentals) {
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

TEST(RtJacobian, SharedAndNested) {
  expect_agrees_with_ddx((x * y) * (x * y) + sin(x * y), std::array{0.3, 0.7});
  expect_agrees_with_ddx(sin(cos(exp(x * y))), std::array{0.3, 0.7});
  expect_agrees_with_ddx(exp(x * y) + log(z) / tanh(x),
                         std::array{0.3, 0.7, 1.4});
}

// max, min and abs have no entry in the compile-time derivative table, so they
// are checked against central differences on the interpreter instead.
TEST(RtJacobian, SelectingOpsAgainstFiniteDifferences) {
  using ddx::rt::Builder;
  using ddx::rt::RTExpression;
  const auto check = [](auto build, std::array<double, 2> pt) {
    Builder<> b;
    const auto vx = var(b, "x");
    const auto vy = var(b, "y");
    const auto f = build(vx, vy);
    const auto g = ddx::rt::build_jacobian_impl(b, f.id(b));
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

TEST(RtJacobian, DerivFromValueReusesThePrimalNode) {
  // exp's derivative is the primal, so the sweep must not build a second exp.
  ddx::rt::Builder<> b;
  const auto vx = var(b, "x");
  const auto f = exp(vx);
  const auto g = ddx::rt::build_jacobian_impl(b, f.id(b));
  EXPECT_EQ(g.partial[0], f.id(b));
}

// Every row of both tables must name a rule that instantiates at the number
// *and* at the graph.  That is the whole of ops/adjoints.hpp's contract: add an
// op without a shared rule and this stops compiling, which is what keeps a
// second spelling from being written for one of them.
#define DDX_TEST_UNARY_RULE(fn, Op, label, functor, Desc)                       \
  static_assert(requires(const double &a) {                                    \
    ddx::impl::detail::Desc<double>::adjoints(a, a);                           \
  });                                                                          \
  static_assert(requires(const ddx::rt::RTExpression<> &a) {                   \
    ddx::impl::detail::Desc<ddx::rt::RTExpression<>>::adjoints(a, a);          \
  });
DDX_RT_UNARY_TABLE(DDX_TEST_UNARY_RULE)
#undef DDX_TEST_UNARY_RULE

#define DDX_TEST_BINARY_RULE(fn, Op, label, functor, Desc)                      \
  static_assert(requires(const double &a) {                                    \
    ddx::impl::detail::Desc<double>::adjoints(a, a, a);                        \
  });                                                                          \
  static_assert(requires(const ddx::rt::RTExpression<> &a) {                   \
    ddx::impl::detail::Desc<ddx::rt::RTExpression<>>::adjoints(a, a, a);       \
  });
DDX_RT_BINARY_TABLE(DDX_TEST_BINARY_RULE)
#undef DDX_TEST_BINARY_RULE

// Why ExtremumOp keeps a rule of its own, and why AbsOpFn's unqualified sign()
// reaches RTExpression's hidden friend rather than the one in ops/adjoints.hpp.
// If this ever becomes true both facts go stale silently.
static_assert(!std::totally_ordered<ddx::rt::RTExpression<>>);

// A node stands for a double, so it answers the commutativity question the way
// a double does -- which is what picks DivideOpFn's branch.
static_assert(ddx::impl::CCommutativeMultiply<ddx::rt::RTExpression<>>);

TEST(RtJacobian, UnusedSymbolHasZeroPartial) {
  ddx::rt::Builder<> b;
  const auto vx = var(b, "x");
  const auto vy = var(b, "y");
  const auto f = vx * vx;
  const auto g = ddx::rt::build_jacobian_impl(b, f.id(b));
  ASSERT_EQ(g.partial.size(), 2u);
  EXPECT_TRUE(b.is_constant(g.partial[1], 0.0));
  (void)vy;
}

// Wider coverage of the ops that now read their rule from ops/adjoints.hpp,
// including the points the spellings were chosen for: hypot and atan2 are
// written to divide by the hypotenuse first so they survive where x*x + y*y
// overflows, and pow's left partial has to stay finite at a == 0.
TEST(RtJacobian, SharedRulesAgreeAcrossTheScaleRange) {
  for (const auto& [a, b] : std::vector<std::pair<double, double>>{
           {1.3, 2.1}, {1e200, 1e200}, {1e-200, 1e-200}, {3.0, 1e-8}}) {
    expect_agrees_with_ddx(hypot(x, y), std::array{a, b});
    expect_agrees_with_ddx(atan2(x, y), std::array{a, b});
    expect_agrees_with_ddx(x / y, std::array{a, b});
    expect_agrees_with_ddx(x * y, std::array{a, b});
    expect_agrees_with_ddx(x + y, std::array{a, b});
  }
  for (const auto& [a, b] :
       std::vector<std::pair<double, double>>{{1.7, 2.3}, {0.5, 3.0}}) {
    expect_agrees_with_ddx(pow(x, y), std::array{a, b});
  }
}

// neg had no cross-check against Equation at all; abs only had one against
// finite differences, which cannot see a sign convention that is wrong at 0.
TEST(RtJacobian, UnaryRulesAgree) {
  for (const double a : {1.3, -1.3, 1e-300, 4.0}) {
    expect_agrees_with_ddx(-x, std::array{a});
    expect_agrees_with_ddx(-(x * x), std::array{a});
  }
}

// build_symbolic_jacobian is a selectable public mode that nothing in the tree
// instantiated before this test: DiffMode::Symbolic carries a tangent up the
// graph rather than an adjoint down it, so it is the one caller that wants the
// bare partial.  It reads it from the same rules, seeded with a unit adjoint.
TEST(RtJacobian, ForwardModeAgreesWithReverse) {
  const auto check = [](auto build, std::array<double, 2> pt) {
    ddx::rt::Builder<> b;
    const auto vx = var(b, "x");
    const auto vy = var(b, "y");
    const auto f = build(vx, vy);
    const auto fwd =
        ddx::rt::build_jacobian_impl<ddx::DiffMode::Symbolic>(b, f.id(b));
    const auto rev =
        ddx::rt::build_jacobian_impl<ddx::DiffMode::Reverse>(b, f.id(b));
    const auto values = ddx::rt::evaluate_all(b, std::span<const double>{pt});
    for (std::size_t i = 0; i < 2; ++i) {
      const double want = values[rev.partial[i]];
      EXPECT_NEAR(values[fwd.partial[i]], want,
                  1e-12 * std::max(1.0, std::abs(want)))
          << "partial " << i;
    }
  };
  using ddx::rt::RTExpression;
  check([](RTExpression<> a, RTExpression<> b2) { return a * b2 + sin(a); },
        {0.3, 0.7});
  check([](RTExpression<> a, RTExpression<> b2) { return a / b2; }, {1.3, 2.1});
  check([](RTExpression<> a, RTExpression<> b2) { return pow(a, b2); },
        {1.7, 2.3});
  check([](RTExpression<> a, RTExpression<> b2) { return hypot(a, b2); },
        {1.3, 2.1});
  check([](RTExpression<> a, RTExpression<> b2) { return -(a * b2); },
        {1.3, 2.1});
  check([](RTExpression<> a, RTExpression<> b2) { return max(a, b2); },
        {1.3, 2.1});
}

} // namespace
