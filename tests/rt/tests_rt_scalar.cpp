#include "ddx.hpp"
#include "rt/derivative.hpp"
#include "rt/interpret.hpp"
#include <array>
#include <gtest/gtest.h>

#include <cmath>

// ===========================================================================
// The graph over a non-arithmetic scalar (rt/builder.hpp, rt/apply.hpp)
//
// Numeric admits matrices and quaternions -- expressions.hpp says so, and the
// compile-time side honours it.  The rewrites that assume a commuting product
// must therefore ask rather than assume, and this is the type that catches
// them: M2 is associative under + and *, commutative under + only.
// ===========================================================================

namespace {

// A 2x2 matrix: associative on + and *, commutative on + only.  It satisfies
// CFieldLike, so Numeric admits it -- exactly the case ddx's comment names.
struct M2 {
  std::array<double, 4> a{};
  constexpr M2() = default;
  constexpr M2(int k) : a{double(k), 0, 0, double(k)} {} // k * identity
  constexpr M2(double x, double y, double z, double w) : a{x, y, z, w} {}
  friend constexpr M2 operator+(M2 l, M2 r) {
    return {l.a[0] + r.a[0], l.a[1] + r.a[1], l.a[2] + r.a[2], l.a[3] + r.a[3]};
  }
  friend constexpr M2 operator-(M2 l, M2 r) {
    return {l.a[0] - r.a[0], l.a[1] - r.a[1], l.a[2] - r.a[2], l.a[3] - r.a[3]};
  }
  friend constexpr M2 operator*(M2 l, M2 r) {
    return {
        l.a[0] * r.a[0] + l.a[1] * r.a[2], l.a[0] * r.a[1] + l.a[1] * r.a[3],
        l.a[2] * r.a[0] + l.a[3] * r.a[2], l.a[2] * r.a[1] + l.a[3] * r.a[3]};
  }
  friend constexpr M2 operator/(M2 l, M2 r) { // right division, l * r^-1
    const double d = r.a[0] * r.a[3] - r.a[1] * r.a[2];
    const M2 inv{r.a[3] / d, -r.a[1] / d, -r.a[2] / d, r.a[0] / d};
    return l * inv;
  }
  friend constexpr M2 operator-(M2 u) {
    return {-u.a[0], -u.a[1], -u.a[2], -u.a[3]};
  }
  friend constexpr bool operator==(M2 l, M2 r) = default;
};
static_assert(ddx::impl::CFieldLike<M2>, "M2 is field-like");
static_assert(ddx::impl::Numeric<M2>, "so Numeric admits it");
static_assert(!ddx::impl::CCommutativeMultiply<M2>,
              "and its * does not commute");

TEST(RtScalar, CommutativeSwapIsAskedNotAssumed) {
  ddx::rt::Builder<M2> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  // A matrix product must not be canonicalised by swapping operands.
  EXPECT_NE((x * y).id(b), (y * x).id(b));

  ddx::rt::Builder<double> d;
  const auto dx = var(d, "x");
  const auto dy = var(d, "y");
  EXPECT_EQ((dx * dy).id(d), (dy * dx).id(d));
}

TEST(RtScalar, OnlyTheCancellationThatHoldsForAnyScalarFires) {
  ddx::rt::Builder<M2> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  // (n/d)*d is n whatever T is: `/` is right division, so d^-1 meets d.
  EXPECT_EQ(((y / x) * x).id(b), y.id(b));
  // d*(n/d) is d n d^-1, which is n only when the factors commute.
  EXPECT_NE((x * (y / x)).id(b), y.id(b));
}

TEST(RtScalar, EvaluatesInTheScalarsOwnArithmetic) {
  ddx::rt::Builder<M2> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const std::array<M2, 2> pt{M2{1, 2, 3, 4}, M2{5, 6, 7, 8}};
  const auto v = ddx::rt::evaluate(b, (x * y).id(b), pt);
  EXPECT_EQ(v, (M2{19, 22, 43, 50}));
}

// The four operators are all CFieldLike promises; a transcendental over a
// matrix is meaningless, and must not stop the type being usable at all.
TEST(RtScalar, UnsupportedOpsDoNotBreakTheInstantiation) {
  ddx::rt::Builder<M2> b;
  const auto x = var(b, "x");
  // Nothing here can fail at run time: the assertion is that it instantiates
  // and builds a node at all, which is what naming the id proves.
  EXPECT_NE((x + x * x - x / x).id(b), ddx::rt::no_node);
}

} // namespace

namespace {

// Forward mode falls out of the interpreter rather than being a second engine,
// exactly as it does on the compile-time side: eval_seeded instantiated at
// Dual<T> is what makes ddx's forward mode, and evaluate_all takes its
// arithmetic from the point's element type for the same reason.
TEST(RtScalar, ADualPointCarriesDerivativesThroughTheSameWalk) {
  ddx::rt::Builder<> b; // the graph itself is over double
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = exp(x) * sin(y);

  using D = ddx::dual;
  const std::array<D, 2> seeded{D{1.0, 1.0}, D{2.0, 0.0}}; // seed x
  const auto v = ddx::rt::evaluate(b, f.id(b), seeded);

  EXPECT_NEAR(v.value(), std::exp(1.0) * std::sin(2.0), 1e-14);
  EXPECT_NEAR(v.deriv(), std::exp(1.0) * std::sin(2.0), 1e-14);

  // And it agrees with the reverse sweep over the same graph.
  const auto g = ddx::rt::gradient(b, f.id(b));
  const auto plain = ddx::rt::evaluate_all(b, std::array{1.0, 2.0});
  EXPECT_NEAR(plain[g.partial[0]], v.deriv(), 1e-14);
}

} // namespace
