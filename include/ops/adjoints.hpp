#pragma once

#include "ops/numeric.hpp"

#include <array>
#include <cmath>
#include <concepts>

// The reverse-mode rule for every op that is not one of unary_math.hpp's
// eighteen transcendentals, arranged the way those are: the body is written
// against `Numeric T` with `const auto &` parameters, so `T` appears only
// inside it.  The compile-time sweep instantiates these at a number and the
// graph builder at `RTExpression`, where the same source builds nodes instead
// of computing.  Both sides call these -- there is no second spelling.
//
// No rule reads its own primal `f`: the compile-time cache never stores a
// node's own slot (symbolic/sweep.hpp), so `pow` and `hypot` recompute it from
// the operands.  That costs the graph nothing -- the recomputed node interns
// straight back onto the primal.
//
// `max`/`min` are the one pair both sides do not share; see ExtremumOpFn.
namespace ddx::impl::detail {

// `sign` has to mean the number for an arithmetic T and the Sign node for a
// graph handle.  Ordinary lookup finds this one and ADL finds RTExpression's
// hidden friend; the constraint is what keeps them from competing.
// sign(0) is 0: the abs/max/min subgradient convention every engine shares.
// a - a reaches only ±0 and NaN, so a NaN operand poisons the derivative
// rather than picking a side.
template <Numeric T>
  requires std::totally_ordered<T>
[[nodiscard]] constexpr T sign(const T &a) noexcept {
  return a > T{} ? T{1} : a < T{} ? T{-1} : T{a - a};
}

// The three ops whose `reads_primals` is false take their operands anyway and
// discard them, so one forwarder shape serves every op.  Their caller passes
// cache slots that were never written -- value-initialised zeroes, well
// defined to read and, here, unread.
template <Numeric T> struct SumOpFn {
  [[nodiscard]] static constexpr std::array<T, 2>
  adjoints(const auto &adj, const auto &l, const auto &r) noexcept {
    (void)l;
    (void)r;
    return {adj, adj};
  }
};

// For c = a*b the differential is da*b + a*db, so the adjoint reaching `a`
// multiplies on the LEFT of b and the one reaching `b` on the RIGHT of a.
// Sided, not complete: a non-commutative scalar would also need a transpose.
template <Numeric T> struct MultiplyOpFn {
  [[nodiscard]] static constexpr std::array<T, 2>
  adjoints(const auto &adj, const auto &a, const auto &b) noexcept {
    return {adj * b, a * adj};
  }
};

template <Numeric T> struct NegateOpFn {
  [[nodiscard]] static constexpr std::array<T, 1>
  adjoints(const auto &adj, const auto &u) noexcept {
    (void)u;
    return {-adj};
  }
};

// `a / b` is right division, so the two spellings do not fold into one: see
// DivideOp::derivative for the argument.  Which branch a graph handle takes is
// decided by its scalar, through rt/expressions.hpp's opt-in.
template <Numeric T> struct DivideOpFn {
  [[nodiscard]] static constexpr std::array<T, 2>
  adjoints(const auto &adj, const auto &a, const auto &b) noexcept {
    // Divide by b twice rather than once by b*b: the square overflows past
    // ~1.3e154 where the quotient does not, and on a graph the a/b node is the
    // primal quotient already.  The two branches now differ only in which side
    // the adjoint multiplies on, which is the whole of the distinction.
    if constexpr (CCommutativeMultiply<T>) {
      return {adj / b, -adj * (a / b) / b};
    } else {
      return {adj / b, -((a / b) * adj) / b};
    }
  }
};

template <Numeric T> struct PowOpFn {
  [[nodiscard]] static constexpr std::array<T, 2>
  adjoints(const auto &adj, const auto &a, const auto &b) noexcept {
    using std::pow, std::log;
    const T p = pow(a, b);
    return {adj * b * pow(a, b - T{1}), adj * p * log(a)};
  }
};

// lhs is y, rhs is x.
template <Numeric T> struct Atan2OpFn {
  [[nodiscard]] static constexpr std::array<T, 2>
  adjoints(const auto &adj, const auto &y, const auto &x) noexcept {
    using std::hypot;
    // Scaled by hypot: x² + y² overflows past 1e154.
    const T h = hypot(x, y);
    return {adj * (x / h) / h, -adj * (y / h) / h};
  }
};

template <Numeric T> struct HypotOpFn {
  [[nodiscard]] static constexpr std::array<T, 2>
  adjoints(const auto &adj, const auto &x, const auto &y) noexcept {
    using std::hypot;
    const T h = hypot(x, y);
    // x / h first: the quotient is in [-1, 1] where x * adj can overflow.
    return {adj * (x / h), adj * (y / h)};
  }
};

template <Numeric T> struct AbsOpFn {
  [[nodiscard]] static constexpr std::array<T, 1>
  adjoints(const auto &adj, const auto &u) noexcept {
    return {adj * sign(u)};
  }
};

template <Numeric T> struct SignOpFn {
  [[nodiscard]] static constexpr std::array<T, 1>
  adjoints(const auto &adj, const auto &u) noexcept {
    (void)adj;
    (void)u;
    return {T{}};
  }
};

// The one rule the two sides do not share.  A branch cannot work at a graph
// type -- there are no comparisons on a node -- so the kink is written as a
// sign, exact away from the tie and NaN at it.  ExtremumOp keeps a branching
// rule of its own: on numbers a compare is cheaper than three arithmetic ops,
// and it can hand a tie half to each side where this gives NaN.
template <Numeric T, bool IsMax> struct ExtremumOpFn {
  [[nodiscard]] static constexpr std::array<T, 2>
  adjoints(const auto &adj, const auto &l, const auto &r) noexcept {
    const T s = sign(l - r);
    const T lo = (T{1} - s) / T{2};
    const T hi = (T{1} + s) / T{2};
    if constexpr (IsMax) {
      return {adj * hi, adj * lo};
    } else {
      return {adj * lo, adj * hi};
    }
  }
};

template <Numeric T> using MaxOpFn = ExtremumOpFn<T, true>;
template <Numeric T> using MinOpFn = ExtremumOpFn<T, false>;

} // namespace ddx::impl::detail
