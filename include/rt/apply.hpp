#pragma once

#include "ops/operations.hpp" // pow_impl, atan2_impl, hypot_impl, max_impl, min_impl, abs_impl
#include "ops/unary_math.hpp" // the eighteen descriptors
#include "rt/opcode.hpp"

#include <functional> // std::plus and friends, for the table's eval column

namespace ddx::rt {

namespace detail {

// A switch instantiates every case, so support has to be probed per operation
// rather than per category.  It probes the free function, not the functor: a
// matrix satisfies pow_impl's `Numeric auto` and fails in the body, which is a
// hard error, where `pow(a, b)` simply does not resolve.  Arithmetic
// short-circuits -- a fundamental type gives ADL no namespace to search.
template <typename Fn> inline constexpr bool is_field_op_v = false;
template <> inline constexpr bool is_field_op_v<std::plus<>> = true;
template <> inline constexpr bool is_field_op_v<std::minus<>> = true;
template <> inline constexpr bool is_field_op_v<std::multiplies<>> = true;
template <> inline constexpr bool is_field_op_v<std::divides<>> = true;
template <> inline constexpr bool is_field_op_v<std::negate<>> = true;

#define DDX_RT_PROBE1(fn, Op, label, ...)                                      \
  template <impl::Numeric T>                                                   \
  inline constexpr bool probes_##Op =                                          \
      impl::CArithmetic<T> || requires(const T &u) { fn(u); };
DDX_UNARY_MATH_TABLE(DDX_RT_PROBE1)
#undef DDX_RT_PROBE1

#define DDX_RT_PROBE2(fn, Op, label, ...)                                      \
  template <impl::Numeric T>                                                   \
  inline constexpr bool probes_##Op =                                          \
      impl::CArithmetic<T> || requires(const T &a, const T &b) { fn(a, b); };
DDX_RT_BINARY_TABLE(DDX_RT_PROBE2)
#undef DDX_RT_PROBE2

template <impl::Numeric T>
inline constexpr bool probes_Abs =
    impl::CArithmetic<T> || requires(const T &u) { abs(u); };
template <impl::Numeric T> inline constexpr bool probes_Neg = true;
// sign_impl only compares, so ordering is the whole requirement.
template <impl::Numeric T>
inline constexpr bool probes_Sign =
    impl::CArithmetic<T> || std::totally_ordered<T>;

template <typename Fn, impl::Numeric T, bool Ok, impl::Numeric... Args>
[[nodiscard]] constexpr T supported(const Args &...args) noexcept {
  if constexpr (Ok || is_field_op_v<Fn>) {
    return T{Fn{}(args...)};
  } else {
    return T{};
  }
}

} // namespace detail

// Dispatch an OpCode onto ddx's own functors.  Templated on the scalar, so the
// interpreter, constant folding and the graph builder share one definition: at
// T = double it computes, at T = RTExpression it builds nodes.
template <impl::Numeric T>
[[nodiscard]] constexpr T apply(OpCode op, const T &u) noexcept {
  switch (op) {
#define DDX_RT_APPLY(fn, Op, label, functor, ...)                              \
  case OpCode::Op:                                                             \
    return detail::supported<functor, T, detail::probes_##Op<T>>(u);
    DDX_RT_UNARY_TABLE(DDX_RT_APPLY)
#undef DDX_RT_APPLY
#define DDX_RT_APPLY(fn, Op, label)                                            \
  case OpCode::Op:                                                             \
    return detail::supported<impl::detail::Op##Fn<T>, T,                       \
                             detail::probes_##Op<T>>(u);
    DDX_UNARY_MATH_TABLE(DDX_RT_APPLY)
#undef DDX_RT_APPLY
  default:
    return T{};
  }
}

template <impl::Numeric T>
[[nodiscard]] constexpr T apply(OpCode op, const T &l, const T &r) noexcept {
  switch (op) {
#define DDX_RT_APPLY(fn, Op, label, functor, ...)                              \
  case OpCode::Op:                                                             \
    return detail::supported<functor, T, detail::probes_##Op<T>>(l, r);
    DDX_RT_BINARY_TABLE(DDX_RT_APPLY)
#undef DDX_RT_APPLY
  default:
    return T{};
  }
}

} // namespace ddx::rt
