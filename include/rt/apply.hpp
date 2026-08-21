#pragma once

#include "expr/operations.hpp" // pow_impl, atan2_impl, hypot_impl, max_impl, min_impl, abs_impl
#include "expr/unary_math.hpp" // the eighteen descriptors
#include "rt/opcode.hpp"

#include <functional> // std::plus and friends, for the table's eval column

namespace ddx::rt {

namespace detail {

// A switch instantiates every case, so a scalar with no `sin` would fail to
// compile the whole table even for a graph that never mentions one.  Numeric
// admits matrices and quaternions, for which most of the table is meaningless,
// while Dual supports all of it -- that is what forward mode is.  So the test
// has to be per operation, not per category.
//
// It probes the free function, not the functor.  pow_impl takes `Numeric auto`,
// so a matrix satisfies its signature and the failure lands in the body, which
// is not the immediate context and is a hard error.  `pow(a, b)` simply does
// not resolve for a matrix, which is substitution failure and detectable.
// Arithmetic short-circuits because a fundamental type has no associated
// namespace for argument-dependent lookup to search.
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
// sign_impl branches on comparisons rather than calling anything, so ordering
// is the whole requirement.
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

// Dispatch an OpCode onto ddx's own functors.  Templated on the scalar so the
// interpreter, constant folding and the graph builder all share one definition:
// at T = double it computes, at T = RTExpression it builds nodes.  Overloaded
// on arity rather than numbered, because the argument list already says which
// is which.
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
