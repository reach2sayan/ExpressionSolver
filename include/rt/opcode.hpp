#pragma once

#include "ops/unary_math.hpp" // DDX_UNARY_MATH_TABLE

#include <array>
#include <cstdint>
#include <string_view>

// The runtime mirror of the compile-time operation set.  Rows are (factory
// spelling, enumerator, label), the shape DDX_UNARY_MATH_TABLE already has, so
// the two sets cannot drift.
namespace ddx::rt {

#define DDX_RT_LEAF_TABLE(X)                                                   \
  X(constant, Const, "const")                                                  \
  X(var, Var, "var")

// The eighteen transcendentals get the same treatment from
// DDX_UNARY_MATH_TABLE, whose descriptors already carry their own derivatives;
#define DDX_RT_UNARY_TABLE(X)                                                  \
  X(neg, Neg, "-", std::negate<>, T{-1})                                       \
  X(abs, Abs, "abs", impl::detail::abs_impl, sign(u))                          \
  X(sign, Sign, "sign", impl::detail::sign_impl, T{0})

#define DDX_RT_BINARY_TABLE(X)                                                 \
  X(add, Add, "+", std::plus<>, T{1}, T{1})                                    \
  X(mul, Mul, "*", std::multiplies<>, r, l)                                    \
  X(div, Div, "/", std::divides<>, T{1} / r, -f / r)                           \
  X(pow, Pow, "pow", impl::detail::pow_impl, r *pow(l, r - T{1}), f *log(l))   \
  X(atan2, Atan2, "atan2", impl::detail::atan2_impl,                           \
    r / hypot(l, r) / hypot(l, r), -l / hypot(l, r) / hypot(l, r))             \
  X(hypot, Hypot, "hypot", impl::detail::hypot_impl, l / f, r / f)             \
  X(max, Max, "max", impl::detail::max_impl, (T{1} + sign(l - r)) / T{2},      \
    (T{1} - sign(l - r)) / T{2})                                               \
  X(min, Min, "min", impl::detail::min_impl, (T{1} - sign(l - r)) / T{2},      \
    (T{1} + sign(l - r)) / T{2})

#define DDX_RT_OP_TABLE(X)                                                     \
  DDX_RT_LEAF_TABLE(X)                                                         \
  DDX_RT_UNARY_TABLE(X)                                                        \
  DDX_UNARY_MATH_TABLE(X)                                                      \
  DDX_RT_BINARY_TABLE(X)

enum class OpCode : std::uint8_t {
#define DDX_RT_ENUMERATOR(fn, Op, label, ...) Op,
  DDX_RT_OP_TABLE(DDX_RT_ENUMERATOR)
#undef DDX_RT_ENUMERATOR
};

inline constexpr std::size_t op_count = [] {
  std::size_t n = 0;
#define DDX_RT_COUNT(fn, Op, label, ...) ++n;
  DDX_RT_OP_TABLE(DDX_RT_COUNT)
#undef DDX_RT_COUNT
  return n;
}();

namespace detail {

struct OpInfo {
  std::string_view label;
  std::uint8_t arity;
};

// Arity is a property of which sub-table a row sits in, not a column any row
// carries, so the three groups fill one array in three passes.  Each row lands
// at its own enumerator rather than at a running index, so nothing here depends
// on the table order matching the enum's.
inline constexpr std::array<OpInfo, op_count> op_info = [] {
  std::array<OpInfo, op_count> t{};
#define DDX_RT_ROW(fn, Op, label, ...)                                         \
  t[static_cast<std::size_t>(OpCode::Op)] = {label, 0};
  DDX_RT_LEAF_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
#define DDX_RT_ROW(fn, Op, label, ...)                                         \
  t[static_cast<std::size_t>(OpCode::Op)] = {label, 1};
  DDX_RT_UNARY_TABLE(DDX_RT_ROW)
  DDX_UNARY_MATH_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
#define DDX_RT_ROW(fn, Op, label, ...)                                         \
  t[static_cast<std::size_t>(OpCode::Op)] = {label, 2};
  DDX_RT_BINARY_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
  return t;
}();

} // namespace detail

// An OpCode reaching these from outside the builder -- a deserialised graph, a
// cast byte -- need not name a row, so both answer out of range rather than
// index past the table.
[[nodiscard]] constexpr std::string_view label_of(OpCode op) noexcept {
  const auto i = static_cast<std::size_t>(op);
  return i < op_count ? detail::op_info[i].label : "?";
}

[[nodiscard]] constexpr std::uint8_t arity_of(OpCode op) noexcept {
  const auto i = static_cast<std::size_t>(op);
  return i < op_count ? detail::op_info[i].arity : std::uint8_t{0};
}

template <impl::Numeric T>
[[nodiscard]] constexpr bool is_commutative(OpCode op) noexcept {
  return (op == OpCode::Mul) ? impl::CCommutativeMultiply<T>
                             : (op == OpCode::Add || op == OpCode::Max ||
                                op == OpCode::Min || op == OpCode::Hypot);
}

[[nodiscard]] constexpr bool is_leaf(OpCode op) noexcept {
  return arity_of(op) == 0;
}

} // namespace ddx::rt
