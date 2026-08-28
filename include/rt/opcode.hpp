#pragma once

#include "ops/unary_math.hpp" // DDX_UNARY_MATH_TABLE

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>
#include <string>
#include <string_view>

// The runtime mirror of the compile-time operation set.  Rows are (factory
// spelling, enumerator, label), the shape DDX_UNARY_MATH_TABLE already has.
namespace ddx::rt {

#define DDX_RT_LEAF_TABLE(X)                                                   \
  X(constant, Const, "const")                                                  \
  X(var, Var, "var")

// The last cell is the ops/adjoints.hpp descriptor carrying the reverse-mode
// rule, which the compile-time ops call too -- the rule is written once, there.
// The eighteen transcendentals get the same treatment from
// DDX_UNARY_MATH_TABLE, whose descriptors are found by name rather than listed.
#define DDX_RT_UNARY_TABLE(X)                                                  \
  X(neg, Neg, "-", std::negate<>, NegateOpFn)                                  \
  X(abs, Abs, "abs", impl::detail::abs_impl, AbsOpFn)                          \
  X(sign, Sign, "sign", impl::detail::sign_impl, SignOpFn)

#define DDX_RT_BINARY_TABLE(X)                                                 \
  X(add, Add, "+", std::plus<>, SumOpFn)                                       \
  X(mul, Mul, "*", std::multiplies<>, MultiplyOpFn)                            \
  X(div, Div, "/", std::divides<>, DivideOpFn)                                 \
  X(pow, Pow, "pow", impl::detail::pow_impl, PowOpFn)                          \
  X(atan2, Atan2, "atan2", impl::detail::atan2_impl, Atan2OpFn)                \
  X(hypot, Hypot, "hypot", impl::detail::hypot_impl, HypotOpFn)                \
  X(max, Max, "max", impl::detail::max_impl, MaxOpFn)                          \
  X(min, Min, "min", impl::detail::min_impl, MinOpFn)

// Their own table, and so no descriptor: the derivative is zero wherever it
// exists, which is what rt/derivative.hpp's `default` already answers.
#define DDX_RT_COMPARE_TABLE(X)                                                \
  X(lt, Lt, "<", impl::detail::lt_impl)                                  \
  X(le, Le, "<=", impl::detail::le_impl)

// The one ternary: `c != 0 ? t : f`, both arms evaluated.
#define DDX_RT_TERNARY_TABLE(X)                                                \
  X(select, Select, "select", impl::detail::select_impl)

#define DDX_RT_OP_TABLE(X)                                                     \
  DDX_RT_LEAF_TABLE(X)                                                         \
  DDX_RT_UNARY_TABLE(X)                                                        \
  DDX_UNARY_MATH_TABLE(X)                                                      \
  DDX_RT_BINARY_TABLE(X)                                                       \
  DDX_RT_COMPARE_TABLE(X)                                                      \
  DDX_RT_TERNARY_TABLE(X)

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

// Arity is which sub-table a row sits in, so the three groups fill one array in
// three passes.  Each row lands at its own enumerator, so nothing depends on
// the table order matching the enum's.
inline constexpr std::array op_info = [] {
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
  DDX_RT_COMPARE_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
#define DDX_RT_ROW(fn, Op, label, ...)                                         \
  t[static_cast<std::size_t>(OpCode::Op)] = {label, 3};
  DDX_RT_TERNARY_TABLE(DDX_RT_ROW)
#undef DDX_RT_ROW
  return t;
}();

// A duplicate would make opcode_of_label() ambiguous, and the compile-time ops
// pick their enumerator by label.
static_assert([] {
  auto labels = op_info;
  std::ranges::sort(labels, {}, &OpInfo::label);
  return std::ranges::adjacent_find(labels, {}, &OpInfo::label) == labels.end();
}());

} // namespace detail

// An OpCode from outside the builder -- a deserialised graph, a cast byte --
// need not name a row, so both answer out of range.
[[nodiscard]] constexpr std::string_view label_of(OpCode op) noexcept {
  const auto i = static_cast<std::size_t>(op);
  return i < op_count ? detail::op_info[i].label : "?";
}

[[nodiscard]] consteval std::optional<OpCode>
opcode_of_label(std::string_view label) noexcept {
  const auto row =
      std::ranges::find(detail::op_info, label, &detail::OpInfo::label);
  return row == detail::op_info.end() ? std::nullopt
                                      : std::optional{static_cast<OpCode>(
                                            row - detail::op_info.begin())};
}

// The same lookup at run time, which is what a loader has: byte values are
// table-order, so appending a transcendental shifts every enumerator above it,
// and a file names its opcodes by label and remaps them on load.
[[nodiscard]] inline std::optional<OpCode>
opcode_of(std::string_view label) noexcept {
  const auto row =
      std::ranges::find(detail::op_info, label, &detail::OpInfo::label);
  return row == detail::op_info.end() ? std::nullopt
                                      : std::optional{static_cast<OpCode>(
                                            row - detail::op_info.begin())};
}

[[nodiscard]] inline std::vector<std::string> opcode_labels() {
  std::vector<std::string> out;
  out.reserve(op_count);
  for (const auto &i : detail::op_info) {
    out.emplace_back(i.label);
  }
  return out;
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
