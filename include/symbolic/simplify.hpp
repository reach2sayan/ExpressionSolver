#pragma once
#include "ops/algebra.hpp"
#include "ops/operations.hpp"
#include "symbolic/expressions.hpp"
#include <compare>
#include <concepts>
#include <cstddef>
#include <optional>
#include <ranges>
#include <string_view>
#include <tuple>
#include <utility>

// The type-level half of the simplifier: ops/algebra.hpp's predicates answered
// with type traits, run by the operator factories in values.hpp so a tree is
// born folded.  rt/builder.hpp answers the same predicates by node id.
namespace ddx::impl::detail {

template <typename Op> inline constexpr bool is_sum_op_v = false;
template <Numeric T> inline constexpr bool is_sum_op_v<SumOp<T>> = true;
template <typename Op> inline constexpr bool is_mul_op_v = false;
template <Numeric T> inline constexpr bool is_mul_op_v<MultiplyOp<T>> = true;
template <typename Op> inline constexpr bool is_div_op_v = false;
template <Numeric T> inline constexpr bool is_div_op_v<DivideOp<T>> = true;
template <typename Op> inline constexpr bool is_pow_op_v = false;
template <Numeric T> inline constexpr bool is_pow_op_v<PowOp<T>> = true;

// Q is a quotient whose denominator is exactly the tree X.
template <typename Q, typename X> inline constexpr bool is_over_v = false;
template <Numeric T, CExpression N, CExpression D, typename X>
inline constexpr bool is_over_v<Expression<DivideOp<T>, N, D>, X> =
    std::same_as<D, X>;

// Only a Lit carries its value in the type; a Constant<T> never folds.
template <CExpression E> consteval bool lit_equals(int n) {
  if constexpr (CLit<E>) {
    return E::value == typename E::value_type(n);
  } else {
    return false;
  }
}
template <CExpression E> inline constexpr bool is_zero_v = lit_equals<E>(0);
template <CExpression E> inline constexpr bool is_one_v = lit_equals<E>(1);

// Lit<T, 0> / Lit<T, 1> are the canonical zero and one for every Numeric T.
// Which RuleOp an operation is, or none for one carrying no identities.
template <COperation Op>
[[nodiscard]] consteval std::optional<algebra::RuleOp> rule_op_of() noexcept {
  if constexpr (is_mul_op_v<Op>) {
    return algebra::RuleOp::Mul;
  } else if constexpr (is_sum_op_v<Op>) {
    return algebra::RuleOp::Add;
  } else if constexpr (is_div_op_v<Op>) {
    return algebra::RuleOp::Div;
  } else if constexpr (is_pow_op_v<Op>) {
    return algebra::RuleOp::Pow;
  } else if constexpr (std::same_as<Op, NegateOp<typename Op::value_type>>) {
    return algebra::RuleOp::Neg;
  } else {
    return std::nullopt;
  }
}

// The predicates, answered structurally: a tree is an empty type carrying its
// whole shape, so type identity *is* structural identity -- no walk needed.
template <COperation Op, CExpression A, CExpression B>
[[nodiscard]] constexpr bool holds(algebra::Pred p) noexcept {
  switch (p) {
  case algebra::Pred::ZeroA:
    return is_zero_v<A>;
  case algebra::Pred::ZeroB:
    return is_zero_v<B>;
  case algebra::Pred::OneA:
    return is_one_v<A>;
  case algebra::Pred::OneB:
    return is_one_v<B>;
  case algebra::Pred::Same:
    return std::same_as<A, B>;
  case algebra::Pred::AOverB:
    return is_over_v<A, B>;
  case algebra::Pred::BOverA:
    return is_over_v<B, A>;
  case algebra::Pred::NegatedA:
    return is_negation_expr_v<A>;
  }
  std::unreachable();
}

// First match wins, as the table is ordered.
template <COperation Op, CExpression A, CExpression B>
[[nodiscard]] consteval std::optional<algebra::Rule> match_rule() noexcept {
  constexpr auto kind = rule_op_of<Op>();
  if constexpr (!kind.has_value()) {
    return std::nullopt;
  } else {
    const auto r = std::ranges::find_if(
        algebra::kRules, [kind](const algebra::Rule &rule) {
          return rule.op == *kind && !algebra::is_unary(rule) &&
                 (!rule.needs_commutative_multiply ||
                  CCommutativeMultiply<typename Op::value_type>) &&
                 holds<Op, A, B>(rule.when);
        });
    return r == std::ranges::cend(algebra::kRules)
               ? std::nullopt
               : std::optional<algebra::Rule>{*r};
  }
}

template <COperation Op, CExpression A, CExpression B>
[[nodiscard]] constexpr auto simplify_node(const A &a, const B &b) noexcept {
  using T = typename Op::value_type;
  constexpr auto matched = match_rule<Op, A, B>();
  if constexpr (!matched.has_value()) {
    return Expression<Op, A, B>{a, b};
  } else if constexpr (matched->then == algebra::Take::LitZero) {
    return Lit<T, 0>{};
  } else if constexpr (matched->then == algebra::Take::LitOne) {
    return Lit<T, 1>{};
  } else if constexpr (matched->then == algebra::Take::OperandA) {
    return a;
  } else if constexpr (matched->then == algebra::Take::OperandB) {
    return b;
  } else if constexpr (matched->then == algebra::Take::NumeratorOfA) {
    return std::get<0>(a.expressions());
  } else {
    static_assert(matched->then == algebra::Take::NumeratorOfB);
    return std::get<0>(b.expressions());
  }
}

// -(-x) is x; literal cases are folded by the caller in values.hpp.
template <COperation Op, CExpression A>
[[nodiscard]] constexpr auto simplify_mono(const A &a) noexcept {
  constexpr auto kind = rule_op_of<Op>();
  constexpr bool negates = kind.has_value() && *kind == algebra::RuleOp::Neg &&
                           is_negation_expr_v<A>;
  if constexpr (negates) {
    return std::get<0>(a.expressions());
  } else {
    return MonoExpression<Op, A>{a};
  }
}

// Fixing the operand order of a commutative node makes x+y and y+x the same
// type.  Sums always reorder, products only under CCommutativeMultiply.  Swaps
// operands of one node, never reassociates.

template <typename E> inline constexpr bool is_variable_expr_v = false;
template <Numeric T, CFixedString auto S, bool F>
inline constexpr bool is_variable_expr_v<Variable<T, S, F>> = true;

// Leaves before branches, symbols by name, branches by size then op; ties
// compare equal, so the order is stable.
struct order_key {
  int kind{};
  std::size_t nodes{};
  std::string_view name{};
  friend constexpr auto operator<=>(const order_key &,
                                    const order_key &) = default;
};

template <CExpression E> consteval order_key key_of() {
  if constexpr (is_variable_expr_v<E>) {
    return {1, 1, E::label.view()};
  } else if constexpr (CExpressionNode<E>) {
    return {2, node_count_v<E>, E::op_type::label};
  } else {
    return {0, 1, {}};
  }
}

template <COperation Op, CExpression A>
[[nodiscard]] constexpr auto ordered_node(const A &a) noexcept {
  return simplify_mono<Op>(a);
}

template <COperation Op, CExpression A, CExpression B>
[[nodiscard]] constexpr auto ordered_node(const A &a, const B &b) noexcept {
  if constexpr ((is_sum_op_v<Op> ||
                 (is_mul_op_v<Op> &&
                  CCommutativeMultiply<typename Op::value_type>)) &&
                key_of<B>() < key_of<A>()) {
    return simplify_node<Op>(b, a);
  } else {
    return simplify_node<Op>(a, b);
  }
}

// Bottom-up, one pass: no rule leaves a fresh redex behind.
template <CExpression E>
[[nodiscard]] constexpr auto canonicalise(const E &e) noexcept {
  if constexpr (CExpressionNode<E>) {
    return std::apply(
        [](const auto &...kids) noexcept {
          return ordered_node<typename E::op_type>(canonicalise(kids)...);
        },
        e.expressions());
  } else {
    return e;
  }
}

} // namespace ddx::impl::detail
