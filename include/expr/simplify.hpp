#pragma once
#include "expr/expressions.hpp"
#include "expr/operations.hpp"
#include <compare>
#include <concepts>
#include <cstddef>
#include <string_view>
#include <tuple>

// Algebraic rewrite rules, run at build time by the operator factories in
// values.hpp, so a tree is born folded.  x*0 -> 0, 0/x -> 0 and (1/x)*x -> 1
// are not IEEE-faithful; they cancel arithmetic the derivative rules
// manufactured, not anything the user wrote.
namespace ddx::impl::detail {

template <typename Op> inline constexpr bool is_sum_op_v = false;
template <Numeric T> inline constexpr bool is_sum_op_v<SumOp<T>> = true;
template <typename Op> inline constexpr bool is_mul_op_v = false;
template <Numeric T> inline constexpr bool is_mul_op_v<MultiplyOp<T>> = true;
template <typename Op> inline constexpr bool is_div_op_v = false;
template <Numeric T> inline constexpr bool is_div_op_v<DivideOp<T>> = true;
template <typename Op> inline constexpr bool is_pow_op_v = false;
template <Numeric T> inline constexpr bool is_pow_op_v<PowOp<T>> = true;

template <typename E> inline constexpr bool is_negation_expr_v = false;
template <Numeric T, CExpression C>
inline constexpr bool is_negation_expr_v<Expression<NegateOp<T>, C>> = true;

// Q is a quotient whose denominator is exactly the tree X.  Trees are empty
// types carrying their whole structure, so type identity *is* structural
// identity -- no walk needed.
template <typename Q, typename X> inline constexpr bool is_over_v = false;
template <Numeric T, CExpression N, CExpression D, typename X>
inline constexpr bool is_over_v<Expression<DivideOp<T>, N, D>, X> =
    std::same_as<D, X>;

// Only a Lit carries its value in the type, so a runtime Constant<T> never
// folds.
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
template <COperation Op, CExpression A, CExpression B>
[[nodiscard]] constexpr auto simplify_node(const A &a, const B &b) noexcept {
  using T = typename Op::value_type;
  if constexpr (is_mul_op_v<Op>) {
    if constexpr (is_zero_v<A> || is_zero_v<B>) {
      return Lit<T, 0>{};
    } else if constexpr (is_one_v<A>) {
      return b;
    } else if constexpr (is_one_v<B>) {
      return a;
    } else if constexpr (is_over_v<A, B>) {
      // (n/a) * a -> n.  `/` is right division, so a^-1 meets a and cancels
      // whatever T is.  It fires where a 1/u a derivative manufactured meets
      // the u the product rule put back: d(u log u)/du is one such.
      return std::get<0>(a.expressions());
    } else if constexpr (is_over_v<B, A> && CCommutativeMultiply<T>) {
      // a * (n/a) is a*n*a^-1 -- only n when the factors commute.
      return std::get<0>(b.expressions());
    } else {
      return Expression<Op, A, B>{a, b};
    }
  } else if constexpr (is_sum_op_v<Op>) {
    if constexpr (is_zero_v<A>) {
      return b;
    } else if constexpr (is_zero_v<B>) {
      return a;
    } else {
      return Expression<Op, A, B>{a, b};
    }
  } else if constexpr (is_div_op_v<Op>) {
    if constexpr (is_zero_v<A>) {
      return Lit<T, 0>{};
    } else if constexpr (is_one_v<B>) {
      return a;
    } else {
      return Expression<Op, A, B>{a, b};
    }
  } else if constexpr (is_pow_op_v<Op>) {
    if constexpr (is_zero_v<B>) {
      return Lit<T, 1>{};
    } else if constexpr (is_one_v<B>) {
      return a;
    } else {
      return Expression<Op, A, B>{a, b};
    }
  } else {
    return Expression<Op, A, B>{a, b};
  }
}

// -(-x) is x; literal cases are folded by the caller in values.hpp.
template <COperation Op, CExpression A>
[[nodiscard]] constexpr auto simplify_mono(const A &a) noexcept {
  if constexpr (std::same_as<Op, NegateOp<typename Op::value_type>> &&
                is_negation_expr_v<A>) {
    return std::get<0>(a.expressions());
  } else {
    return MonoExpression<Op, A>{a};
  }
}

// Canonical ordering: fixing the operand order of a commutative node makes x+y
// and y+x the same type.  Sums always reorder, products only when the scalar
// declares CCommutativeMultiply.  Swaps operands of one node, never
// reassociates, and runs where Equation is built rather than in the operators.

template <typename E> inline constexpr bool is_variable_expr_v = false;
template <Numeric T, CFixedString auto S, bool F>
inline constexpr bool is_variable_expr_v<Variable<T, S, F>> = true;

// Leaves before branches, symbols by name, branches by size then op.  Ties
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
