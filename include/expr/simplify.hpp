#pragma once
#include "expr/expressions.hpp"
#include "expr/operations.hpp"
#include <compare>
#include <cstddef>
#include <string_view>
#include <tuple>

// The algebraic rewrite rules, in one place.
//
// They run at build time -- the operator factories in values.hpp call
// simplify_node() instead of naming Expression<> directly -- so a tree is born
// folded and the garbage is never instantiated.  That matters most for
// derivative(), whose results are flooded with the 0s and 1s the leaves
// manufacture: without these rules d(x*y)/dx is `1*y + x*0`, seven nodes for one
// symbol lookup; with them it is `y`.
//
// x*0 -> 0 and 0/x -> 0 are not IEEE-faithful (0*Inf is NaN, not 0).  Every AD
// library makes the same trade: these zeros come from d(const)/dx, not from
// anything the user wrote.
namespace diff::detail {

// Which op a node carries.  There is no op-kind enum (Notation and precedence
// are shared by ops with different algebra), so the table matches the class
// templates themselves.
template <typename Op> inline constexpr bool is_sum_op_v = false;
template <Numeric T> inline constexpr bool is_sum_op_v<SumOp<T>> = true;
template <typename Op> inline constexpr bool is_mul_op_v = false;
template <Numeric T> inline constexpr bool is_mul_op_v<MultiplyOp<T>> = true;
template <typename Op> inline constexpr bool is_div_op_v = false;
template <Numeric T> inline constexpr bool is_div_op_v<DivideOp<T>> = true;
template <typename Op> inline constexpr bool is_pow_op_v = false;
template <Numeric T> inline constexpr bool is_pow_op_v<PowOp<T>> = true;

// A node that is already a negation, so that -(-x) can collapse.
template <typename E> inline constexpr bool is_negation_expr_v = false;
template <Numeric T, CExpression C>
inline constexpr bool is_negation_expr_v<Expression<NegateOp<T>, C>> = true;

// Only a Lit carries its value in the type; both spellings expose `value`.  A
// stored Constant<T> holds a number that exists only at run time, so it matches
// neither and x * PC(0.0) correctly does not fold.
template <CExpression E> consteval bool lit_equals(int n) {
  if constexpr (CLit<E>) {
    return E::value == typename E::value_type(n);
  } else {
    return false;
  }
}
template <CExpression E> inline constexpr bool is_zero_v = lit_equals<E>(0);
template <CExpression E> inline constexpr bool is_one_v = lit_equals<E>(1);

// Lit<T, 0> and Lit<T, 1> -- the int spelling -- are the canonical zero and one:
// the only ones available for every Numeric T, so folding always lands on the
// same type whatever the scalar is.
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

// Unary negation.  -(-x) is x; the literal cases are folded by the caller in
// values.hpp, which can spell the negated NTTP directly.
template <COperation Op, CExpression A>
[[nodiscard]] constexpr auto simplify_mono(const A &a) noexcept {
  if constexpr (std::same_as<Op, NegateOp<typename Op::value_type>> &&
                is_negation_expr_v<A>) {
    return std::get<0>(a.expressions());
  } else {
    return MonoExpression<Op, A>{a};
  }
}

// ---------------------------------------------------------------------------
// Canonical ordering
//
// Putting the operands of a commutative node in a fixed order makes x+y and y+x
// the same *type*: for a stateless tree, type identity is structural equality.
//
// The library assumes exactly the algebra of a (possibly non-commutative) ring:
//
//   (a+b)+c == a+(b+c)   yes        a + b == b + a   yes
//   (a*b)*c == a*(b*c)   yes        a * b == b * a   only when declared
//
// So a sum is always reordered, a product only when its scalar has declared
// CCommutativeMultiply (expressions.hpp) -- Numeric admits matrices and
// quaternions, and commuting is a property of the scalar, not the tree.
//
// This pass only ever swaps the two operands of one node; it never reassociates
// across nodes.  It runs where Equation is built rather than in the operators,
// because silently reordering what a user typed is a surprise.
// ---------------------------------------------------------------------------

template <typename E> inline constexpr bool is_variable_expr_v = false;
template <Numeric T, CFixedString auto S, bool F>
inline constexpr bool is_variable_expr_v<Variable<T, S, F>> = true;

// Leaves sort ahead of branches so 2+x never comes out as x+2, symbols by name,
// branches by size then by op.  Ties compare equal, leaving the operands where
// they were -- the order is stable, and a tie never costs correctness.
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

// Bottom-up, one pass.  Every rule returns an already-canonical child, a Lit, or
// a node whose children are canonical and whose own rule has just run, so no rule
// leaves a fresh redex behind.
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

} // namespace diff::detail
