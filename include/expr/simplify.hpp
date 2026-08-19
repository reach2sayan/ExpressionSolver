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
// derivative(): Op::derivative builds its result out of the ordinary operators,
// and the 0s and 1s that leaves() manufacture flood it.  Without these rules
// d(x*y)/dx is `1*y + x*0`, seven nodes and seven node_cache_t slots for what is
// one symbol lookup; with them it is `y`.
//
// x*0 -> 0 and 0/x -> 0 are not IEEE-faithful: 0*Inf is NaN, not 0.  Every AD
// library makes the same trade, because the zeros here come from d(const)/dx
// rather than from anything the user wrote.
namespace diff::detail {

// Which op a node carries.  There is no op-kind enum -- Notation and precedence
// are shared by ops with quite different algebra -- so the rule table matches
// the class templates themselves.
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

// Only a Lit carries its value in the type, and both spellings of it (the
// T-valued NTTP and the int one) expose `value`.  A stored Constant<T> holds a
// number that exists only at run time, so it matches neither and x * PC(0.0)
// correctly does not fold.
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
// they are the only ones available for every Numeric T, so folding always lands
// on the same type whatever the scalar is.
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
// Addition only.  Putting the operands of a sum in a fixed order makes x+y and
// y+x the same *type*, which is what a later CSE/DAG pass needs: for a
// stateless tree, type identity already is structural equality, so
// std::is_same_v is the value numbering.
//
// A product is reordered only when its scalar has declared that it may be.
// value_type is only required to be Numeric -- CArithmetic or CFieldLike -- and
// CFieldLike asks for +, -, *, / and unary minus without anywhere requiring
// a*b == b*a, so Numeric admits matrices, quaternions, any non-commutative
// algebra.  Commuting a product is a property of the scalar plugged in, not of
// the expression type, so it is asked rather than assumed:
// CCommutativeMultiply (expressions.hpp) is opt-in, and double, Dual,
// TaylorDual and VectorDual all opt in.  Addition needs no such guard -- a
// ring's addition commutes by definition.
//
// The library assumes exactly the algebra of a (possibly non-commutative) ring:
//
//   (a+b)+c == a+(b+c)   yes        a + b == b + a   yes
//   (a*b)*c == a*(b*c)   yes        a * b == b * a   only when declared
//
// Associativity is a separate question from commutativity and is assumed for
// both operators -- though reassociating moves floating-point results, and this
// pass never reorders across nodes anyway; it only swaps the two operands of
// one node.
//
// Ordering runs where Equation is built rather than in the operators, because
// silently reordering what a user typed is a surprise; the trees Equation
// stores are the ones that need it.
// ---------------------------------------------------------------------------

template <typename E> inline constexpr bool is_variable_expr_v = false;
template <Numeric T, CFixedString auto S, bool F>
inline constexpr bool is_variable_expr_v<Variable<T, S, F>> = true;

// Leaves sort ahead of branches so 2+x never comes out as x+2, symbols sort by
// name, and branches by size then by op.  Ties compare equal, which leaves the
// operands where they were: the order is stable, and a tie only ever costs a
// later CSE pass a share it could have had, never correctness.
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

// Bottom-up, one pass.  Every rule returns an already-canonical child, a Lit,
// or a node whose children are canonical and whose own rule has just run, so no
// rule leaves a fresh redex behind and there is nothing to iterate.
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
