#pragma once

#include <concepts>
#include <ostream>
#include <tuple>
#include <type_traits>

namespace diff {

/// Numeric: any scalar type that supports arithmetic operations.
template <typename T>
concept Numeric = std::is_arithmetic_v<T> || requires(T a, T b) {
  T{};
  { a + b } -> std::convertible_to<T>;
  { a - b } -> std::convertible_to<T>;
  { a * b } -> std::convertible_to<T>;
  { a / b } -> std::convertible_to<T>;
  { -a } -> std::convertible_to<T>;
};

/// AnOp: any operation struct that carries a Numeric value_type.
template <typename O>
concept AnOp =
    requires { typename O::value_type; } && Numeric<typename O::value_type>;

// ===========================================================================
// Forward declarations
// ===========================================================================
template <AnOp Op, typename LHS, typename RHS> class Expression;
template <AnOp Op, typename Exp> class MonoExpression;
template <Numeric T> class Constant;
template <Numeric T, char> class Variable;
// ===========================================================================
// Tag trait: true for any first-class expression node.
// ===========================================================================
template <typename T> struct is_expression_type : std::false_type {};
template <Numeric T> struct is_expression_type<Constant<T>> : std::true_type {};

template <Numeric T, char C>
struct is_expression_type<Variable<T, C>> : std::true_type {};

template <AnOp Op, typename LHS, typename RHS>
struct is_expression_type<Expression<Op, LHS, RHS>> : std::true_type {};

template <AnOp Op, typename Exp>
struct is_expression_type<MonoExpression<Op, Exp>> : std::true_type {};

template <typename T>
concept CExpression = is_expression_type<std::remove_cvref_t<T>>::value;

// ===========================================================================
// EvalResult<T>
// ===========================================================================
template <Numeric T> struct EvalResult {
  using value_type = T;
  T value;
  [[nodiscard]] constexpr T eval() const { return value; }
  constexpr operator T() const { return value; }
};

template <Numeric T>
struct is_expression_type<EvalResult<T>> : std::true_type {};

// ===========================================================================
// BaseExpression
// ===========================================================================
template <AnOp Op> struct BaseExpression {
  using value_type = typename Op::value_type;
};

// ===========================================================================
// MonoExpression — unary expression node.
// ===========================================================================
template <AnOp Op, typename Exp>
class MonoExpression : public BaseExpression<Op> {
  Exp expression;
  friend std::ostream &operator<<(std::ostream &out, const MonoExpression &e) {
    out << e.expression;
    return out;
  }

public:
  [[nodiscard]] constexpr const auto &expressions() const { return expression; }
  using lhs_type = Exp;
  using value_type = typename BaseExpression<Op>::value_type;
  constexpr MonoExpression(Exp expr) : expression{std::move(expr)} {}

  template <std::size_t I> [[nodiscard]] constexpr auto get() const {
    static_assert(I < 2);
    if constexpr (requires { std::tuple_size<value_type>::value; }) {
      return eval().template get<I>();
    } else if constexpr (I == 0) {
      return eval();
    } else {
      return static_cast<value_type>(derivative());
    }
  }

  [[nodiscard]] constexpr auto derivative() const {
    return Op::derivative(expression);
  }
  constexpr operator value_type() const { return eval(); }
  [[nodiscard]] constexpr auto eval() const { return Op::eval(expression); }

  template <typename Syms, std::size_t N>
  [[nodiscard]] constexpr auto
  eval_seeded(const std::array<value_type, N> &vals) const {
    return Op::eval(
        EvalResult<value_type>{expression.template eval_seeded<Syms>(vals)});
  }

  constexpr void update(const auto &symbols, const auto &updates) {
    expression.update(symbols, updates);
  }
  constexpr void collect(const auto &symbols, auto &out) const {
    expression.collect(symbols, out);
  }
  constexpr void backward(const auto &syms, value_type adj, auto &grads) const {
    Op::backward(expression, adj, syms, grads);
  }
};

// ===========================================================================
// Expression — binary expression node.
// ===========================================================================
template <AnOp Op, typename LHS, typename RHS>
class Expression : public BaseExpression<Op> {
  std::pair<LHS, RHS> inner_expressions;
  friend std::ostream &operator<<(std::ostream &out, const Expression &e) {
    out << '(';
    std::apply([&out](const auto &...e) { Op::print(out, e...); },
               e.inner_expressions);
    out << ')';
    return out;
  }

public:
  using op_type = Op;
  using lhs_type = LHS;
  using rhs_type = RHS;
  using value_type = typename BaseExpression<Op>::value_type;
  [[nodiscard]] constexpr const auto &expressions() const {
    return inner_expressions;
  }
  constexpr Expression(LHS lhs, RHS rhs)
      : inner_expressions({std::move(lhs), std::move(rhs)}) {}

  template <std::size_t I> [[nodiscard]] constexpr auto get() const {
    static_assert(I < 2);
    if constexpr (requires { std::tuple_size<value_type>::value; }) {
      return eval().template get<I>();
    } else if constexpr (I == 0) {
      return eval();
    } else {
      return static_cast<value_type>(derivative());
    }
  }

  [[nodiscard]] constexpr auto eval() const {
    return std::apply([](const auto &...e) { return Op::eval(e...); },
                      inner_expressions);
  }
  constexpr operator value_type() const { return eval(); }
  [[nodiscard]] constexpr auto derivative() const {
    return std::apply([](const auto &...e) { return Op::derivative(e...); },
                      inner_expressions);
  }

  template <typename Syms, std::size_t N>
  [[nodiscard]] constexpr auto
  eval_seeded(const std::array<value_type, N> &vals) const {
    return Op::eval(
        EvalResult<value_type>{
            inner_expressions.first.template eval_seeded<Syms>(vals)},
        EvalResult<value_type>{
            inner_expressions.second.template eval_seeded<Syms>(vals)});
  }

  constexpr void update(const auto &symbols, const auto &updates) {
    inner_expressions.first.update(symbols, updates);
    inner_expressions.second.update(symbols, updates);
  }
  constexpr void collect(const auto &symbols, auto &out) const {
    inner_expressions.first.collect(symbols, out);
    inner_expressions.second.collect(symbols, out);
  }
  constexpr void backward(const auto &syms, value_type adj, auto &grads) const {
    std::apply([&](const auto &...e) { Op::backward(e..., adj, syms, grads); },
               inner_expressions);
  }
};

namespace detail {
template <typename V, std::size_t I, typename = void>
struct expression_element {
  using type = V;
};

template <typename V, std::size_t I>
struct expression_element<V, I,
                          std::void_t<typename std::tuple_element_t<I, V>>> {
  using type = std::tuple_element_t<I, V>;
};
} // namespace detail

} // namespace diff

namespace std {
template <diff::AnOp Op, typename LHS, typename RHS>
struct tuple_size<diff::Expression<Op, LHS, RHS>>
    : integral_constant<size_t, 2> {};

template <size_t I, diff::AnOp Op, typename LHS, typename RHS>
struct tuple_element<I, diff::Expression<Op, LHS, RHS>> {
  using type = typename diff::detail::expression_element<
      typename diff::Expression<Op, LHS, RHS>::value_type, I>::type;
};

template <diff::AnOp Op, typename Exp>
struct tuple_size<diff::MonoExpression<Op, Exp>>
    : integral_constant<size_t, 2> {};

template <size_t I, diff::AnOp Op, typename Exp>
struct tuple_element<I, diff::MonoExpression<Op, Exp>> {
  using type = typename diff::detail::expression_element<
      typename diff::MonoExpression<Op, Exp>::value_type, I>::type;
};
} // namespace std
