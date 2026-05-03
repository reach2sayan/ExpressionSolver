#pragma once
#include "values.hpp"
#include <array>
#include <boost/mp11/algorithm.hpp>
#include <type_traits>

namespace diff {

namespace mp = boost::mp11;

// Direct index_sequence fold — no Boost mp_for_each intermediate lambda.
template <std::size_t N, class F> constexpr void static_for(F &&f) {
  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    (std::forward<F>(f).template operator()<Is>(), ...);
  }(std::make_index_sequence<N>{});
}

template <typename T> constexpr static bool is_const = false;
template <typename T> constexpr static bool is_const<Constant<T>> = true;

template <typename T, char symbol = 'X'> struct make_all_constant {
  using type = T;
};
template <typename T, char symbol>
struct make_all_constant<Variable<T, symbol>> {
  using type = Constant<T>;
};
template <typename Op, typename... TExpressions>
struct make_all_constant<Expression<Op, TExpressions...>> {
  using type =
      Expression<Op, typename make_all_constant<TExpressions>::type...>;
};

template <typename TExpression>
using as_const_expression = make_all_constant<
    Expression<typename TExpression::op_type, typename TExpression::lhs_type,
               typename TExpression::rhs_type>>::type;

template <char symbol, typename Expr> struct replace_matching_variable_as_const;
template <char symbol, typename T> struct replace_matching_variable_as_const {
  using type = T;
};
template <char symbol, typename T>
struct replace_matching_variable_as_const<symbol, Variable<T, symbol>> {
  using type = Constant<T>;
};
template <char symbol, typename Op, typename... TExpressions>
struct replace_matching_variable_as_const<symbol,
                                          Expression<Op, TExpressions...>> {
  using type = Expression<Op, typename replace_matching_variable_as_const<
                                  symbol, TExpressions>::type...>;
};

template <char symbol, typename TExpression>
using replace_matching_variable_as_const_t =
    typename replace_matching_variable_as_const<symbol, TExpression>::type;

template <char symbol, typename T>
constexpr auto make_const_variable(const Variable<T, symbol> &var) {
  return Constant<T>(var);
}

template <char symbol, typename T, char othersymbol>
  requires(symbol != othersymbol)
constexpr auto make_const_variable(const Variable<T, othersymbol> &var)
    -> Variable<T, othersymbol> {
  return var;
}

template <char symbol, typename T>
constexpr auto make_const_variable(const Constant<T> &c) {
  return c;
}

template <char symbol, typename Op, typename LHS, typename RHS>
constexpr auto make_const_variable(const Expression<Op, LHS, RHS> &expr)
    -> Expression<Op, replace_matching_variable_as_const_t<symbol, LHS>,
                  replace_matching_variable_as_const_t<symbol, RHS>> {
  auto [lexpr, rexpr] = expr.expressions();
  return {make_const_variable<symbol>(std::move(lexpr)),
          make_const_variable<symbol>(std::move(rexpr))};
}

template <char symbol, typename Op, typename LHS>
constexpr auto make_const_variable(const MonoExpression<Op, LHS> &expr)
    -> MonoExpression<Op, replace_matching_variable_as_const_t<symbol, LHS>> {
  return {make_const_variable<symbol>(expr.expressions())};
}

template <typename T, char C, std::size_t N>
constexpr void make_labels_array(const Variable<T, C> &,
                                 std::array<char, N> &out, std::size_t &index) {
  out[index++] = C;
}

template <typename T, std::size_t N>
constexpr void make_labels_array(const Constant<T> &, std::array<char, N> &,
                                 std::size_t &) {}


template <typename Op, typename LHS, typename RHS, std::size_t N>
constexpr void make_labels_array(const Expression<Op, LHS, RHS> &expr,
                                 std::array<char, N> &out, std::size_t &index) {
  make_labels_array(expr.expressions().first, out, index);
  make_labels_array(expr.expressions().second, out, index);
}

template <char symbol, typename Expr> struct constify_unmatched_var;

template <char symbol, typename T>
struct constify_unmatched_var<symbol, Variable<T, symbol>> {
  using type = Variable<T, symbol>;
};
template <char symbol, typename T, char othersymbol>
struct constify_unmatched_var<symbol, Variable<T, othersymbol>> {
  using type = Constant<T>;
};
template <char symbol, typename T>
struct constify_unmatched_var<symbol, Constant<T>> {
  using type = Constant<T>;
};
template <char symbol, typename Op, typename LHS, typename RHS>
struct constify_unmatched_var<symbol, Expression<Op, LHS, RHS>> {
  using type =
      Expression<Op, typename constify_unmatched_var<symbol, LHS>::type,
                 typename constify_unmatched_var<symbol, RHS>::type>;
};
template <char symbol, typename Op, typename Expr>
struct constify_unmatched_var<symbol, MonoExpression<Op, Expr>> {
  using type =
      MonoExpression<Op, typename constify_unmatched_var<symbol, Expr>::type>;
};

template <char symbol, typename Expr>
using constify_unmatched_var_t =
    typename constify_unmatched_var<symbol, Expr>::type;

template <char symbol, typename T>
constexpr auto make_all_constant_except(const Variable<T, symbol> &v) {
  return v;
}

template <char symbol, typename T, char othersymbol>
  requires(symbol != othersymbol)
constexpr auto make_all_constant_except(const Variable<T, othersymbol> &var)
    -> Constant<T> {
  return Constant<T>{var};
}


template <char Symbol, typename T>
constexpr auto make_all_constant_except(const Constant<T> &c) {
  return c;
}

template <char symbol, typename Op, typename LHS, typename RHS>
constexpr auto make_all_constant_except(const Expression<Op, LHS, RHS> &expr)
    -> constify_unmatched_var_t<symbol, Expression<Op, LHS, RHS>> {
  auto new_lhs = make_all_constant_except<symbol>(expr.expressions().first);
  auto new_rhs = make_all_constant_except<symbol>(expr.expressions().second);
  return {new_lhs, new_rhs};
}

template <char symbol, typename Op, typename Expr>
constexpr auto make_all_constant_except(const MonoExpression<Op, Expr> &expr)
    -> constify_unmatched_var_t<symbol, MonoExpression<Op, Expr>> {
  return make_all_constant_except<symbol>(expr.expressions());
}

// ===========================================================================
// Type-list metaprogramming via Boost.MP11
// ===========================================================================

template <typename A, typename B>
using ic_less = mp::mp_bool<(A::value < B::value)>;

template <typename List> using sort_tuple_t = mp::mp_sort<List, ic_less>;

template <typename List>
using unique_tuple_t = mp::mp_unique<sort_tuple_t<List>>;

template <typename... Lists>
using tuple_union_t = unique_tuple_t<mp::mp_append<Lists...>>;

// ===========================================================================
// Extract the set of Variable symbols from an expression type.
// ===========================================================================

template <typename T>
auto extract_variable_symbols_impl(std::type_identity<T>) -> mp::mp_list<>;

template <typename T, char symbol>
auto extract_variable_symbols_impl(std::type_identity<Variable<T, symbol>>)
    -> mp::mp_list<std::integral_constant<char, symbol>>;

template <typename T>
using extract_variable_symbols_t =
    decltype(extract_variable_symbols_impl(std::type_identity<T>{}));

template <typename T> struct extract_symbols_from_expr {
  using type = extract_variable_symbols_t<T>;
};

template <typename Op, typename... Exprs>
struct extract_symbols_from_expr<Expression<Op, Exprs...>> {
  static_assert(sizeof...(Exprs) > 0);
  using type =
      tuple_union_t<typename extract_symbols_from_expr<Exprs>::type...>;
};

template <typename Op, typename Expr>
struct extract_symbols_from_expr<MonoExpression<Op, Expr>> {
  using type = typename extract_symbols_from_expr<Expr>::type;
};

template <std::size_t N> consteval auto idx() noexcept {
  return std::integral_constant<std::size_t, N>{};
}

template <size_t value> struct idx_t : std::integral_constant<size_t, value> {};

} // namespace diff

#define IDX(value)                                                             \
  diff::idx_t<value> {}
