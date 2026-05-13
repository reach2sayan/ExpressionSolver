#pragma once
#include "values.hpp"
#include <array>
#include <boost/mp11/algorithm.hpp>
#include <type_traits>

namespace diff {

namespace mp = boost::mp11;

// Direct index_sequence fold — no Boost mp_for_each intermediate lambda.
template <std::size_t N, class F> constexpr void static_for(F &&f) noexcept {
  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    (std::forward<F>(f).template operator()<Is>(), ...);
  }(std::make_index_sequence<N>{});
}

template <typename T> constexpr static bool is_const = false;
template <typename T> constexpr static bool is_const<Constant<T>> = true;

template <typename T> inline constexpr bool is_variable_v = false;
template <typename T, char C>
inline constexpr bool is_variable_v<Variable<T, C>> = true;

template <typename T> inline constexpr char variable_symbol_v = '\0';
template <typename T, char C>
inline constexpr char variable_symbol_v<Variable<T, C>> = C;

template <typename T> inline constexpr bool is_mono_expression_v = false;
template <typename Op, typename E>
inline constexpr bool is_mono_expression_v<MonoExpression<Op, E>> = true;

template <typename T> inline constexpr bool is_binary_expression_v = false;
template <typename Op, typename L, typename R>
inline constexpr bool is_binary_expression_v<Expression<Op, L, R>> = true;

template <typename T> consteval auto make_all_constant_impl() {
  if constexpr (is_variable_v<T>) {
    return std::type_identity<Constant<typename T::value_type>>{};
  } else if constexpr (is_binary_expression_v<T>) {
    using Op  = typename T::op_type;
    using L   = typename T::lhs_type;
    using R   = typename T::rhs_type;
    using NewL = typename decltype(make_all_constant_impl<L>())::type;
    using NewR = typename decltype(make_all_constant_impl<R>())::type;
    return std::type_identity<Expression<Op, NewL, NewR>>{};
  } else if constexpr (is_mono_expression_v<T>) {
    using Op  = typename T::op_type;
    using E   = typename T::lhs_type;
    using NewE = typename decltype(make_all_constant_impl<E>())::type;
    return std::type_identity<MonoExpression<Op, NewE>>{};
  } else {
    return std::type_identity<T>{};
  }
}

template <typename T>
using make_all_constant_t = typename decltype(make_all_constant_impl<T>())::type;

template <typename TExpression>
using as_const_expression = make_all_constant_t<
    Expression<typename TExpression::op_type, typename TExpression::lhs_type,
               typename TExpression::rhs_type>>;

template <char symbol, typename T> consteval auto replace_matching_var_impl() {
  if constexpr (is_variable_v<T> && variable_symbol_v<T> == symbol) {
    return std::type_identity<Constant<typename T::value_type>>{};
  } else if constexpr (is_binary_expression_v<T>) {
    using L = typename T::lhs_type;
    using R = typename T::rhs_type;
    using Op = typename T::op_type;
    using NewL =
        typename decltype(replace_matching_var_impl<symbol, L>())::type;
    using NewR =
        typename decltype(replace_matching_var_impl<symbol, R>())::type;
    return std::type_identity<Expression<Op, NewL, NewR>>{};
  } else if constexpr (is_mono_expression_v<T>) {
    using E = typename T::lhs_type;
    using Op = typename T::op_type;
    using NewE =
        typename decltype(replace_matching_var_impl<symbol, E>())::type;
    return std::type_identity<MonoExpression<Op, NewE>>{};
  } else {
    return std::type_identity<T>{};
  }
}

template <char symbol, typename T>
using replace_matching_variable_as_const_t =
    typename decltype(replace_matching_var_impl<symbol, T>())::type;

template <char symbol, typename T>
constexpr auto make_const_variable(const Variable<T, symbol> &var) noexcept {
  return Constant<T>(var);
}

template <char symbol, typename T, char othersymbol>
  requires(symbol != othersymbol)
constexpr auto make_const_variable(const Variable<T, othersymbol> &var) noexcept
    -> Variable<T, othersymbol> {
  return var;
}

template <char symbol, typename T>
constexpr auto make_const_variable(const Constant<T> &c) noexcept {
  return c;
}

template <char symbol, typename Op, typename LHS, typename RHS>
constexpr auto make_const_variable(const Expression<Op, LHS, RHS> &expr) noexcept
    -> Expression<Op, replace_matching_variable_as_const_t<symbol, LHS>,
                  replace_matching_variable_as_const_t<symbol, RHS>> {
  auto [lexpr, rexpr] = expr.expressions();
  return {make_const_variable<symbol>(std::move(lexpr)),
          make_const_variable<symbol>(std::move(rexpr))};
}

template <char symbol, typename Op, typename LHS>
constexpr auto make_const_variable(const MonoExpression<Op, LHS> &expr) noexcept
    -> MonoExpression<Op, replace_matching_variable_as_const_t<symbol, LHS>> {
  return {make_const_variable<symbol>(expr.expressions())};
}

template <typename T, char C, std::size_t N>
consteval void make_labels_array(const Variable<T, C> &,
                                 std::array<char, N> &out, std::size_t &index) {
  out[index++] = C;
}

template <typename T, std::size_t N>
consteval void make_labels_array(const Constant<T> &, std::array<char, N> &,
                                 std::size_t &) {}

template <typename Op, typename LHS, typename RHS, std::size_t N>
consteval void make_labels_array(const Expression<Op, LHS, RHS> &expr,
                                 std::array<char, N> &out, std::size_t &index) {
  make_labels_array(expr.expressions().first, out, index);
  make_labels_array(expr.expressions().second, out, index);
}

template <char symbol, typename Expr>
consteval auto constify_unmatched_var_impl() {
  if constexpr (is_constant_v<Expr>) {
    return std::type_identity<Expr>{};
  } else if constexpr (is_variable_v<Expr>) {
    if constexpr (variable_symbol_v<Expr> == symbol) {
      return std::type_identity<Expr>{};
    } else {
      return std::type_identity<Constant<typename Expr::value_type>>{};
    }
  } else if constexpr (is_binary_expression_v<Expr>) {
    using L = typename Expr::lhs_type;
    using R = typename Expr::rhs_type;
    using Op = typename Expr::op_type;
    using NewL =
        typename decltype(constify_unmatched_var_impl<symbol, L>())::type;
    using NewR =
        typename decltype(constify_unmatched_var_impl<symbol, R>())::type;
    return std::type_identity<Expression<Op, NewL, NewR>>{};
  } else if constexpr (is_mono_expression_v<Expr>) {
    using E = typename Expr::lhs_type;
    using Op = typename Expr::op_type;
    using NewE =
        typename decltype(constify_unmatched_var_impl<symbol, E>())::type;
    return std::type_identity<MonoExpression<Op, NewE>>{};
  }
}

template <char symbol, typename Expr>
using constify_unmatched_var_t =
    typename decltype(constify_unmatched_var_impl<symbol, Expr>())::type;

template <char symbol, typename T>
constexpr auto make_all_constant_except(const Variable<T, symbol> &v) noexcept {
  return v;
}

template <char symbol, typename T, char othersymbol>
  requires(symbol != othersymbol)
constexpr auto make_all_constant_except(const Variable<T, othersymbol> &var) noexcept
    -> Constant<T> {
  return Constant<T>{var};
}

template <char Symbol, typename T>
constexpr auto make_all_constant_except(const Constant<T> &c) noexcept {
  return c;
}

template <char symbol, typename Op, typename LHS, typename RHS>
constexpr auto make_all_constant_except(const Expression<Op, LHS, RHS> &expr) noexcept
    -> constify_unmatched_var_t<symbol, Expression<Op, LHS, RHS>> {
  auto new_lhs = make_all_constant_except<symbol>(expr.expressions().first);
  auto new_rhs = make_all_constant_except<symbol>(expr.expressions().second);
  return {new_lhs, new_rhs};
}

template <char symbol, typename Op, typename Expr>
constexpr auto make_all_constant_except(const MonoExpression<Op, Expr> &expr) noexcept
    -> constify_unmatched_var_t<symbol, MonoExpression<Op, Expr>> {
  return make_all_constant_except<symbol>(expr.expressions());
}

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

template <typename T> consteval auto extract_symbols_impl() {
  if constexpr (is_variable_v<T>) {
    return std::type_identity<
        mp::mp_list<std::integral_constant<char, variable_symbol_v<T>>>>{};
  } else if constexpr (is_binary_expression_v<T>) {
    using L = typename T::lhs_type;
    using R = typename T::rhs_type;
    using SL = typename decltype(extract_symbols_impl<L>())::type;
    using SR = typename decltype(extract_symbols_impl<R>())::type;
    return std::type_identity<tuple_union_t<SL, SR>>{};
  } else if constexpr (is_mono_expression_v<T>) {
    using E = typename T::lhs_type;
    return extract_symbols_impl<E>();
  } else {
    return std::type_identity<mp::mp_list<>>{};
  }
}

template <typename T>
using extract_symbols_from_expr_t =
    typename decltype(extract_symbols_impl<T>())::type;

template <std::size_t N> consteval auto idx() noexcept {
  return std::integral_constant<std::size_t, N>{};
}

template <size_t value> struct idx_t : std::integral_constant<size_t, value> {};

} // namespace diff

#define IDX(value)                                                             \
  diff::idx_t<value> {}
