#pragma once
#include "values.hpp"
#include "mpl.hpp"
#include <type_traits>

namespace diff {

namespace mp = diff::mpl;

// Direct index_sequence fold — no Boost mp_for_each intermediate lambda.
template <std::size_t N, class F> constexpr void static_for(F &&f) noexcept {
  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    (std::forward<F>(f).template operator()<Is>(), ...);
  }(std::make_index_sequence<N>{});
}

template <typename T> constexpr static bool is_const = false;
template <typename T> constexpr static bool is_const<Constant<T>> = true;

template <typename T> inline constexpr bool is_variable_v = false;
template <typename T, CFixedString auto C, typename S>
inline constexpr bool is_variable_v<Variable<T, C, S>> = true;

// True for any internal node (an Expression of any arity), false for leaves.
template <typename T> inline constexpr bool is_expression_node_v = false;
template <typename Op, typename... C>
inline constexpr bool is_expression_node_v<Expression<Op, C...>> = true;

template <typename T> consteval auto make_all_constant_impl() {
  if constexpr (is_variable_v<T>) {
    return std::type_identity<Constant<typename T::value_type>>{};
  } else if constexpr (is_expression_node_v<T>) {
    return []<typename Op, typename... C>(std::type_identity<Expression<Op, C...>>) {
      return std::type_identity<
          Expression<Op, typename decltype(make_all_constant_impl<C>())::type...>>{};
    }(std::type_identity<T>{});
  } else {
    return std::type_identity<T>{};
  }
}

template <typename T>
using make_all_constant_t =
    typename decltype(make_all_constant_impl<T>())::type;

template <typename TExpression>
using as_const_expression = make_all_constant_t<TExpression>;

template <CFixedString auto symbol, typename T>
consteval auto replace_matching_var_impl() {
  if constexpr (is_variable_v<T>) {
    if constexpr (T::label == symbol) {
      return std::type_identity<Constant<typename T::value_type>>{};
    } else {
      return std::type_identity<T>{};
    }
  } else if constexpr (is_expression_node_v<T>) {
    return []<typename Op, typename... C>(std::type_identity<Expression<Op, C...>>) {
      return std::type_identity<Expression<
          Op, typename decltype(replace_matching_var_impl<symbol, C>())::type...>>{};
    }(std::type_identity<T>{});
  } else {
    return std::type_identity<T>{};
  }
}

template <CFixedString auto symbol, typename T>
using replace_matching_variable_as_const_t =
    typename decltype(replace_matching_var_impl<symbol, T>())::type;

template <CFixedString auto symbol, typename T, typename S>
constexpr auto make_const_variable(const Variable<T, symbol, S> &var) noexcept {
  return Constant<T>{static_cast<T>(var)};
}

template <CFixedString auto symbol, typename T, CFixedString auto othersymbol,
          typename S>
  requires(symbol != othersymbol)
constexpr auto
make_const_variable(const Variable<T, othersymbol, S> &var) noexcept
    -> Variable<T, othersymbol, S> {
  return var;
}

template <CFixedString auto symbol, typename T>
constexpr auto make_const_variable(const Constant<T> &c) noexcept {
  return c;
}

template <CFixedString auto symbol, typename Op, typename... C>
constexpr auto make_const_variable(const Expression<Op, C...> &expr) noexcept
    -> Expression<Op, replace_matching_variable_as_const_t<symbol, C>...> {
  return std::apply(
      [](const auto &...child) {
        return Expression<Op,
                          replace_matching_variable_as_const_t<symbol, C>...>{
            make_const_variable<symbol>(child)...};
      },
      expr.expressions());
}

template <CFixedString auto symbol, typename Expr>
consteval auto constify_unmatched_var_impl() {
  if constexpr (is_constant_v<Expr>) {
    return std::type_identity<Expr>{};
  } else if constexpr (is_variable_v<Expr>) {
    if constexpr (Expr::label == symbol) {
      return std::type_identity<Expr>{};
    } else {
      return std::type_identity<Constant<typename Expr::value_type>>{};
    }
  } else if constexpr (is_expression_node_v<Expr>) { // already nested correctly
    return []<typename Op, typename... C>(std::type_identity<Expression<Op, C...>>) {
      return std::type_identity<Expression<
          Op, typename decltype(constify_unmatched_var_impl<symbol, C>())::type...>>{};
    }(std::type_identity<Expr>{});
  }
}

template <CFixedString auto symbol, typename Expr>
using constify_unmatched_var_t =
    typename decltype(constify_unmatched_var_impl<symbol, Expr>())::type;

template <CFixedString auto symbol, typename T, typename S>
constexpr auto
make_all_constant_except(const Variable<T, symbol, S> &v) noexcept {
  return v;
}

template <CFixedString auto symbol, typename T, CFixedString auto othersymbol,
          typename S>
  requires(symbol != othersymbol)
constexpr auto
make_all_constant_except(const Variable<T, othersymbol, S> &var) noexcept
    -> Constant<T> {
  return Constant<T>{static_cast<T>(var)};
}

template <CFixedString auto symbol, typename T>
constexpr auto make_all_constant_except(const Constant<T> &c) noexcept {
  return c;
}

template <CFixedString auto symbol, typename Op, typename... C>
constexpr auto
make_all_constant_except(const Expression<Op, C...> &expr) noexcept
    -> constify_unmatched_var_t<symbol, Expression<Op, C...>> {
  return std::apply(
      [](const auto &...child) {
        return constify_unmatched_var_t<symbol, Expression<Op, C...>>{
            make_all_constant_except<symbol>(child)...};
      },
      expr.expressions());
}

template <typename A, typename B>
using symbol_less = mp::mp_bool<(A::name < B::name)>;
template <typename List> using sort_tuple_t = mp::mp_sort<List, symbol_less>;

template <typename List>
using unique_tuple_t = mp::mp_unique<sort_tuple_t<List>>;

template <typename... Lists>
using tuple_union_t = unique_tuple_t<mp::mp_append<Lists...>>;

template <typename T> consteval auto extract_symbols_impl() {
  if constexpr (is_variable_v<T>) {
    return std::type_identity<mp::mp_list<symbol_type<T::label>>>{};
  } else if constexpr (is_expression_node_v<T>) {
    return []<typename Op, typename... C>(std::type_identity<Expression<Op, C...>>) {
      return std::type_identity<
          tuple_union_t<typename decltype(extract_symbols_impl<C>())::type...>>{};
    }(std::type_identity<T>{});
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
