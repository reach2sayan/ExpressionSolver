#pragma once
#include "fixed_string.hpp" // CFixedString
#include "values.hpp"
#include "mpl.hpp"
#include <type_traits>

namespace diff {

namespace mp = diff::mpl;

// Direct index_sequence fold — no Boost mp_for_each intermediate lambda.
template <std::size_t N, CIndexedCallable<N> F>
constexpr void static_for(F &&f) noexcept {
  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    (std::forward<F>(f).template operator()<Is>(), ...);
  }(std::make_index_sequence<N>{});
}

template <typename T> inline constexpr bool is_variable_v = false;
template <typename T, CFixedString auto C, bool F>
inline constexpr bool is_variable_v<Variable<T, C, F>> = true;

template <typename T>
concept CVariable = is_variable_v<std::remove_cvref_t<T>>;

// Freezing a symbol: same leaf, same value lookup, zero derivative.  Leaves
// carry no value, so this is a pure type transform — which is precisely what
// lets the symbolic Jacobian be built out of empty types.
template <CVariable T> struct frozen_variable;
template <Numeric T, CFixedString auto C, bool F>
struct frozen_variable<Variable<T, C, F>> {
  using type = Variable<T, C, true>;
};
template <CVariable T>
using frozen_variable_t = typename frozen_variable<T>::type;

template <CExpression T> consteval auto make_all_constant_impl() {
  if constexpr (CVariable<T>) {
    return std::type_identity<frozen_variable_t<T>>{};
  } else if constexpr (CExpressionNode<T>) {
    return []<COperation Op, CExpression... C>(
               std::type_identity<Expression<Op, C...>>) {
      return std::type_identity<Expression<
          Op, typename decltype(make_all_constant_impl<C>())::type...>>{};
    }(std::type_identity<T>{});
  } else {
    return std::type_identity<T>{};
  }
}

template <CExpression TExpression>
using as_const_expression =
    typename decltype(make_all_constant_impl<TExpression>())::type;

template <CFixedString auto symbol, CExpression T>
consteval auto replace_matching_var_impl() {
  if constexpr (CVariable<T>) {
    if constexpr (T::label == symbol) {
      return std::type_identity<frozen_variable_t<T>>{};
    } else {
      return std::type_identity<T>{};
    }
  } else if constexpr (CExpressionNode<T>) {
    return []<COperation Op, CExpression... C>(
               std::type_identity<Expression<Op, C...>>) {
      return std::type_identity<
          Expression<Op, typename decltype(replace_matching_var_impl<
                                           symbol, C>())::type...>>{};
    }(std::type_identity<T>{});
  } else {
    return std::type_identity<T>{};
  }
}

template <CFixedString auto symbol, CExpression T>
using replace_matching_variable_as_const_t =
    typename decltype(replace_matching_var_impl<symbol, T>())::type;

template <CFixedString auto symbol, Numeric T, bool F>
constexpr auto make_const_variable(const Variable<T, symbol, F> &) noexcept {
  return Variable<T, symbol, true>{};
}

template <CFixedString auto symbol, Numeric T, CFixedString auto othersymbol,
          bool F>
  requires(symbol != othersymbol)
constexpr auto
make_const_variable(const Variable<T, othersymbol, F> &var) noexcept
    -> Variable<T, othersymbol, F> {
  return var;
}

template <CFixedString auto symbol, Numeric T>
constexpr auto make_const_variable(const Constant<T> &c) noexcept {
  return c;
}

template <CFixedString auto symbol, Numeric T, T V>
constexpr auto make_const_variable(const Lit<T, V> &l) noexcept {
  return l;
}

template <CFixedString auto symbol, COperation Op, CExpression... C>
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

template <CFixedString auto symbol, CExpression Expr>
consteval auto constify_unmatched_var_impl() {
  if constexpr (CConstant<Expr>) {
    return std::type_identity<Expr>{};
  } else if constexpr (CVariable<Expr>) {
    if constexpr (Expr::label == symbol) {
      return std::type_identity<Expr>{};
    } else {
      return std::type_identity<frozen_variable_t<Expr>>{};
    }
  } else if constexpr (CExpressionNode<Expr>) { // already nested correctly
    return []<COperation Op, CExpression... C>(
               std::type_identity<Expression<Op, C...>>) {
      return std::type_identity<
          Expression<Op, typename decltype(constify_unmatched_var_impl<
                                           symbol, C>())::type...>>{};
    }(std::type_identity<Expr>{});
  }
}

template <CFixedString auto symbol, CExpression Expr>
using constify_unmatched_var_t =
    typename decltype(constify_unmatched_var_impl<symbol, Expr>())::type;

template <CFixedString auto symbol, Numeric T, bool F>
constexpr auto
make_all_constant_except(const Variable<T, symbol, F> &v) noexcept {
  return v;
}

template <CFixedString auto symbol, Numeric T, CFixedString auto othersymbol,
          bool F>
  requires(symbol != othersymbol)
constexpr auto make_all_constant_except(
    const Variable<T, othersymbol, F> &) noexcept -> Variable<T, othersymbol, true> {
  return {};
}

template <CFixedString auto symbol, Numeric T>
constexpr auto make_all_constant_except(const Constant<T> &c) noexcept {
  return c;
}

template <CFixedString auto symbol, Numeric T, T V>
constexpr auto make_all_constant_except(const Lit<T, V> &l) noexcept {
  return l;
}

template <CFixedString auto symbol, COperation Op, CExpression... C>
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

// Canonical symbol order: alphabetical by name.  A consteval predicate rather
// than an integral_constant metafunction — the ordering is spelled once, as the
// comparison that actually decides it.
inline constexpr auto symbol_less = []<CSymbol A, CSymbol B>() consteval {
  return A::name < B::name;
};
template <CSymbolList List>
using sort_tuple_t = mp::mp_sort<List, symbol_less>;

template <CSymbolList List>
using unique_tuple_t = mp::mp_unique<sort_tuple_t<List>>;

template <CSymbolList... Lists>
using tuple_union_t = unique_tuple_t<mp::mp_append<Lists...>>;

template <CExpression T> consteval auto extract_symbols_impl() {
  if constexpr (CVariable<T>) {
    return std::type_identity<mp::mp_list<symbol_type<T::label>>>{};
  } else if constexpr (CExpressionNode<T>) {
    return []<COperation Op, CExpression... C>(
               std::type_identity<Expression<Op, C...>>) {
      return std::type_identity<tuple_union_t<
          typename decltype(extract_symbols_impl<C>())::type...>>{};
    }(std::type_identity<T>{});
  } else {
    return std::type_identity<mp::mp_list<>>{};
  }
}

template <CExpression T>
using extract_symbols_from_expr_t =
    typename decltype(extract_symbols_impl<T>())::type;

// A compile-time index carried as a *value*, for the one place that cannot
// take a template argument: operator[].  Spell it idx<N>() at the call site.
// Prefer the template form — eq.get<N>() — everywhere else; it needs no tag
// type at all.
//
// Not a std::integral_constant specialisation: the only thing inherited from it
// was ::value, and spelling that here keeps the index API inside the library
// (and out of overload resolution for anything else keyed on integral_constant).
template <std::size_t N> struct idx_t {
  static constexpr std::size_t value = N;
};

template <std::size_t N> [[nodiscard]] consteval idx_t<N> idx() noexcept {
  return {};
}

} // namespace diff
