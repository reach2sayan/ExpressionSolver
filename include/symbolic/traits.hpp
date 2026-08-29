#pragma once
#include "symbolic/values.hpp"
// CFixedString appears only as a constrained-auto NTTP placeholder, which
// include-cleaner does not count as a reference -- hence the pragma.
#include "symbolic/symbol.hpp"
#include "util/fixed_string.hpp" // IWYU pragma: keep
#include <tuple>
#include <type_traits>

namespace ddx::impl {

// Same leaf, same value lookup, zero derivative.  A pure type transform, which
// keeps the symbolic Jacobian made of empty types.
template <CVariable T> struct frozen_variable;
template <Numeric T, CFixedString auto C, bool F>
struct frozen_variable<Variable<T, C, F>> {
  using type = Variable<T, C, true>;
};
template <CVariable T>
using frozen_variable_t = typename frozen_variable<T>::type;

// The two leaf rewrites are complements: one freezes the symbol named, the
// other freezes every symbol but it.  The rest -- rebuilding a node from
// rewritten children, leaving any other leaf alone -- is one walk, and a pure
// type transform, so the symbolic Jacobian stays made of empty types.
template <CFixedString auto symbol, bool FreezeMatch, CExpression T>
constexpr auto refreeze(const T &e) noexcept {
  if constexpr (CVariable<T>) {
    if constexpr ((T::label == symbol) == FreezeMatch) {
      return frozen_variable_t<T>{};
    } else {
      return e;
    }
  } else if constexpr (CExpressionNode<T>) {
    return std::apply(
        [](const auto &...child) {
          return Expression<typename T::op_type,
                            decltype(refreeze<symbol, FreezeMatch>(child))...>{
              refreeze<symbol, FreezeMatch>(child)...};
        },
        e.expressions());
  } else {
    return e;
  }
}

template <CFixedString auto symbol, CExpression E>
constexpr auto make_const_variable(const E &e) noexcept {
  return refreeze<symbol, true>(e);
}

template <CFixedString auto symbol, CExpression E>
constexpr auto make_all_constant_except(const E &e) noexcept {
  return refreeze<symbol, false>(e);
}

// Alphabetical by name; a metafunction because that is what mp_sort takes.
template <CSymbol A, CSymbol B>
struct symbol_less : std::bool_constant<(A::name < B::name)> {};

// mp_sort is stable, so a tie -- two symbols of the same name, collapsed by
// mp_unique below -- does not reorder the rest.
template <CSymbolList List> using sort_tuple_t = mp::mp_sort<List, symbol_less>;

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

namespace detail {

// The two things every driver asks of an expression: its canonical symbol list
// and how long that is.
template <CExpression Expr>
using expr_symbols_t = extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;

template <CExpression Expr>
inline constexpr std::size_t expr_arity_v =
    mp::mp_size<expr_symbols_t<Expr>>::value;

} // namespace detail

template <std::size_t N> using idx_t = std::integral_constant<std::size_t, N>;

template <std::size_t N> [[nodiscard]] consteval idx_t<N> idx() noexcept {
  return {};
}

} // namespace ddx::impl
