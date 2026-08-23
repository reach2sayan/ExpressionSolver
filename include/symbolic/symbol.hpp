#pragma once

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

#include <cstddef>
#include <type_traits>

namespace ddx::impl {

// Boost.Mp11 is the list vocabulary; `mp::` is how the whole library spells it.
namespace mp = boost::mp11;

template <auto S> struct symbol_type;

// A symbol lifted to a type; spelled here because the list vocabulary needs it.
// Both spellings are used: the variable by the concept below, the trait by
// mp_all_of, which takes a metafunction rather than a value.
template <typename T> inline constexpr bool is_symbol_v = false;
template <auto S> inline constexpr bool is_symbol_v<symbol_type<S>> = true;
template <typename T> struct is_symbol : std::bool_constant<is_symbol_v<T>> {};

template <typename T>
concept CSymbol = is_symbol_v<std::remove_cvref_t<T>>;

template <typename L>
concept CTypeList = mp::mp_is_list<std::remove_cvref_t<L>>::value;

// mp_list<symbol_type<...>...>, alphabetical by name: the `Syms` threaded
// through every seeded sweep, and what turns a label into a slot index.
//
// CTypeList first and the conjunction is load-bearing: mp_all_of over a
// non-list is ill-formed rather than false, and a concept only reaches its
// second operand when the first held.
template <typename L>
concept CSymbolList =
    CTypeList<L> && mp::mp_all_of<std::remove_cvref_t<L>, is_symbol>::value;

// Which slot a label names, or the list's length if it names none.
//
// `auto S` and not `CFixedString auto S`, which is what every caller has: on
// gcc 15, writing the pattern symbol_type<S> against a *constrained* auto NTTP
// makes every later deduction of symbol_type<S> fail -- m[sym<"x">] stops
// finding ValueMap::operator[], with S's type reported as the class's first
// template argument. Forming the type behind an unconstrained parameter costs
// nothing and keeps the two apart. Callers keep their constraint.
template <auto S, CTypeList List>
consteval std::size_t symbol_index() noexcept {
  return mp::mp_find<List, symbol_type<S>>::value;
}

} // namespace ddx::impl
