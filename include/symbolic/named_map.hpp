#pragma once
// FixedString keys, values of unrelated types.  The keys live in the type, so a
// lookup is a base-class cast, not a search.
//
//   constexpr auto m = map(named<"n">(3), named<"x">(1.5));
//   static_assert(m.get<"n">() == 3);   // int
//   static_assert(m["x"_s] == 1.5);     // double
//
// ValueMap (bound.hpp) is the homogeneous sibling: one Scalar per slot, in
// canonical order, which is what a point of an expression is.

#include "symbolic/expressions.hpp" // symbol_type, for operator[]
#include "symbolic/named_value.hpp"
#include "symbolic/symbol.hpp" // mp
#include "util/config.hpp"     // DDX_KEYED_ACCESSORS
#include "util/fixed_string.hpp"

#include <array>
#include <cstddef>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ddx::impl {

template <CNamedValue... Entries> struct Map;

namespace detail {

// Lifted to symbol_type so Mp11 answers by type identity: gcc 14 miscounts a
// fold over `==` between class-type NTTPs, answering 2 for
// key_count<"n", "n", "x">().
template <FixedString... Keys>
using key_list = mp::mp_list<symbol_type<Keys>...>;

template <FixedString Key, FixedString... Keys>
inline constexpr std::size_t key_count =
    mp::mp_count<key_list<Keys...>, symbol_type<Key>>::value;

// Position of Key in Keys..., or sizeof...(Keys) when it is absent -- which is
// what mp_find answers with.
template <FixedString Key, FixedString... Keys>
inline constexpr std::size_t key_index =
    mp::mp_find<key_list<Keys...>, symbol_type<Key>>::value;

template <FixedString... Keys>
inline constexpr bool keys_unique =
    mp::mp_size<mp::mp_unique<key_list<Keys...>>>::value == sizeof...(Keys);

template <FixedString Key, CNamedValue E>
[[nodiscard]] constexpr auto entry_unless(const E &e) {
  if constexpr (E::symbol == Key) {
    return std::tuple<>{};
  } else {
    return std::tuple<E>{e};
  }
}

template <CNamedValue... Es>
[[nodiscard]] constexpr Map<Es...>
map_from_entries(const std::tuple<Es...> &es) {
  return std::apply([](const Es &...e) { return Map<Es...>{e...}; }, es);
}

} // namespace detail

// Base classes, not a tuple member: that keeps Map an aggregate, so
// Map{named<"n">(3), ...} and Map<E...>{{3}, ...} are one initialisation.
template <CNamedValue... Entries> struct Map : Entries... {
  static_assert(detail::keys_unique<Entries::symbol...>, "Map: duplicate key");

  using entry_types = std::tuple<Entries...>;

  static constexpr bool kNamedMap = true;
  static constexpr std::size_t size = sizeof...(Entries);

  template <FixedString Key>
  static constexpr std::size_t index_of =
      detail::key_index<Key, Entries::symbol...>;

  template <FixedString Key>
  [[nodiscard]] static consteval bool contains() noexcept {
    return index_of<Key> < size;
  }

  template <FixedString Key>
  using entry_of = std::tuple_element_t<index_of<Key>, entry_types>;

  template <FixedString Key>
  using value_type_of = typename entry_of<Key>::value_type;

  // An lvalue hands out a reference to its slot, a temporary a copy.
  template <FixedString Key>
  [[nodiscard]] static constexpr decltype(auto) slot(auto &&self) noexcept {
    if constexpr (!contains<Key>()) {
      static_assert(contains<Key>(), "Map: key not present (see keys())");
    } else if constexpr (std::is_lvalue_reference_v<decltype(self)>) {
      using Self = std::remove_reference_t<decltype(self)>;
      using Base = std::conditional_t<std::is_const_v<Self>,
                                      const entry_of<Key>, entry_of<Key>>;
      // Parenthesised, or decltype(auto) deduces the member's declared type.
      return (static_cast<Base &>(self).value);
    } else {
      return value_type_of<Key>{static_cast<const entry_of<Key> &>(self).value};
    }
  }

  DDX_KEYED_ACCESSORS(FixedString S, CFixedString auto S, S, symbol_type<S>)

  [[nodiscard]] static constexpr std::array<std::string_view, size>
  keys() noexcept {
    return {Entries::symbol.view()...};
  }

  // In place, and the slot keeps its type; erase<Key>().insert(...) does not.
  template <FixedString Key, typename U>
  constexpr void set(U &&v) noexcept(
      std::is_nothrow_assignable_v<value_type_of<Key> &, U &&>) {
    slot<Key>(*this) = std::forward<U>(v);
  }

  // The key set is part of the type, so these return new maps.
  template <FixedString... Keys, typename... Vs>
  [[nodiscard]] constexpr auto insert(NamedValue<Keys, Vs>... es) const {
    static_assert((... && !contains<Keys>()),
                  "Map::insert: key already present (use set<Key>)");
    return Map<Entries..., NamedValue<Keys, Vs>...>{
        static_cast<const Entries &>(*this)..., std::move(es)...};
  }

  template <FixedString Key> [[nodiscard]] constexpr auto erase() const {
    static_assert(contains<Key>(), "Map::erase: key not present");
    return detail::map_from_entries(std::tuple_cat(
        detail::entry_unless<Key>(static_cast<const Entries &>(*this))...));
  }

  // f(key, value) in entry order; key.name feeds straight back into
  // operator[].  Unconstrained on purpose: std::invocable would instantiate a
  // generic lambda's body, and the const overload would then hard-error on an
  // `f` that writes through its second parameter.
  constexpr void for_each(auto &&f) const {
    (f(sym<Entries::symbol>, static_cast<const Entries &>(*this).value), ...);
  }
  constexpr void for_each(auto &&f) {
    (f(sym<Entries::symbol>, static_cast<Entries &>(*this).value), ...);
  }

  // Permuted keys are a different type, and do not compare at all.
  constexpr bool operator==(const Map &) const = default;
};

template <CNamedValue... Entries> Map(Entries...) -> Map<Entries...>;

template <typename T>
concept CNamedMap = requires { requires std::remove_cvref_t<T>::kNamedMap; };

[[nodiscard]] constexpr auto map(CNamedValue auto... es) {
  // Reached before a duplicate key becomes a duplicate base class.
  static_assert(
      detail::keys_unique<std::remove_cvref_t<decltype(es)>::symbol...>,
      "map: duplicate key");
  return Map{std::move(es)...};
}

} // namespace ddx::impl
