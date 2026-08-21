#pragma once
// A compile-time map: FixedString keys, values of unrelated types.  The keys
// live in the type, so a lookup is a base-class cast, not a search.
//
//   constexpr auto m = map(named<"n">(3), named<"x">(1.5));
//   static_assert(m.get<"n">() == 3);   // int
//   static_assert(m["x"_s] == 1.5);     // double
//
// ValueMap (bound.hpp) is the homogeneous sibling -- one Scalar per slot,
// symbols in canonical order -- which is what a point of an expression is.

#include "expr/expressions.hpp" // symbol_type, for operator[]
#include "expr/named_value.hpp"
#include "util/config.hpp" // DDX_KEYED_ACCESSORS
#include "util/fixed_string.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ddx::impl {

template <CNamedValue... Entries> struct Map;

namespace detail {

// The comparison is on the character views, not on the FixedStrings: gcc 14
// miscounts a fold over == between class-type NTTPs, and answers 2 for
// key_count<"n", "n", "x">().  A length difference is a content difference, so
// views agree with FixedString's own operator== and "x" and "xy" still never
// collide.
template <FixedString Key, FixedString... Keys>
consteval std::size_t key_count() noexcept {
  return (std::size_t{0} + ... +
          static_cast<std::size_t>(Keys.view() == Key.view()));
}

// Position of Key in Keys..., or sizeof...(Keys) when it is absent.
template <FixedString Key, FixedString... Keys>
consteval std::size_t key_index() noexcept {
  constexpr std::size_t n = sizeof...(Keys);
  const std::array<bool, n> hit{(Keys.view() == Key.view())...};
  return static_cast<std::size_t>(std::ranges::find(hit, true) - hit.begin());
}

// The inner Keys... is expanded where it stands, so the outer fold walks the
// pack one key at a time and asks how often that key occurs in the whole.
template <FixedString... Keys> consteval bool keys_unique() noexcept {
  return (... && (key_count<Keys, Keys...>() == 1));
}

static_assert(keys_unique<"n", "x">() && keys_unique<"x", "xy">() &&
                  !keys_unique<"a", "b", "a">(),
              "keys_unique: miscounting -- see the comparison note above");

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

// The entries are base classes, not a tuple member: that is what keeps Map an
// aggregate over them, so Map{named<"n">(3), ...} and Map<E...>{{3}, ...} are
// one initialisation.
template <CNamedValue... Entries> struct Map : Entries... {
  static_assert(detail::keys_unique<Entries::symbol...>(),
                "Map: duplicate key");

  using entry_types = std::tuple<Entries...>;

  static constexpr bool kNamedMap = true;
  static constexpr std::size_t size = sizeof...(Entries);

  template <FixedString Key>
  static constexpr std::size_t index_of =
      detail::key_index<Key, Entries::symbol...>();

  template <FixedString Key>
  [[nodiscard]] static consteval bool contains() noexcept {
    return index_of<Key> < size;
  }

  template <FixedString Key>
  using entry_of = std::tuple_element_t<index_of<Key>, entry_types>;

  template <FixedString Key>
  using value_type_of = typename entry_of<Key>::value_type;

  // An lvalue map hands out a reference to its slot; a temporary hands out a
  // copy, because the slot dies with the expression it came from.
  template <FixedString Key>
  [[nodiscard]] static constexpr decltype(auto) slot(auto &&self) noexcept {
    if constexpr (!contains<Key>()) {
      static_assert(contains<Key>(), "Map: key not present (see keys())");
    } else if constexpr (std::is_lvalue_reference_v<decltype(self)>) {
      using Self = std::remove_reference_t<decltype(self)>;
      using Base = std::conditional_t<std::is_const_v<Self>,
                                      const entry_of<Key>, entry_of<Key>>;
      // Parenthesised: decltype(auto) over a bare member access deduces the
      // member's declared type, not the reference to it.
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

  // In place, and the slot keeps its type: use erase<Key>().insert(...) to
  // change that.
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

  // f(key, value) in entry order; the key is a symbol_type, so key.name is the
  // label and it can be fed straight back to operator[].
  // Unconstrained on purpose.  The natural argument is a generic lambda with a
  // deduced return type, and std::invocable has to deduce that return type --
  // which instantiates the body.  Constraining the const overload would then
  // hard-error on any `f` that writes through its second parameter, before
  // overload resolution ever got to prefer the non-const one.
  template <typename F> constexpr void for_each(F &&f) const {
    (f(sym<Entries::symbol>, static_cast<const Entries &>(*this).value), ...);
  }

  template <typename F> constexpr void for_each(F &&f) {
    (f(sym<Entries::symbol>, static_cast<Entries &>(*this).value), ...);
  }

  // Permuted keys are a different type, and do not compare at all.
  constexpr bool operator==(const Map &) const = default;
};

template <CNamedValue... Entries> Map(Entries...) -> Map<Entries...>;

template <typename T>
concept CNamedMap = requires { requires std::remove_cvref_t<T>::kNamedMap; };

template <CNamedValue... Entries>
[[nodiscard]] constexpr auto map(Entries... es) {
  // Reached before a duplicate key becomes a duplicate base class, which is
  // the less legible complaint.
  static_assert(detail::keys_unique<Entries::symbol...>(),
                "map: duplicate key");
  return Map<Entries...>{std::move(es)...};
}

} // namespace ddx::impl
