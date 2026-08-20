#pragma once

#include "expr/expressions.hpp" // Numeric, symbol_type, Variable
#include "util/fixed_string.hpp"

#include <concepts>
#include <type_traits>
#include <utility>

namespace ddx::impl {

namespace detail {

// The label a tag names: sym<"x"> / "x"_s carry it as ::value, var<"x"> as
// ::label.  Left undefined for everything else, which is what CSymbolTag tests.
template <typename T> struct tag_key_of {};
template <auto S> struct tag_key_of<symbol_type<S>> {
  static constexpr auto value = S;
};
template <Numeric T, CFixedString auto S, bool F>
struct tag_key_of<Variable<T, S, F>> {
  static constexpr auto value = S;
};

template <typename T> using tag_key_t = tag_key_of<std::remove_cvref_t<T>>;

} // namespace detail

// A symbol named at the type level: "x"_s, sym<"x">, or var<"x">.
template <typename T>
concept CSymbolTag = requires { detail::tag_key_t<T>::value; };

// One keyword argument: a label bound to a value.  The label is a template
// parameter, so it costs nothing at run time and is matchable at compile time.
//
//   named<"x">(1.5)      NamedValue{"x"_s, 1.5}      NamedValue{var<"x">, 1.5}
//
// V is any object type, not just Numeric: Map (named_map.hpp) keys arbitrary
// values by this same spelling.  The numeric entry points -- values(), bind(),
// Equation::eval() -- constrain V to Numeric where they take it.
template <FixedString Sym, typename V>
  requires std::is_object_v<V>
struct NamedValue {
  using value_type = V;
  static constexpr auto symbol = Sym;

  V value;

  constexpr NamedValue()
    requires std::default_initializable<V>
  = default;

  // Not explicit: Map<...>{{1}, {2.5}} initialises its entries through this.
  constexpr NamedValue(V v) noexcept(std::is_nothrow_move_constructible_v<V>)
      : value(std::move(v)) {}

  template <CSymbolTag Tag>
    requires(detail::tag_key_t<Tag>::value == Sym)
  constexpr NamedValue(const Tag &,
                       V v) noexcept(std::is_nothrow_move_constructible_v<V>)
      : value(std::move(v)) {}

  constexpr bool operator==(const NamedValue &) const = default;
};

template <CSymbolTag Tag, typename V>
NamedValue(const Tag &, V) -> NamedValue<detail::tag_key_t<Tag>::value, V>;

// named<"x">(1.25) -- bind a value to a symbol by name.
template <FixedString Sym, typename V>
[[nodiscard]] constexpr NamedValue<Sym, V>
named(V v) noexcept(std::is_nothrow_move_constructible_v<V>) {
  return {std::move(v)};
}

// named("x"_s, 1.25) / named(var<"x">, 1.25) -- keyed by a tag in hand.
template <CSymbolTag Tag, typename V>
[[nodiscard]] constexpr auto
named(const Tag &, V v) noexcept(std::is_nothrow_move_constructible_v<V>) {
  return NamedValue<detail::tag_key_t<Tag>::value, V>{std::move(v)};
}

template <typename T> inline constexpr bool is_named_value_v = false;
template <FixedString S, typename V>
inline constexpr bool is_named_value_v<NamedValue<S, V>> = true;

template <typename T>
concept CNamedValue = is_named_value_v<std::remove_cvref_t<T>>;

} // namespace ddx::impl
