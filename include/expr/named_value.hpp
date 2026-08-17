#pragma once

#include "expr/expressions.hpp" // Numeric
#include "util/fixed_string.hpp"

#include <type_traits>

namespace diff {

template <FixedString Sym, Numeric V> struct NamedValue {
  static constexpr auto symbol = Sym;
  V value;
};

// named<"x">(1.25) — bind a value to a symbol by name.
template <FixedString Sym, Numeric V>
[[nodiscard]] constexpr NamedValue<Sym, V> named(V v) noexcept {
  return {v};
}

template <typename T> inline constexpr bool is_named_value_v = false;
template <FixedString S, Numeric V>
inline constexpr bool is_named_value_v<NamedValue<S, V>> = true;

template <typename T>
concept CNamedValue = is_named_value_v<std::remove_cvref_t<T>>;

} // namespace diff
