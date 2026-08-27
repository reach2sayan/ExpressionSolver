#pragma once

#include <algorithm>
#include <concepts>
#include <iterator>
#include <ranges>
#include <utility>

namespace ddx::impl {

// `r | to<C>()`, forwarding to std::ranges::to but piped by us: libstdc++ 14.2
// pipes through an adaptor naming std::forward_like's deduced return type while
// constraints are still being checked, which clang rejects.
template <typename C> struct to_closure {
  template <std::ranges::input_range R>
    requires requires(R &&r) { std::ranges::to<C>(std::forward<R>(r)); }
  [[nodiscard]] friend constexpr C operator|(R &&r, to_closure) {
    return std::ranges::to<C>(std::forward<R>(r));
  }
};

template <typename C> [[nodiscard]] constexpr to_closure<C> to() noexcept {
  return {};
}

// missing c.append_range(r) in libstdc++ 14 (available on 15)
template <typename C, std::ranges::input_range R>
  requires std::convertible_to<std::ranges::range_reference_t<R>,
                               typename C::value_type>
constexpr void append(C &c, R &&r) {
  std::ranges::copy(std::forward<R>(r), std::back_inserter(c));
}

} // namespace ddx::impl
