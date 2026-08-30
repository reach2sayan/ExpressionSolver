#pragma once

#include "util/config.hpp"

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
  [[nodiscard]] friend constexpr C operator|(std::ranges::input_range auto &&r,
                                             to_closure)
    requires requires { std::ranges::to<C>(DDX_FWD(r)); }
  {
    return std::ranges::to<C>(DDX_FWD(r));
  }
};

template <typename C> [[nodiscard]] constexpr to_closure<C> to() noexcept {
  return {};
}

// missing c.append_range(r) in libstdc++ 14 (available on 15)
constexpr void append(auto &c, std::ranges::input_range auto &&r)
  requires std::convertible_to<std::ranges::range_reference_t<decltype(r)>,
                               std::ranges::range_value_t<decltype(c)>>
{
  std::ranges::copy(DDX_FWD(r), std::back_inserter(c));
}

} // namespace ddx::impl
