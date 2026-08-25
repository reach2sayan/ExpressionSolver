#pragma once

#include <ranges>
#include <utility>

namespace ddx::impl {

// `r | to<C>()`, forwarding to std::ranges::to but piped by us: libstdc++ 14.2
// pipes through an adaptor whose call operator names std::forward_like's
// deduced return type while constraints are still being checked, which clang
// rejects outright.  14.3 rewrote it as a trait;
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

} // namespace ddx::impl
