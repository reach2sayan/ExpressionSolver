#pragma once

#include <ranges>
#include <utility>

namespace ddx::impl {

// `r | to<C>()`, spelled as std::ranges::to is and forwarding to it, but piped
// by us.  libstdc++ 14.2 pipes through an adaptor whose call operator names
// std::forward_like's deduced return type while constraints are still being
// checked; clang rejects that outright, and the diagnostic is a fatal error in
// <bits/move.h> rather than anything to fix here.  14.3 rewrote the offending
// alias as a trait, so this is what keeps the spelling available on the
// toolchains in between.  The call form was never affected.
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
