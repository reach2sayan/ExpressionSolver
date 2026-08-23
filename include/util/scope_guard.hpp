#pragma once

#include "util/pinned.hpp"

#include <concepts>
#include <type_traits>
#include <utility>

namespace ddx::impl {

template <typename T>
concept CRestorable =
    std::move_constructible<T> && std::assignable_from<T &, T>;

// Hold `slot` at `next` for the rest of the enclosing scope, then put back what
// was there.  Saved, not recomputed: undoing a seed as `slot -= 1` is not the
// identity in floating point.  Saving is also what makes the guards nest, and
// what restores the slot when an exception escapes.
template <CRestorable T> class scoped_value : private pinned {
  static constexpr bool kNothrow = std::is_nothrow_move_constructible_v<T> &&
                                   std::is_nothrow_move_assignable_v<T>;

public:
  constexpr scoped_value(T &slot, T next) noexcept(kNothrow)
      : slot_(slot), saved_(std::exchange(slot, std::move(next))) {}

  constexpr ~scoped_value() { slot_ = std::move(saved_); }

private:
  T &slot_;
  T saved_;
};

// Seed `slot` with Seed for the rest of the enclosing scope:
//   const auto guard = scoped_seed<1.0>(dof[ai].deriv().value);
template <auto Seed, CRestorable T>
  requires std::convertible_to<decltype(Seed), T>
[[nodiscard]] constexpr scoped_value<T> scoped_seed(T &slot) noexcept(
    noexcept(scoped_value<T>{slot, static_cast<T>(Seed)})) {
  return scoped_value<T>{slot, static_cast<T>(Seed)};
}

} // namespace ddx::impl
