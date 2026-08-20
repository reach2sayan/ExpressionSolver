#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

namespace diff::impl {

template <typename T>
concept CRestorable =
    std::move_constructible<T> && std::assignable_from<T &, T>;

template <auto Seed, CRestorable T>
  requires std::convertible_to<decltype(Seed), T>
class scoped_value {
  static constexpr bool kNothrow = std::is_nothrow_move_constructible_v<T> &&
                                   std::is_nothrow_move_assignable_v<T>;
public:
  explicit constexpr scoped_value(T &slot) noexcept(kNothrow)
      : slot_(slot), saved_(std::move(slot)) {
    slot_ = static_cast<T>(Seed);
  }

  constexpr ~scoped_value() { slot_ = std::move(saved_); }

  scoped_value(const scoped_value &) = delete;
  scoped_value &operator=(const scoped_value &) = delete;
  scoped_value(scoped_value &&) = delete;
  scoped_value &operator=(scoped_value &&) = delete;

private:
  T &slot_;
  T saved_;
};

// Seed `slot` with Seed for the rest of the enclosing scope:
//   const auto guard = scoped_seed<1.0>(dof[ai].deriv().value);
template <auto Seed, CRestorable T>
[[nodiscard]] constexpr auto
scoped_seed(T &slot) noexcept(noexcept(scoped_value<Seed, T>{slot})) {
  return scoped_value<Seed, T>{slot};
}

} // namespace diff::impl
