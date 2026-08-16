#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

namespace diff {

// What a guarded slot has to support: move the old value out, write a new one
// in, move the old one back.  Deliberately weaker than std::movable — the slots
// guarded here are bare scalars and Dual components, and swappability is not
// part of the contract.
template <typename T>
concept CRestorable =
    std::move_constructible<T> && std::assignable_from<T &, T>;

// std::lock_guard for a value slot: seed on construction, restore on scope
// exit.
//
// Every driver in this library differentiates by writing a seed into ONE scalar
// of a pre-built dof/seed array, running a sweep, and writing that scalar back
// — the per-probe optimisation that avoids reconstructing a whole dual2nd or
// VectorDual per seed.  Spelling the pair by hand leaves the reset as a bare
// statement that can drift from its seed as a loop body grows, and that a throw
// out of the energy skips entirely.  Here the seed's lifetime IS the scope, so
// nesting two guards expresses "inner-derivative seed held across the whole
// i-loop, outer-derivative seed toggled per probe" as structure rather than as
// a comment (see hessian_scalar in forward_driver.hpp).
//
// The guard is over the scalar, never over the enclosing dual: two seeds may
// live in different scalars of the SAME dual2nd, and a guard over the whole
// number would clobber its sibling.
//
// Seed is an NTTP: every seeding site in this library writes a literal (1, or
// S{1}), so the value belongs to the type rather than to a runtime argument.
// That leaves the guard one reference plus the saved scalar, with nothing to
// pass at the call site.  Because a class template cannot deduce T while Seed
// is given explicitly, construct it through scoped_seed() below.
//
// Restore is save-and-write-back, not an arithmetic undo.  `slot += 1` on entry
// and `slot -= 1` on exit would drop the saved member entirely, but a
// floating-point round trip is NOT the identity (0.1 + 1 - 1 != 0.1), and a
// library that symmetrizes Hessians to shed last-ULP noise should not restore
// its own seeds approximately.  Writing the saved bits back is exact for every
// base value, not just the zero these drivers happen to use.
template <auto Seed, CRestorable T>
  requires std::convertible_to<decltype(Seed), T>
class scoped_value {
  // Conditional rather than unconditional so the guard can sit inside the
  // `constexpr noexcept` symbolic sweeps without lying about them.
  static constexpr bool kNothrow = std::is_nothrow_move_constructible_v<T> &&
                                   std::is_nothrow_move_assignable_v<T>;

public:
  explicit constexpr scoped_value(T &slot) noexcept(kNothrow)
      : slot_(slot), saved_(std::move(slot)) {
    slot_ = static_cast<T>(Seed);
  }

  constexpr ~scoped_value() { slot_ = std::move(saved_); }

  // A guard that could be copied or moved would need a disengaged state and a
  // branch in the destructor; nothing here transfers ownership.
  scoped_value(const scoped_value &) = delete;
  scoped_value &operator=(const scoped_value &) = delete;
  scoped_value(scoped_value &&) = delete;
  scoped_value &operator=(scoped_value &&) = delete;

private:
  T &slot_;
  T saved_;
};

// Seed `slot` with Seed for the rest of the enclosing scope:
//
//   const auto guard = scoped_seed<1.0>(dof[ai].deriv().value);
//
// The guard is immovable, so this relies on guaranteed copy elision — the
// prvalue initialises the caller's object directly and no move ever happens.
//
// [[nodiscard]] is load-bearing: a discarded `scoped_seed<1.0>(slot);` would
// destruct immediately and silently do nothing.  Naming the guard is the idiom.
template <auto Seed, CRestorable T>
[[nodiscard]] constexpr auto scoped_seed(T &slot) noexcept(
    noexcept(scoped_value<Seed, T>{slot})) {
  return scoped_value<Seed, T>{slot};
}

} // namespace diff
