#pragma once

#include "dual/dual.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

namespace diff {

// An energy as every driver here calls one: with a pointer into the driver's
// own seed buffer, returning a number of the same type.  The buffer's length is
// not part of the type; the callable reads a window the caller has guaranteed
// (see SeededExprEnergy in seeded_energy.hpp).  D is what distinguishes the
// drivers: `dual` for the gradient sweep, `dual2nd` for the Hessian.
template <typename F, typename D>
concept CEnergyOf =
    std::invocable<F &, const D *> &&
    std::convertible_to<std::invoke_result_t<F &, const D *>, D>;

// Every driver here indexes `active` and asks for its size, and nothing else --
// so a `views::iota` and a real span both model it, and the all-variables case
// costs no container.
template <typename R>
concept CIndexRange =
    std::ranges::random_access_range<R> && std::ranges::sized_range<R> &&
    std::convertible_to<std::ranges::range_value_t<R>, std::size_t>;

// The point is a `std::span<const double>`, a concrete type and not a concept:
// the parameter is non-deduced, so a vector, array, C array, contiguous view or
// temporary all convert at the call site with nothing spelled.
namespace detail {
// 0,1,...,n-1 without materialising it -- what a driver means by "all variables".
[[nodiscard]] constexpr auto all_indices(std::size_t n) noexcept {
  return std::views::iota(std::size_t{0}, n);
}
} // namespace detail

// What a second-order sweep hands back: owning std types the caller takes over,
// in two shapes that destructure alike (`auto [v, g, H, n] = ...`).
//
//   value    f(x)
//   gradient n entries
//   hessian  n*n entries, ROW-MAJOR: element (i, j) is hessian[i * n + j]

// Runtime extent: two owning buffers plus the extent they are sized by.
using HessianOwned = std::tuple<double, std::unique_ptr<double[]>,
                                std::unique_ptr<double[]>, std::size_t>;

// Compile-time extent: no allocation, and constant-evaluable.  N is the extent.
template <std::size_t N>
using HessianStatic =
    std::tuple<double, std::array<double, N>, std::array<double, N * N>>;

namespace detail {
// Average each (i, j) / (j, i) pair in place.  Mirrored entries come from
// independent sweeps and can differ in the last ULP; H is analytically
// symmetric, so this removes only that noise.  The sparse path names each entry
// once and does not call it.
constexpr void symmetrize(double *const h, const std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double s = 0.5 * (h[i * n + j] + h[j * n + i]);
      h[i * n + j] = s;
      h[j * n + i] = s;
    }
  }
}

// Copy a static-extent result into owning buffers.  Needed only where one entry
// point can reach both shapes and has to return one type -- the subset-taking
// hessian().  The static form stays the return type everywhere else.
template <std::size_t N>
[[nodiscard]] inline HessianOwned to_owned(HessianStatic<N> &&h) {
  auto grad = std::make_unique_for_overwrite<double[]>(N);
  auto hess = std::make_unique_for_overwrite<double[]>(N * N);
  const auto& [value, grad_calc, hess_calc] = h;
  std::ranges::copy(grad_calc, grad.get());
  std::ranges::copy(hess_calc, hess.get());
  return {value, std::move(grad), std::move(hess), N};
}

// Uninitialised owning buffer.  make_unique_for_overwrite, not make_unique: the
// sweep writes every cell, so value-initialising first is a wasted memset.
[[nodiscard]] inline std::unique_ptr<double[]> raw_buffer(std::size_t k) {
  return std::make_unique_for_overwrite<double[]>(k);
}
} // namespace detail

// Caller-owned scratch for a sweep's seeded variables.  The dof array is pure
// working storage, so one workspace handed to a loop over many points allocates
// on the first call and never again.  Small points are seeded into an inline
// block instead -- *raw* storage, so an unused tail is never constructed, which
// is why D must be trivially destructible.

namespace detail {
// The inline block, or nothing at all.  An array member costs a stack-protector
// canary on every function holding the workspace, so a D too wide to ever fit a
// whole point gets no block and [[no_unique_address]] gives the stand-in no size.
template <Numeric D, std::size_t Bytes> struct inline_block {
  alignas(D) std::byte storage[Bytes];
};
struct no_inline_block {};
} // namespace detail

template <Numeric D> struct SweepWorkspace {
  static_assert(std::is_trivially_destructible_v<D>,
                "the inline block is reused and abandoned without destruction");

  // A byte budget, not an element count: D may be much wider than dual2nd's
  // 32 B.  A block that cannot hold `inline_floor` seeds is not declared.
  static constexpr std::size_t inline_bytes = 512;
  static constexpr std::size_t inline_floor = 8;
  static constexpr std::size_t inline_capacity =
      (inline_bytes / sizeof(D) >= inline_floor) ? inline_bytes / sizeof(D) : 0;

  [[no_unique_address]] std::conditional_t<
      (inline_capacity > 0), detail::inline_block<D, inline_bytes>,
      detail::no_inline_block> block;
  std::vector<D> heap;

  // Seed each slot from the point via `make`.  Returns storage valid until the
  // next seed on this workspace.
  //
  // Not const: this fills the scratch.  Not noexcept: growth may allocate.
  // regular_invocable, not invocable: the heap path hands `make` to
  // std::ranges::transform, which may call it once per element in any order and
  // expects the same seed each time.
  template <std::regular_invocable<double> Make>
    requires std::convertible_to<std::invoke_result_t<Make &, double>, D>
  [[nodiscard]] D *seed_with(const std::span<const double> x, Make make) {
    if constexpr (inline_capacity > 0) {
      if (x.size() <= inline_capacity) {
        // Raw storage, so a slot may hold no object yet: each seed is a
        // construction rather than an assignment.  D is trivially destructible,
        // so whatever a previous call left in the block simply ends.
        D *const dof = reinterpret_cast<D *>(block.storage);
        for (std::size_t k = 0; k < x.size(); ++k) {
          std::construct_at(dof + k, make(x[k]));
        }
        return dof;
      }
    }
    // Grow-only, and resize() rather than a capacity check: the vector's size is
    // what a later .data() read is entitled to.  These slots already hold live
    // objects, so seeding them is plain assignment.
    if (heap.size() < x.size()) {
      heap.resize(x.size());
    }
    std::ranges::transform(x, heap.begin(), make);
    return heap.data();
  }
  // `D{v}` lifts a scalar to a zero-derivative dual at any nesting depth.
  [[nodiscard]] D *seed(const std::span<const double> x) {
    return seed_with(x, [](const double v) { return D{v}; });
  }
};

using GradientWorkspace = SweepWorkspace<dual>;
using HessianWorkspace = SweepWorkspace<dual2nd>;

} // namespace diff
