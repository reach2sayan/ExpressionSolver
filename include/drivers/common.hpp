#pragma once

#include "dual/dual.hpp"
#include "md/md.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

namespace ddx::impl {

// An energy called with a pointer into the driver's seed buffer.  D is `dual`
// for the gradient sweep, `dual2nd` for the Hessian.
template <typename F, typename D>
concept CEnergyOf =
    std::invocable<F &, const D *> &&
    std::convertible_to<std::invoke_result_t<F &, const D *>, D>;

// Indexed and sized, nothing else -- so views::iota models it too.
template <typename R>
concept CIndexRange =
    std::ranges::random_access_range<R> && std::ranges::sized_range<R> &&
    std::convertible_to<std::ranges::range_value_t<R>, std::size_t>;

namespace detail {
// "all variables", unmaterialised.
[[nodiscard]] constexpr auto all_indices(std::size_t n) noexcept {
  return std::views::iota(std::size_t{0}, n);
}
} // namespace detail

// What a second-order sweep hands back: f(x), n gradient entries, and n*n
// row-major Hessian entries -- (i, j) is hessian[i * n + j].
//
// Named members rather than a tuple: `arity` is the only thing telling a caller
// how to index `hessian`, and std::get<3> is not a spelling that says so.  The
// two shapes carry the same names so that code generic over both -- the tests
// are -- reads one way.
struct HessianOwned {
  double value = 0.0;
  std::unique_ptr<double[]> gradient;
  std::unique_ptr<double[]> hessian;
  std::size_t arity = 0;
};

// Compile-time extent: no allocation, and constant-evaluable.
template <std::size_t N> struct HessianStatic {
  double value = 0.0;
  std::array<double, N> gradient{};
  std::array<double, N * N> hessian{};
  static constexpr std::size_t arity = N;
};

namespace detail {
// Average each mirrored pair: independent sweeps differ in the last ULP.
//
// An mdspan rather than a pointer and a stride: the caller's buffer is square
// and row-major, and saying so once is what keeps `h[i, j]` from being an index
// expression the reader has to check.
constexpr void symmetrize(std::span<double> h, const std::size_t n) noexcept {
  const md::mdspan m{h.data(), md::dextents<std::size_t, 2>{n, n}};
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double s = 0.5 * (m[i, j] + m[j, i]);
      m[i, j] = s;
      m[j, i] = s;
    }
  }
}

// Only for the subset-taking hessian(), which can reach both shapes.
template <std::size_t N>
[[nodiscard]] inline HessianOwned to_owned(const HessianStatic<N> &h) {
  auto grad = std::make_unique_for_overwrite<double[]>(N);
  auto hess = std::make_unique_for_overwrite<double[]>(N * N);
  std::ranges::copy(h.gradient, grad.get());
  std::ranges::copy(h.hessian, hess.get());
  return {.value = h.value,
          .gradient = std::move(grad),
          .hessian = std::move(hess),
          .arity = N};
}

// Uninitialised: the sweep writes every cell.
[[nodiscard]] inline std::unique_ptr<double[]> raw_buffer(std::size_t k) {
  return std::make_unique_for_overwrite<double[]>(k);
}
} // namespace detail

// Caller-owned scratch for a sweep's seeded variables: reused across a loop
// over many points, so it allocates once.

namespace detail {
// An array member costs a stack-protector canary in every function holding the
// workspace, so a D too wide to ever fit a point gets no block at all.
template <Numeric D, std::size_t Bytes> struct inline_block {
  alignas(D) std::byte storage[Bytes];
};
struct no_inline_block {};
} // namespace detail

template <Numeric D> struct SweepWorkspace {
  static_assert(std::is_trivially_destructible_v<D>,
                "the inline block is reused and abandoned without destruction");

  // A byte budget, not an element count: D may be much wider than dual2nd.
  static constexpr std::size_t inline_bytes = 512;
  static constexpr std::size_t inline_floor = 8;
  static constexpr std::size_t inline_capacity =
      (inline_bytes / sizeof(D) >= inline_floor) ? inline_bytes / sizeof(D) : 0;

  [[no_unique_address]] std::conditional_t<
      (inline_capacity > 0), detail::inline_block<D, inline_bytes>,
      detail::no_inline_block> block;
  std::vector<D> heap;

  // Storage valid until the next seed on this workspace.
  template <std::regular_invocable<double> Make>
    requires std::convertible_to<std::invoke_result_t<Make &, double>, D>
  [[nodiscard]] D *seed_with(const std::span<const double> x, Make make) {
    if constexpr (inline_capacity > 0) {
      if (x.size() <= inline_capacity) {
        // Raw storage: each seed constructs rather than assigns, which is
        // what uninitialized_ means here -- there is no D to assign over.
        D *const dof = reinterpret_cast<D *>(block.storage);
        std::ranges::uninitialized_copy(x | std::views::transform(make),
                                        std::span{dof, x.size()});
        return dof;
      }
    }
    // resize(), not reserve(): size is what a later .data() read may touch.
    if (heap.size() < x.size()) {
      heap.resize(x.size());
    }
    std::ranges::transform(x, heap.begin(), make);
    return heap.data();
  }
  [[nodiscard]] D *seed(const std::span<const double> x) {
    return seed_with(x, [](const double v) { return D{v}; });
  }
};

using GradientWorkspace = SweepWorkspace<dual>;
using HessianWorkspace = SweepWorkspace<dual2nd>;

} // namespace ddx::impl
