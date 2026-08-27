#pragma once

#include "md/md.hpp"
#include "ops/scalar.hpp"

#include <boost/container/small_vector.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <memory>
#include <numeric> // std::midpoint
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

namespace ddx::impl {

// An energy called with a pointer into the driver's seed buffer.  D is `dual`
// for the first-derivative sweep and `dual2nd` for the Hessian.
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

// f(x), n Jacobian entries, and n*n row-major Hessian entries: (i, j) is
// hessian[i * n + j].  Named members rather than a tuple, `arity` being what
// tells a caller how to index `hessian`.
struct HessianOwned {
  double value = 0.0;
  std::unique_ptr<double[]> jacobian;
  std::unique_ptr<double[]> hessian;
  std::size_t arity = 0;
};

// Compile-time extent: no allocation, and constant-evaluable.
template <std::size_t N> struct HessianStatic {
  double value = 0.0;
  std::array<double, N> jacobian{};
  std::array<double, N * N> hessian{};
  static constexpr std::size_t arity = N;
};

namespace detail {
// Average each mirrored pair: independent sweeps differ in the last ULP.  An
// mdspan rather than a pointer and a stride, so `h[i, j]` is not an index
// expression the reader has to check.
constexpr void symmetrize(std::span<double> h, const std::size_t n) noexcept {
  const md::mdspan m{h.data(), md::dextents<std::size_t, 2>{n, n}};
  for (const std::size_t i : std::views::iota(0uz, n)) {
    for (const std::size_t j : std::views::iota(i + 1, n)) {
      const double s = std::midpoint(m[i, j], m[j, i]);
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
  std::ranges::copy(h.jacobian, grad.get());
  std::ranges::copy(h.hessian, hess.get());
  return {.value = h.value,
          .jacobian = std::move(grad),
          .hessian = std::move(hess),
          .arity = N};
}

// Uninitialised: the sweep writes every cell.
[[nodiscard]] inline std::unique_ptr<double[]> raw_buffer(std::size_t k) {
  return std::make_unique_for_overwrite<double[]>(k);
}
} // namespace detail

// Caller-owned scratch for a sweep's seeded variables, reused across a loop
// over many points.

template <Numeric D> struct SweepWorkspace {
  // A byte budget, not an element count: D may be much wider than dual2nd.
  // Below the floor the inline buffer is dropped rather than shrunk -- an array
  // member costs a stack-protector canary in every function holding one, and a
  // small_vector<D, 0> has no array member to pay for.
  static constexpr std::size_t inline_bytes = 512;
  static constexpr std::size_t inline_floor = 8;
  static constexpr std::size_t inline_capacity =
      (inline_bytes / sizeof(D) >= inline_floor) ? inline_bytes / sizeof(D) : 0;

  boost::container::small_vector<D, inline_capacity> dof;

  [[nodiscard]] D *seed(const std::span<const double> x) {
    return seed_with(x, [](const double v) { return D{v}; });
  }

private:
  // Storage valid until the next seed on this workspace.
  template <std::regular_invocable<double> Make>
    requires std::convertible_to<std::invoke_result_t<Make &, double>, D>
  [[nodiscard]] D *seed_with(const std::span<const double> x, Make make) {
    // assign(), not resize() then transform: the second writes every element
    // twice, once value-initialised and once with the seed.
    const auto seeded = x | std::views::transform(make);
    dof.assign(std::ranges::begin(seeded), std::ranges::end(seeded));
    return dof.data();
  }
};

} // namespace ddx::impl
