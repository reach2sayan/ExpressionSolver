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

// An energy called on the driver's seed buffer.  D is `dual` for the
// first-derivative sweep and `dual2nd` for the Hessian.
template <typename F, typename D>
concept CEnergyOf =
    std::invocable<F &, std::span<const D>> &&
    std::convertible_to<std::invoke_result_t<F &, std::span<const D>>, D>;

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

// f(x), n Jacobian entries, and the n*n Hessian, row-major; hessian_view()
// is how (i, j) is spelled.
struct HessianOwned {
  double value = 0.0;
  std::unique_ptr<double[]> jacobian;
  std::unique_ptr<double[]> hessian;
  std::size_t arity = 0;

  [[nodiscard]] constexpr md::mdspan<double, md::dextents<std::size_t, 2>>
  hessian_view() noexcept {
    return {hessian.get(), md::dextents<std::size_t, 2>{arity, arity}};
  }
  [[nodiscard]] constexpr md::mdspan<const double, md::dextents<std::size_t, 2>>
  hessian_view() const noexcept {
    return {hessian.get(), md::dextents<std::size_t, 2>{arity, arity}};
  }
};

// Compile-time extent: no allocation, and constant-evaluable.
template <std::size_t N> struct HessianStatic {
  double value = 0.0;
  std::array<double, N> jacobian{};
  std::array<double, N * N> hessian{};
  static constexpr std::size_t arity = N;

  [[nodiscard]] constexpr md::mdspan<double, md::extents<std::size_t, N, N>>
  hessian_view() noexcept {
    return md::mdspan<double, md::extents<std::size_t, N, N>>{hessian.data()};
  }
  [[nodiscard]] constexpr md::mdspan<const double,
                                     md::extents<std::size_t, N, N>>
  hessian_view() const noexcept {
    return md::mdspan<const double, md::extents<std::size_t, N, N>>{
        hessian.data()};
  }
};

namespace detail {
// Average each mirrored pair: independent sweeps differ in the last ULP.
constexpr void
symmetrize(const md::mdspan<double, md::dextents<std::size_t, 2>> m) noexcept {
  const std::size_t n = m.extent(0);
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

  // Storage valid until the next seed on this workspace.  assign(), not
  // resize() then transform: the second writes every element twice.
  [[nodiscard]] std::span<D> seed(const std::span<const double> x) {
    const auto seeded =
        x | std::views::transform([](const double v) { return D{v}; });
    dof.assign(std::ranges::begin(seeded), std::ranges::end(seeded));
    return std::span<D>{dof};
  }
};

} // namespace ddx::impl
