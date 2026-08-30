#pragma once

// The drivers for a runtime callable: seed a dual into a scratch buffer, hand
// the callable a view of it, read the derivative back out.  jacobian sweeps
// once per active variable, hessian once per probe pair.

#include "dual/workspace.hpp"
#include "util/config.hpp"
#include "util/scope_guard.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace ddx::impl {

namespace detail {
// Over storage the caller has already seeded.
constexpr void jacobian_into(CEnergyOf<dual> auto &&f,
                             const std::span<dual> dof,
                             CIndexRange auto &&active,
                             const std::span<double> out) {
  assert(out.size() == std::ranges::size(active));
  const std::span<const dual> point = dof;
  for (auto &&[slot, dof_index] : std::views::zip(out, active)) {
    const auto seed = scoped_seed<1.0>(dof[dof_index].deriv());
    slot = f(point).deriv();
  }
}
} // namespace detail

// Writing form: nothing allocates once `ws` has been used once.
void jacobian(CEnergyOf<dual> auto &&f, const std::span<const double> x,
              CIndexRange auto &&active, JacobianWorkspace &ws,
              const std::span<double> out) {
  detail::jacobian_into(DDX_FWD(f), ws.seed(x), DDX_FWD(active), out);
}

// Owning form: a local workspace.
std::vector<double> jacobian(CEnergyOf<dual> auto &&f,
                             const std::span<const double> x,
                             CIndexRange auto &&active) {
  JacobianWorkspace ws;
  std::vector<double> g(std::ranges::size(active));
  jacobian(DDX_FWD(f), x, DDX_FWD(active), ws, std::span<double>{g});
  return g;
}

// Convenience: differentiate every variable.
std::vector<double> jacobian(CEnergyOf<dual> auto &&f,
                             const std::span<const double> x) {
  return jacobian(DDX_FWD(f), x, detail::all_indices(x.size()));
}

namespace detail {

// An O(m^2) forward-over-forward sweep on dual2nd.  Probe (i, j) seeds
// active[i] in the outer derivative slot and active[j] in the inner one, so f
// returns ((f, df/dx_j), (df/dx_i, d2f/dx_i dx_j)); only those two scalars move
// per probe.  Each guard holds a bare double, never the enclosing dual2nd: at
// ai == aj they name two scalars of the same number.
constexpr double
hessian_into(CEnergyOf<dual2nd> auto &&f, const std::span<dual2nd> dof,
             CIndexRange auto &&active, const std::span<double> grad,
             const md::mdspan<double, md::dextents<std::size_t, 2>> hess) {
  [[maybe_unused]] const std::size_t m = std::ranges::size(active);
  assert(grad.size() == m && hess.extent(0) == m && hess.extent(1) == m);
  const std::span<const dual2nd> point = dof;
  double value{};
  for (const auto [index, aj] : active | std::views::enumerate) {
    const auto j = static_cast<std::size_t>(index);
    // Inner seed e_j is constant across the i-loop.
    const auto inner_seed = scoped_seed<1.0>(dof[aj].value().deriv());
    for (const auto [inner, ai] :
         (active | std::views::take(index + 1)) | std::views::enumerate) {
      const auto i = static_cast<std::size_t>(inner);

      // When ai == aj this lands in aj's other slot.
      const auto outer_seed = scoped_seed<1.0>(dof[ai].deriv().value());

      const dual2nd r = f(point);
      const auto &[A, B] = r;   // value-component, outer-derivative component
      const auto &[a0, a1] = A; // f(x), df/dx_j
      const auto &[b0, b1] = B; // df/dx_i, d2f/dx_i dx_j

      value = a0;
      grad[i] = b0;
      grad[j] = a1;
      hess[i, j] = b1;
      hess[j, i] = b1;
    }
  }
  return value;
}

// Writing form: nothing allocates once the buffers and scratch are warm.
double
hessian(CEnergyOf<dual2nd> auto &&f, const std::span<const double> x,
        CIndexRange auto &&active, HessianWorkspace &ws,
        const std::span<double> grad_out,
        const md::mdspan<double, md::dextents<std::size_t, 2>> hess_out) {
  return hessian_into(DDX_FWD(f), ws.seed(x), DDX_FWD(active), grad_out,
                      hess_out);
}

// std::array throughout, so this is the one Hessian driver a constant
// evaluation can run.
template <std::size_t N>
constexpr HessianStatic<N> hessian_static(CEnergyOf<dual2nd> auto &&f,
                                          const std::span<const double> x) {
  std::array<dual2nd, N> dof{};
  std::ranges::transform(x | std::views::take(N), dof.begin(),
                         [](const double v) { return dual2nd{v}; });
  HessianStatic<N> out{};
  out.value = hessian_into(DDX_FWD(f), std::span<dual2nd>{dof}, all_indices(N),
                           std::span<double>{out.jacobian}, out.hessian_view());
  return out;
}

// Owning form.
HessianOwned hessian(CEnergyOf<dual2nd> auto &&f,
                     const std::span<const double> x,
                     CIndexRange auto &&active) {
  const std::size_t m = std::ranges::size(active);
  HessianWorkspace ws;
  // Uninitialised: the sweep writes every cell.
  HessianOwned out{.jacobian = std::make_unique_for_overwrite<double[]>(m),
                   .hessian = std::make_unique_for_overwrite<double[]>(m * m),
                   .arity = m};
  out.value = hessian_into(DDX_FWD(f), ws.seed(x), DDX_FWD(active),
                           std::span<double>{out.jacobian.get(), m},
                           out.hessian_view());
  return out;
}

// Convenience: differentiate every variable.
HessianOwned hessian(CEnergyOf<dual2nd> auto &&f,
                     const std::span<const double> x) {
  return hessian(DDX_FWD(f), x, all_indices(x.size()));
}

} // namespace detail

} // namespace ddx::impl
