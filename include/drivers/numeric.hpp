#pragma once

// The drivers for a runtime callable: seed a dual into a scratch buffer, hand
// the callable a pointer to it, read the derivative back out.  gradient sweeps
// once per active variable, hessian once per probe pair.

#include "drivers/common.hpp"
#include "util/scope_guard.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
#include <vector>

namespace diff::impl {

namespace detail {
// Over storage the caller has already seeded.
template <CEnergyOf<dual> F, CIndexRange R>
constexpr void gradient_into(F &&f, dual *const dof, R &&active,
                             const std::span<double> out) {
  for (auto &&[slot, dof_index] : std::views::zip(out, active)) {
    const auto seed = scoped_seed<1.0>(dof[dof_index].deriv());
    slot = f(dof).deriv();
  }
}
} // namespace detail

// Writing form: nothing allocates once `ws` has been used once.
template <CEnergyOf<dual> F>
void gradient(F &&f, const std::span<const double> x,
              CIndexRange auto &&active, GradientWorkspace &ws,
              const std::span<double> out) {
  detail::gradient_into(static_cast<F &&>(f), ws.seed(x),
                        static_cast<decltype(active) &&>(active), out);
}

// Owning form: a local workspace.
template <CEnergyOf<dual> F>
std::vector<double> gradient(F &&f, const std::span<const double> x,
                             CIndexRange auto &&active) {
  GradientWorkspace ws;
  std::vector<double> g(std::ranges::size(active));
  gradient(static_cast<F &&>(f), x,
           static_cast<decltype(active) &&>(active), ws, std::span<double>{g});
  return g;
}

// Convenience: differentiate every variable.
template <CEnergyOf<dual> F>
std::vector<double> gradient(F &&f, const std::span<const double> x) {
  return gradient(static_cast<F &&>(f), x, detail::all_indices(x.size()));
}

namespace detail {

// Value, gradient and symmetric Hessian at x w.r.t. `active`: an O(m^2)
// forward-over-forward sweep on dual2nd.  Probe (i, j) seeds active[i] in the
// outer derivative slot and active[j] in the inner one, so f returns
// ((f, df/dx_j), (df/dx_i, d2f/dx_i dx_j)).  Only those two scalars move per
// probe, so they are toggled in place.  Writes every cell of both outputs.
//
// Each guard holds a bare double, never the enclosing dual2nd: when ai == aj
// they name two scalars of the same number and a guard over it would clobber
// the sibling.
template <CEnergyOf<dual2nd> F, CIndexRange R>
constexpr double hessian_into(F &&f, dual2nd *const dof, R &&active,
                                     double *const grad_out,
                                     double *const hess_out) {
  const std::size_t m = std::ranges::size(active);
  double value{};
  for (std::size_t j = 0; j < m; ++j) {
    const std::size_t aj = active[j];
    // Inner seed e_j is constant across the i-loop.
    const auto inner_seed = scoped_seed<1.0>(dof[aj].value().deriv());
    for (std::size_t i = 0; i <= j; ++i) {
      const std::size_t ai = active[i];

      // When ai == aj this lands in aj's other slot.
      const auto outer_seed = scoped_seed<1.0>(dof[ai].deriv().value());

      const dual2nd r = f(dof);
      const auto &[A, B] = r;   // value-component, outer-derivative component
      const auto &[a0, a1] = A; // f(x), df/dx_j
      const auto &[b0, b1] = B; // df/dx_i, d2f/dx_i dx_j

      value = a0;
      grad_out[i] = b0;
      grad_out[j] = a1;
      hess_out[i * m + j] = b1;
      hess_out[j * m + i] = b1;
    }
  }
  return value;
}

// Writing form: nothing allocates once the buffers and scratch are warm.
template <CEnergyOf<dual2nd> F>
double hessian(F &&f, const std::span<const double> x,
                      CIndexRange auto &&active, HessianWorkspace &ws,
                      const std::span<double> grad_out,
                      const std::span<double> hess_out) {
  return hessian_into(static_cast<F &&>(f), ws.seed(x),
                             static_cast<decltype(active) &&>(active),
                             grad_out.data(), hess_out.data());
}

// Compile-time arity: std::array throughout, so this path allocates nothing.
template <std::size_t N, CEnergyOf<dual2nd> F>
HessianStatic<N> hessian_static(F &&f, const std::span<const double> x) {
  std::array<dual2nd, N> dof{};
  std::ranges::transform(x | std::views::take(N), dof.begin(),
                         [](const double v) { return dual2nd{v}; });
  HessianStatic<N> out{};
  std::get<0>(out) =
      hessian_into(static_cast<F &&>(f), dof.data(), all_indices(N),
                          std::get<1>(out).data(), std::get<2>(out).data());
  return out;
}

// Owning form.
template <CEnergyOf<dual2nd> F>
HessianOwned hessian(F &&f, const std::span<const double> x,
                            CIndexRange auto &&active) {
  const std::size_t m = std::ranges::size(active);
  HessianWorkspace ws;
  auto grad = raw_buffer(m);
  auto hess = raw_buffer(m * m);
  const double value = hessian_into(
      static_cast<F &&>(f), ws.seed(x),
      static_cast<decltype(active) &&>(active), grad.get(), hess.get());
  return {value, std::move(grad), std::move(hess), m};
}

// Convenience: differentiate every variable.
template <CEnergyOf<dual2nd> F>
HessianOwned hessian(F &&f, const std::span<const double> x) {
  return hessian(static_cast<F &&>(f), x, all_indices(x.size()));
}

} // namespace detail

} // namespace diff::impl
