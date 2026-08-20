#pragma once

// The drivers that differentiate a *runtime callable* -- an energy lambda the
// library cannot see inside.  Both work the same way: seed a dual into a
// scratch buffer, hand the callable a pointer to it, read the derivative back
// out.  `gradient` seeds `dual` and sweeps once per active variable; `hessian`
// seeds `dual2nd` and sweeps once per probe pair.
//
// The graph drivers, which can see the expression and sweep it backward, are in
// `symbolic.hpp`; `hessian.hpp` chooses between the two.

#include "drivers/common.hpp"
#include "util/scope_guard.hpp" // scoped_value — RAII for the per-probe seed toggle

#include <algorithm>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
#include <vector>

namespace diff {

namespace detail {
// The gradient sweep itself, over storage the caller has already seeded.
template <CEnergyOf<dual> F, CIndexRange R>
constexpr void gradient_into(F &&f, dual *const dof, R &&active,
                             const std::span<double> out) {
  // One seeded pass per active variable
  for (auto &&[slot, dof_index] : std::views::zip(out, active)) {
    const auto seed = scoped_seed<1.0>(dof[dof_index].deriv());
    slot = f(dof).deriv();
  }
}
} // namespace detail

// Writing form: the caller owns the output, the library only fills it.  Nothing
// here allocates once `ws` has been used once.
template <CEnergyOf<dual> F>
void gradient(F &&f, const std::span<const double> x,
              CIndexRange auto &&active, GradientWorkspace &ws,
              const std::span<double> out) {
  detail::gradient_into(static_cast<F &&>(f), ws.seed(x),
                        static_cast<decltype(active) &&>(active), out);
}

// Owning forms, unchanged in signature.  They keep a local workspace, so a
// one-shot call costs one scratch allocation plus the returned vector.
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

// Value, gradient, and (symmetric) Hessian of f at x, w.r.t. `active`.
//
// Scalar O(m^2) forward-over-forward driver on dual2nd (= Dual<Dual<double>>).
// For a probe pair (i, j) we seed variable active[i] in the outer derivative
// slot and active[j] in the inner derivative slot; evaluating f then yields, in
// the result D = ((A0,A1),(B0,B1)):
//   A0 = f(x),  B0 = df/dx_i,  A1 = df/dx_j,  B1 = d2f/dx_i dx_j.
// The sweep proper, over storage the caller has already seeded and buffers the
// caller already owns.  Every entry point below funnels through this, so the
// probe loop -- and the seeding convention it encodes -- exists exactly once.
//
// Writes every cell of both outputs: hessian[i*m+j] and its mirror for all
// i <= j is the whole matrix, and gradient[k] is covered as i and j sweep.  That
// is what lets every caller hand it uninitialised storage.  Returns f(x).
//
// Of a dual2nd's four scalars [value.value, value.deriv, deriv.value,
// deriv.deriv] only the two first-order seed slots ever move per probe:
// value.deriv carries e_j (inner, d/dx_j) and deriv.value carries e_i (outer,
// d/dx_i).  value.value == x[k] and the second-order seed deriv.deriv == 0 are
// loop-invariant, so the sweep toggles the two derivative scalars in place
// rather than reconstructing the whole dual2nd on every seed and reset.
//
// Each toggle is a scoped_value whose scope IS the span the seed covers.  Both
// guards hold a bare double, never the enclosing dual2nd: when ai == aj they
// name two different scalars of the same number, and a guard over the number
// would clobber its sibling.
template <CEnergyOf<dual2nd> F, CIndexRange R>
constexpr double hessian_into(F &&f, dual2nd *const dof, R &&active,
                                     double *const grad_out,
                                     double *const hess_out) {
  const std::size_t m = std::ranges::size(active);
  double value{};
  for (std::size_t j = 0; j < m; ++j) {
    const std::size_t aj = active[j];
    // Inner seed e_j is constant across the whole i-loop: held by this guard.
    const auto inner_seed = scoped_seed<1.0>(dof[aj].value().deriv());
    for (std::size_t i = 0; i <= j; ++i) {
      const std::size_t ai = active[i];

      // Outer seed e_i on ai (when ai == aj this lands in aj's other slot).
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

// Writing form: the caller owns both output buffers and the scratch, so nothing
// here allocates once they have been used once.  This is the only entry point
// that writes into memory the library did not create, and it says so in the
// signature.  Returns f(x).
template <CEnergyOf<dual2nd> F>
double hessian(F &&f, const std::span<const double> x,
                      CIndexRange auto &&active, HessianWorkspace &ws,
                      const std::span<double> grad_out,
                      const std::span<double> hess_out) {
  return hessian_into(static_cast<F &&>(f), ws.seed(x),
                             static_cast<decltype(active) &&>(active),
                             grad_out.data(), hess_out.data());
}

// Compile-time arity: the expression entry points know how many symbols there
// are, so both the seeded-variable array and the result are std::array and this
// path allocates nothing at all.  N is the arity, which is also the extent -- an
// expression is differentiated with respect to all of its own symbols.
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

// Owning form: allocates the two result buffers, fills them, and hands them over.
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

} // namespace diff
