#pragma once

#include "dual.hpp"
#include "forward_driver.hpp" // HessianResult + scalar hessian() fallback
#include "vector_dual.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace diff {

// kVForwardN and dual_vforward live in vector_dual.hpp so consumers can pull in
// just the element type (for their explicit energy instantiation) without the
// driver's <span>/<vector> baggage.

namespace detail {

// Vector-forward Hessian for a known compile-time capacity N >= m.
//
// One forward sweep per active variable i (m sweeps total).  The *value* level
// of every dof carries the full identity tangent pack (lane t ==
// d/dx_active[t]), so each sweep also yields the gradient; the *outer*
// derivative level carries the scalar seed e_i.  After f(dof):
//   r.get<0>() : value-level VectorDual -> value = f(x), grad[t] = df/dx_t
//   r.get<1>() : outer-deriv VectorDual -> grad[t] = d2f/dx_i dx_t  (Hessian
//   row i)
// Cost: m sweeps of N-wide (SIMD) arithmetic vs the m(m+1)/2 scalar sweeps of
// hessian().  Transcendentals (log/exp) are evaluated m times, not m(m+1)/2.
template <std::size_t N, typename F>
HessianResult hessian_vforward_impl(F &&f, std::span<const double> x,
                                    std::span<const std::size_t> active) {
  const std::size_t n = x.size();
  const std::size_t m = active.size(); // caller guarantees 0 < m <= N
  using V = VectorDual<N>;
  using D = Dual<V>;

  HessianResult res;
  res.gradient.assign(m, 0.0);
  res.hessian.assign(m * m, 0.0);

  // Value-level seeds: identity tangents over `active`, identical every sweep.
  std::vector<V> base(n);
  for (std::size_t k = 0; k < n; ++k) {
    base[k] = V{x[k]};
  }
  for (std::size_t t = 0; t < m; ++t) {
    base[active[t]].grad[t] = 1.0;
  }

  std::vector<D> dof(n);
  for (std::size_t i = 0; i < m; ++i) {
    const std::size_t ai = active[i];
    for (std::size_t k = 0; k < n; ++k) {
      // outer-derivative slot = scalar seed e_i (value 1 at k==ai, grad 0)
      dof[k] = D{base[k], V{(k == ai) ? double{1} : double{0}}};
    }

    const D r = f(dof.data());
    const V A = r.template get<0>(); // value component
    const V B = r.template get<1>(); // outer-derivative component
    if (i == 0) {
      res.value = A.value;
      for (std::size_t t = 0; t < m; ++t) {
        res.gradient[t] = A.grad[t];
      }
    }
    for (std::size_t t = 0; t < m; ++t) {
      res.hessian[i * m + t] = B.grad[t];
    }
  }

  // Each row was computed by an independent sweep, so H(i,j) and H(j,i) can
  // differ in the last ULP.  Symmetrize to honour the same exactly-symmetric
  // contract the scalar hessian() returns (downstream LDLT/eigensolvers expect
  // it); analytically H is symmetric, so averaging only removes FP noise.
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = i + 1; j < m; ++j) {
      const double s = 0.5 * (res.hessian[i * m + j] + res.hessian[j * m + i]);
      res.hessian[i * m + j] = s;
      res.hessian[j * m + i] = s;
    }
  }
  return res;
}

} // namespace detail

// Value, gradient and (symmetric) Hessian of f at x w.r.t. `active`, via
// vector-forward-over-forward mode.  Drop-in for hessian(): identical
// signature and HessianResult, but O(m) sweeps of the energy lambda instead of
// O(m^2).  `f` must be callable with any forward-dual element type (write the
// call-site lambda as `[&](const auto *dof){ ... }`): the active branch
// evaluates it at diff::dual_vforward, while m == 0 or m > kVForwardN falls
// back to the scalar driver at diff::dual2nd.
template <typename F>
HessianResult hessian_vforward(F &&f, std::span<const double> x,
                               std::span<const std::size_t> active) {
  const std::size_t m = active.size();
  if (m == 0 || m > kVForwardN) {
    return hessian(static_cast<F &&>(f), x, active); // scalar O(m^2) fallback
  }
  return detail::hessian_vforward_impl<kVForwardN>(static_cast<F &&>(f), x,
                                                   active);
}

// Convenience: differentiate every variable.
template <typename F>
HessianResult hessian_vforward(F &&f, std::span<const double> x) {
  const auto all = detail::iota_indices(x.size());
  return hessian_vforward(static_cast<F &&>(f), x, all);
}

} // namespace diff
