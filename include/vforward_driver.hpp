#pragma once

#include "dual.hpp"
#include "forward_driver.hpp" // HessianResult + scalar hessian() fallback
#include "vector_dual.hpp"

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <span>
#include <vector>

namespace diff {

namespace detail {

// Vector-forward Hessian for a known compile-time capacity N >= m.
//
// One forward sweep per active variable i (m sweeps total).  The *value* level
// of every dof carries the full identity tangent pack (lane t ==
// d/dx_active[t]), so each sweep also yields the gradient; the *outer*
// derivative level carries the scalar seed e_i.  After f(dof):
// r.get<0>() : value-level VectorDual -> value = f(x), grad[t] = df/dx_t
// r.get<1>() : outer-deriv VectorDual -> grad[t] = d2f/dx_i dx_t (Hessianrow i)

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
  std::ranges::transform(x, base.begin(), [](double v) { return V{v}; });
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
    const auto &[A, B] = r; // value & outer-derivative component (no copy)
    if (i == 0) {
      res.value = A.value;
      std::ranges::copy(A.grad | std::views::take(m), res.gradient.begin());
    }
    std::ranges::copy(B.grad | std::views::take(m),
                      res.hessian.begin() + i * m);
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

// Compile-time bucket ladder 1,2,4,...,kVForwardN.  Walk it at runtime and
// dispatch the smallest capacity N >= m, so a small active set runs narrow
// lanes instead of always paying for the full kVForwardN-wide pack.  Only the
// buckets on this ladder are instantiated (log2(kVForwardN)+1 of them).
template <std::size_t N, typename F>
HessianResult vforward_pick(std::size_t m, F &&f, std::span<const double> x,
                            std::span<const std::size_t> active) {
  if constexpr (N >= kVForwardN) {
    // Top of the ladder: caller already guaranteed m <= kVForwardN.
    return hessian_vforward_impl<kVForwardN>(static_cast<F &&>(f), x, active);
  } else {
    if (m <= N) {
      return hessian_vforward_impl<N>(static_cast<F &&>(f), x, active);
    }
    constexpr std::size_t Next = (N * 2 < kVForwardN) ? N * 2 : kVForwardN;
    return vforward_pick<Next>(m, static_cast<F &&>(f), x, active);
  }
}

} // namespace detail

// Value, gradient and (symmetric) Hessian of f at x w.r.t. `active`, via
// vector-forward-over-forward mode.  Drop-in for hessian(): identical
// signature and HessianResult, but O(m) sweeps of the energy lambda instead of
// O(m^2).  The lane capacity is bucketed to the smallest power of two >= m
// (capped at kVForwardN); m > kVForwardN falls back to the scalar O(m^2)
// driver.
template <typename F>
HessianResult hessian_vforward(F &&f, std::span<const double> x,
                               std::span<const std::size_t> active) {
  const std::size_t m = active.size();
  if (m == 0 || m > kVForwardN) {
    // scalar O(m^2) fallback
    return detail::hessian_scalar(static_cast<F &&>(f), x, active);
  }
  return detail::vforward_pick<1>(m, static_cast<F &&>(f), x, active);
}

// Convenience: differentiate every variable.
template <typename F>
HessianResult hessian_vforward(F &&f, std::span<const double> x) {
  const auto all = detail::iota_indices(x.size());
  return hessian_vforward(static_cast<F &&>(f), x, all);
}

template <typename F>
HessianResult hessian(F &&f, std::span<const double> x,
                      std::span<const std::size_t> active) {
  return hessian_vforward(static_cast<F &&>(f), x, active);
}
template <typename F> HessianResult hessian(F &&f, std::span<const double> x) {
  return hessian_vforward(static_cast<F &&>(f), x);
}

} // namespace diff
