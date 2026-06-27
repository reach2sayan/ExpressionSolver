#pragma once

#include "dual.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

namespace diff {

namespace detail {
inline std::vector<std::size_t> iota_indices(std::size_t n) {
  std::vector<std::size_t> v(n);
  std::iota(v.begin(), v.end(), std::size_t{0});
  return v;
}
} // namespace detail

// Result of a second-order forward sweep over `active` variables.
struct HessianResult {
  double value{};                  // f(x)
  std::vector<double> gradient;    // size = active.size()
  std::vector<double> hessian;     // row-major, active.size() x active.size()

  std::size_t n() const noexcept { return gradient.size(); }
  double &h(std::size_t i, std::size_t j) noexcept {
    return hessian[i * n() + j];
  }
  double h(std::size_t i, std::size_t j) const noexcept {
    return hessian[i * n() + j];
  }
};

template <typename F>
std::vector<double> gradient(F &&f, std::span<const double> x,
                             std::span<const std::size_t> active) {
  const std::size_t n = x.size();
  const std::size_t m = active.size();
  std::vector<double> g(m, 0.0);

  // Build the seed pack once (all derivative parts zero); across passes only the
  // active[j] seed's derivative toggles 1->0, so flip that one entry per pass
  // instead of rebuilding all n duals every pass.
  std::vector<dual> dof(n);
  std::ranges::transform(x, dof.begin(), [](double v) { return dual{v, 0.0}; });

  for (std::size_t j = 0; j < m; ++j) {
    dof[active[j]].template get<1>() = 1.0;
    g[j] = f(dof.data()).template get<1>();
    dof[active[j]].template get<1>() = 0.0;
  }
  return g;
}

// Convenience: differentiate every variable.
template <typename F>
std::vector<double> gradient(F &&f, std::span<const double> x) {
  const auto all = detail::iota_indices(x.size());
  return gradient(static_cast<F &&>(f), x, all);
}

namespace detail {

// Value, gradient, and (symmetric) Hessian of f at x, w.r.t. `active`.
//
// Scalar O(m^2) forward-over-forward driver on dual2nd (= Dual<Dual<double>>).
// For a probe pair (i, j) we seed variable active[i] in the outer derivative
// slot and active[j] in the inner derivative slot; evaluating f then yields, in
// the result D = ((A0,A1),(B0,B1)):
//   A0 = f(x),  B0 = df/dx_i,  A1 = df/dx_j,  B1 = d2f/dx_i dx_j.
//
// This is the fallback/reference path; the public hessian() in vforward_driver.hpp
// dispatches to the O(m) vector-forward driver when m <= kVForwardN.
template <typename F>
HessianResult hessian_scalar(F &&f, std::span<const double> x,
                             std::span<const std::size_t> active) {
  const std::size_t n = x.size();
  const std::size_t m = active.size();

  HessianResult res;
  res.gradient.assign(m, 0.0);
  res.hessian.assign(m * m, 0.0);

  std::vector<dual2nd> dof(n);
  using Inner = Dual<double>;

  // Seed once to the zero-derivative base; each (i,j) pair then touches only the
  // (at most) two active dofs and resets them afterwards — O(m^2) seeding, not
  // O(m^2 * n) (autodiff seeds the same way).  Non-active dofs stay at base.
  for (std::size_t k = 0; k < n; ++k) {
    dof[k] = dual2nd{Inner{x[k], 0.0}, Inner{0.0, 0.0}};
  }

  for (std::size_t j = 0; j < m; ++j) {
    const std::size_t aj = active[j];
    for (std::size_t i = 0; i <= j; ++i) {
      const std::size_t ai = active[i];

      // Seed: inner slot = e_j on aj, outer slot = e_i on ai (i==j coincides).
      dof[aj] = dual2nd{Inner{x[aj], 1.0}, Inner{(ai == aj) ? 1.0 : 0.0, 0.0}};
      if (ai != aj) {
        dof[ai] = dual2nd{Inner{x[ai], 0.0}, Inner{1.0, 0.0}};
      }

      const dual2nd r = f(dof.data());
      const auto &[A, B] = r;   // value-component, outer-derivative component
      const auto &[a0, a1] = A; // f(x), df/dx_j
      const auto &[b0, b1] = B; // df/dx_i, d2f/dx_i dx_j

      res.value = a0;
      res.gradient[i] = b0;
      res.gradient[j] = a1;
      res.hessian[i * m + j] = b1;
      res.hessian[j * m + i] = b1;

      // Reset the touched dofs back to base for the next pair.
      dof[aj] = dual2nd{Inner{x[aj], 0.0}, Inner{0.0, 0.0}};
      if (ai != aj) {
        dof[ai] = dual2nd{Inner{x[ai], 0.0}, Inner{0.0, 0.0}};
      }
    }
  }
  return res;
}

// Convenience: differentiate every variable.
template <typename F>
HessianResult hessian_scalar(F &&f, std::span<const double> x) {
  const auto all = iota_indices(x.size());
  return hessian_scalar(static_cast<F &&>(f), x, all);
}

} // namespace detail

} // namespace diff
