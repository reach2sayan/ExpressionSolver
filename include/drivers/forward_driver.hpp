#pragma once

#include "dual/dual.hpp"
#include "md/md.hpp"            // md::mdspan — the Hessian's 2D view
#include "util/scope_guard.hpp" // scoped_value — RAII for the per-probe seed toggle

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>

namespace diff {

// An energy in the shape every driver here hands one: it is called with a
// pointer into the driver's own seed buffer and returns a number of the same
// type.  The buffer's length is deliberately not part of the type — the
// driver owns it and the callable reads a window the caller has guaranteed
// (see SeededExprEnergy in seeded_energy.hpp).
//
// D is the number the driver seeds with, and it is what distinguishes the
// drivers from each other: `dual` for the gradient sweep, `dual2nd` for the
// scalar forward-over-forward Hessian, Dual<VectorDual<N>> for the
// vector-forward one.  Naming it here is what lets an energy that is not
// generic enough fail at its call site rather than inside a sweep.
template <typename F, typename D>
concept CEnergyOf =
    std::invocable<F &, const D *> &&
    std::convertible_to<std::invoke_result_t<F &, const D *>, D>;

namespace detail {
inline std::vector<std::size_t> iota_indices(std::size_t n) {
  std::vector<std::size_t> v(n);
  std::iota(v.begin(), v.end(), std::size_t{0});
  return v;
}
} // namespace detail

// Result of a second-order forward sweep over `active` variables.
//
// Owning: `gradient` and `hessian` are this object's own buffers, not views
// into the driver or into the point that produced them.  Everything the
// drivers hand back is owned by the caller, so a HessianResult can be stored,
// moved, or returned without regard to what is still alive.
struct HessianResult {
  double value{};               // f(x)
  std::vector<double> gradient; // size = active.size()
  std::vector<double> hessian;  // active.size() x active.size(), see matrix()

  // The extent is stored rather than recovered from gradient.size().  It is
  // the same number, but a Hessian's shape is not a fact about the gradient's
  // length, and every index below now goes through a mapping that has to be
  // handed one.
  std::size_t extent{};

  // The single place the buffers get their size, so the two vectors and the
  // extent cannot drift apart.
  void resize(std::size_t m) {
    extent = m;
    gradient.assign(m, 0.0);
    hessian.assign(m * m, 0.0);
  }

  [[nodiscard]] constexpr std::size_t n() const noexcept { return extent; }

  // The Hessian as what it actually is: a rank-2 row-major view, so a caller
  // gets extents and a mapping instead of being told in a comment.
  //
  // The drivers deliberately do NOT go through this.  Slicing their sweeps with
  // submdspan reads better but measured 8-16% slower on the HessExpr
  // benchmarks, and a Hessian sweep is the one place in this library where that
  // cost lands in a loop.  Here it is opt-in and costs nothing unless asked
  // for; h() likewise indexes the buffer directly rather than building a view
  // per call.
  [[nodiscard]] constexpr auto matrix() & noexcept {
    return md::mdspan<double, md::dextents<std::size_t, 2>>(hessian.data(),
                                                            extent, extent);
  }
  [[nodiscard]] constexpr auto matrix() const & noexcept {
    return md::mdspan<const double, md::dextents<std::size_t, 2>>(
        hessian.data(), extent, extent);
  }
  // A view onto a temporary's buffer would dangle the moment the full
  // expression ends, and unlike h() there is no copy-out to fall back on.
  auto matrix() && = delete;
  auto matrix() const && = delete;

  // Lvalue-only: this hands out a reference into `hessian`, so binding it to
  // the result of a hessian() call directly would outlive the buffer.  On a
  // temporary the const overload below is selected instead and copies out.
  [[nodiscard]] constexpr double &h(std::size_t i, std::size_t j) & noexcept {
    return hessian[i * extent + j];
  }
  [[nodiscard]] constexpr double h(std::size_t i,
                                   std::size_t j) const & noexcept {
    return hessian[i * extent + j];
  }
  [[nodiscard]] constexpr double h(std::size_t i,
                                   std::size_t j) const && noexcept {
    return hessian[i * extent + j];
  }

  // H[i, j] — the same thing h() does, spelled the way md_tensor and every
  // md::mdspan in this library spell it.
  //
  // It exists because `hessian()` and `hessian<DiffMode::Reverse>()` return
  // different types — this one and an md_tensor — and a caller switching
  // between them should not also have to switch subscript syntax.  Indexed
  // directly rather than through matrix(): building a view per access is pure
  // overhead, and this is the accessor a caller puts in a loop.
  [[nodiscard]] constexpr double &operator[](std::size_t i,
                                             std::size_t j) & noexcept {
    return hessian[i * extent + j];
  }
  [[nodiscard]] constexpr double operator[](std::size_t i,
                                            std::size_t j) const & noexcept {
    return hessian[i * extent + j];
  }
  [[nodiscard]] constexpr double operator[](std::size_t i,
                                            std::size_t j) const && noexcept {
    return hessian[i * extent + j];
  }

  // Average each (i,j)/(j,i) pair in place.
  //
  // A driver that fills the matrix one row (or column) at a time computes the
  // two halves in independent sweeps, so mirrored entries can differ in the
  // last ULP.  H is analytically symmetric, so averaging only removes that FP
  // noise — and it is what makes the exactly-symmetric contract every driver
  // advertises true of the buffer as well as of the maths.  Drivers whose
  // layout names each entry exactly once (the sparse path) have no pair to
  // average and do not call this.
  constexpr void symmetrize() noexcept {
    for (std::size_t i = 0; i < extent; ++i) {
      for (std::size_t j = i + 1; j < extent; ++j) {
        const double s = 0.5 * (hessian[i * extent + j] + //
                                hessian[j * extent + i]);
        hessian[i * extent + j] = s;
        hessian[j * extent + i] = s;
      }
    }
  }
};

template <CEnergyOf<dual> F>
std::vector<double> gradient(F &&f, std::span<const double> x,
                             std::span<const std::size_t> active) {
  const std::size_t n = x.size();
  const std::size_t m = active.size();
  std::vector<double> g(m, 0.0);
  std::vector<dual> dof(n);
  std::ranges::transform(x, dof.begin(), [](double v) { return dual{v, 0.0}; });

  // One seeded pass per active variable: the gradient slot and the dof it comes
  // from advance together.  The seed lives exactly as long as the guard, so it
  // is cleared before the next pass even if the energy throws.
  for (auto &&[slot, dof_index] : std::views::zip(g, active)) {
    const auto seed = scoped_seed<1.0>(dof[dof_index].deriv());
    slot = f(dof.data()).deriv();
  }
  return g;
}

template <CEnergyOf<dual> F>
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
template <CEnergyOf<dual2nd> F>
HessianResult hessian_scalar(F &&f, std::span<const double> x,
                             std::span<const std::size_t> active) {
  const std::size_t n = x.size();
  const std::size_t m = active.size();

  HessianResult res;
  res.resize(m);

  std::vector<dual2nd> dof(n);
  using Inner = Dual<double>;

  // Seed once to the zero-derivative base.  Of a dual2nd's four scalars
  // [value.value, value.deriv, deriv.value, deriv.deriv] only the two
  // first-order seed slots ever move per probe: value.deriv carries e_j (inner,
  // d/dx_j) and deriv.value carries e_i (outer, d/dx_i).  value.value == x[k]
  // and the second-order seed deriv.deriv == 0 are loop-invariant, so we toggle
  // the two derivative scalars in place rather than reconstructing the whole
  // dual2nd on every seed and reset (the gradient() driver toggles the same
  // way).
  //
  // Each toggle is a scoped_value whose scope IS the span the seed covers.
  // Both guards hold a bare double, never the enclosing dual2nd: when ai == aj
  // they name two different scalars of the same number, and a guard over the
  // number would clobber its sibling.
  std::ranges::transform(x, dof.begin(), [](double v) {
    return dual2nd{Inner{v, 0.0}, Inner{0.0, 0.0}};
  });

  for (std::size_t j = 0; j < m; ++j) {
    const std::size_t aj = active[j];
    // Inner seed e_j is constant across the whole i-loop: held by this guard.
    const auto inner_seed = scoped_seed<1.0>(dof[aj].value().deriv());
    for (std::size_t i = 0; i <= j; ++i) {
      const std::size_t ai = active[i];

      // Outer seed e_i on ai (when ai == aj this lands in aj's other slot).
      const auto outer_seed = scoped_seed<1.0>(dof[ai].deriv().value());

      const dual2nd r = f(dof.data());
      const auto &[A, B] = r;   // value-component, outer-derivative component
      const auto &[a0, a1] = A; // f(x), df/dx_j
      const auto &[b0, b1] = B; // df/dx_i, d2f/dx_i dx_j

      res.value = a0;
      res.gradient[i] = b0;
      res.gradient[j] = a1;
      res.hessian[i * m + j] = b1;
      res.hessian[j * m + i] = b1;
    }
  }
  return res;
}

// Convenience: differentiate every variable.
template <CEnergyOf<dual2nd> F>
HessianResult hessian_scalar(F &&f, std::span<const double> x) {
  const auto all = iota_indices(x.size());
  return hessian_scalar(static_cast<F &&>(f), x, all);
}

} // namespace detail

} // namespace diff
