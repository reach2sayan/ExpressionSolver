#pragma once

#include "dual/dual.hpp"
#include "md/md.hpp"            // md::mdspan — the Hessian's 2D view
#include "util/scope_guard.hpp" // scoped_value — RAII for the per-probe seed toggle

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <type_traits>
#include <memory>
#include <tuple>
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

// The set of variables a sweep is taken over, as a *range* rather than a
// materialised container.
//
// Every driver here indexes `active` and asks for its size, and nothing else.
// Spelling that as `std::span<const std::size_t>` forced the overwhelmingly
// common case -- differentiate everything -- to build a vector 0,1,...,n-1 and
// heap-allocate it, purely so it could be pointed at.  A `views::iota` models
// the same operations, allocates nothing, and folds `active[i]` to `i`, so the
// all-variables path is now *cheaper* than the explicit-subset one instead of
// paying a container for the privilege.  A real span still binds unchanged.
template <typename R>
concept CIndexRange =
    std::ranges::random_access_range<R> && std::ranges::sized_range<R> &&
    std::convertible_to<std::ranges::range_value_t<R>, std::size_t>;

// The point a driver differentiates at.
//
// `std::span<const double>` was the parameter type, which meant every caller
// holding a vector, an array, or a C array had to spell the conversion at the
// call site even though it is lossless.  The drivers ask a point for exactly two
// things -- its size, and its elements in order -- so anything contiguous and
// sized will do, and saying so is both more usable and no less precise.
//
// Contiguity is required rather than merely sized: the entry points funnel into
// one span-based implementation, so a generic range costs a thin wrapper per
// container type and nothing at all in the sweep.  That keeps the generality
// from turning into template bloat in the hot code.
template <typename R>
concept CPointRange =
    std::ranges::contiguous_range<R> && std::ranges::sized_range<R> &&
    std::convertible_to<std::ranges::range_value_t<R>, double>;

// The one place a point range becomes the span the drivers work in.
[[nodiscard]] constexpr std::span<const double>
as_point(const CPointRange auto &x) noexcept {
  return {std::ranges::data(x), std::ranges::size(x)};
}

namespace detail {
// 0,1,...,n-1 without materialising it.  This is what a driver means by "all
// variables"; it is a view, so it costs nothing to form or to pass.
[[nodiscard]] constexpr auto all_indices(std::size_t n) noexcept {
  return std::views::iota(std::size_t{0}, n);
}
} // namespace detail

// What a second-order sweep hands back.
//
// Plain std types, and owning ones: the caller takes the buffers and the library
// keeps no reference to them.  Two shapes, because the extent is a compile-time
// constant for an expression graph and a runtime value for an opaque lambda.
// Both destructure the same way, so `auto [v, g, H, n] = ...` reads identically
// and the binding itself says which you have.
//
//   value    f(x)
//   gradient n entries
//   hessian  n*n entries, ROW-MAJOR: element (i, j) is hessian[i * n + j]
//
// There is deliberately no accessor type wrapping these.  The layout is part of
// the contract rather than something a member function hides, and a caller that
// wants matrix semantics maps Eigen onto the buffer (interop/eigen_interop.hpp)
// without this header having to know Eigen exists.

// Runtime extent: two owning buffers plus the extent they are sized by.
using HessianOwned = std::tuple<double, std::unique_ptr<double[]>,
                                std::unique_ptr<double[]>, std::size_t>;

// Compile-time extent: no allocation at all, and constant-evaluable, which the
// unique_ptr form cannot be.  The extent is N, so it is not carried.
template <std::size_t N>
using HessianStatic =
    std::tuple<double, std::array<double, N>, std::array<double, N * N>>;

namespace detail {
// Average each (i, j) / (j, i) pair in place.
//
// A driver that fills the matrix one row (or column) at a time computes the two
// halves in independent sweeps, so mirrored entries can differ in the last ULP.
// H is analytically symmetric, so averaging only removes that FP noise -- and it
// is what makes the exactly-symmetric contract the drivers advertise true of the
// buffer as well as of the maths.  Drivers whose layout names each entry exactly
// once (the sparse path) have no pair to average and do not call this.
constexpr void symmetrize(double *const h, const std::size_t n) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double s = 0.5 * (h[i * n + j] + h[j * n + i]);
      h[i * n + j] = s;
      h[j * n + i] = s;
    }
  }
}

// Copy a static-extent result into owning buffers.
//
// Needed only where one entry point can reach both shapes: the subset-taking
// hessian() can dispatch to the compile-time-arity reverse driver (when the
// subset happens to be every symbol) or to the runtime probe driver, and a
// function returns one type.  The static form is the cheaper of the two, so it
// stays the return type wherever it is the *only* thing an entry point can
// produce -- which is the plain hessian(expr, x) overload, the one the graph
// benchmarks and most callers use.
template <std::size_t N>
[[nodiscard]] inline HessianOwned to_owned(HessianStatic<N> &&h) {
  auto grad = std::make_unique_for_overwrite<double[]>(N);
  auto hess = std::make_unique_for_overwrite<double[]>(N * N);
  const auto& [value, grad_calc, hess_calc] = h;
  std::ranges::copy(grad_calc, grad.get());
  std::ranges::copy(hess_calc, hess.get());
  return {value, std::move(grad), std::move(hess), N};
}

// Uninitialised owning buffer.  make_unique_for_overwrite, not make_unique: the
// sweep writes every cell, so value-initialising first is a memset the driver
// immediately overwrites -- the same waste HessianResult::resize used to pay.
[[nodiscard]] inline std::unique_ptr<double[]> raw_buffer(std::size_t k) {
  return std::make_unique_for_overwrite<double[]>(k);
}
} // namespace detail

// Caller-owned scratch for a sweep's seeded variables.
//
// The dof array is pure working storage: it is written at the top of every
// sweep and never read afterwards, so there is no reason for each call to buy
// its own.  Handing one of these to a loop over many points allocates on the
// first call and never again.  It grows and does not shrink, for the same
// reason HessianResult does.
//
// `D{v}` is the uniform lift of a plain scalar to a zero-derivative dual at any
// nesting depth: Dual's converting constructor sets the value component and
// value-initialises the derivative, recursively.
// Deliberately heap-backed, with no inline std::array fast path.  An inline
// buffer was tried and measured *worse*: Dual's members carry {} initialisers,
// so std::array<dual2nd, K> is value-initialised on every construction, and a
// 2 KB inline buffer traded one 2 KB allocation for a 2 KB zero-fill on every
// call -- 7,920 -> 9,581 Ir at n = 4.  There is no way to have the storage
// without constructing it, so the allocation is the cheaper of the two.
//
// The win here is reuse, not stack residency: the buffer is grow-only, so a
// workspace handed to a loop over many points allocates once and never again.
template <Numeric D> struct SweepWorkspace {
  std::vector<D> dof;

  // Seed each slot from the point via `make`.  Returns storage valid until the
  // next seed on this workspace.
  //
  // Not const: this fills the scratch.  Not noexcept: growth allocates.
  template <typename Make>
  [[nodiscard]] D *seed_with(const std::span<const double> x, Make make) {
    if (dof.size() < x.size()) {
      dof.resize(x.size());
    }
    std::ranges::transform(x, dof.begin(), make);
    return dof.data();
  }

  // `D{v}` is the uniform lift of a plain scalar to a zero-derivative dual at
  // any nesting depth: Dual's converting constructor sets the value component
  // and value-initialises the derivative, recursively.
  [[nodiscard]] D *seed(const std::span<const double> x) {
    return seed_with(x, [](const double v) { return D{v}; });
  }
};

using GradientWorkspace = SweepWorkspace<dual>;
using HessianWorkspace = SweepWorkspace<dual2nd>;

namespace detail {
// The gradient sweep itself, over storage the caller has already seeded.  Both
// the std::array (compile-time arity) and std::vector (runtime arity) paths
// funnel through here, so the sweep exists once.
template <CEnergyOf<dual> F, CIndexRange R>
constexpr void gradient_into(F &&f, dual *const dof, R &&active,
                             const std::span<double> out) {
  // One seeded pass per active variable: the gradient slot and the dof it comes
  // from advance together.  The seed lives exactly as long as the guard, so it
  // is cleared before the next pass even if the energy throws.
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

template <CEnergyOf<dual> F>
std::vector<double> gradient(F &&f, const std::span<const double> x) {
  return gradient(static_cast<F &&>(f), x, detail::all_indices(x.size()));
}

// Any contiguous sized range of doubles -- vector, array, C array, span, or a
// contiguous view over one.  Converts once and calls the span form; the sweep is
// not re-instantiated per container type.
template <CEnergyOf<dual> F, CPointRange P>
  requires(!std::same_as<std::remove_cvref_t<P>, std::span<const double>>)
std::vector<double> gradient(F &&f, const P &x) {
  return gradient(static_cast<F &&>(f), as_point(x));
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
constexpr double hessian_scalar_into(F &&f, dual2nd *const dof, R &&active,
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
double hessian_scalar(F &&f, const std::span<const double> x,
                      CIndexRange auto &&active, HessianWorkspace &ws,
                      const std::span<double> grad_out,
                      const std::span<double> hess_out) {
  return hessian_scalar_into(static_cast<F &&>(f), ws.seed(x),
                             static_cast<decltype(active) &&>(active),
                             grad_out.data(), hess_out.data());
}

// Compile-time arity: the expression entry points know how many symbols there
// are, so both the seeded-variable array and the result are std::array and this
// path allocates nothing at all.  N is the arity, which is also the extent -- an
// expression is differentiated with respect to all of its own symbols.
template <std::size_t N, CEnergyOf<dual2nd> F>
HessianStatic<N> hessian_scalar_static(F &&f, const std::span<const double> x) {
  std::array<dual2nd, N> dof{};
  std::ranges::transform(x | std::views::take(N), dof.begin(),
                         [](const double v) { return dual2nd{v}; });
  HessianStatic<N> out{};
  std::get<0>(out) =
      hessian_scalar_into(static_cast<F &&>(f), dof.data(), all_indices(N),
                          std::get<1>(out).data(), std::get<2>(out).data());
  return out;
}

// Owning form: allocates the two result buffers, fills them, and hands them over.
template <CEnergyOf<dual2nd> F>
HessianOwned hessian_scalar(F &&f, const std::span<const double> x,
                            CIndexRange auto &&active) {
  const std::size_t m = std::ranges::size(active);
  HessianWorkspace ws;
  auto grad = raw_buffer(m);
  auto hess = raw_buffer(m * m);
  const double value = hessian_scalar_into(
      static_cast<F &&>(f), ws.seed(x),
      static_cast<decltype(active) &&>(active), grad.get(), hess.get());
  return {value, std::move(grad), std::move(hess), m};
}

// Convenience: differentiate every variable.
template <CEnergyOf<dual2nd> F>
HessianOwned hessian_scalar(F &&f, const std::span<const double> x) {
  return hessian_scalar(static_cast<F &&>(f), x, all_indices(x.size()));
}

template <CEnergyOf<dual2nd> F, CPointRange P>
  requires(!std::same_as<std::remove_cvref_t<P>, std::span<const double>>)
HessianOwned hessian_scalar(F &&f, const P &x) {
  return hessian_scalar(static_cast<F &&>(f), as_point(x));
}

} // namespace detail

} // namespace diff
