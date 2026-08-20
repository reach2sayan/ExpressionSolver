#pragma once

// Vocabulary shared by every driver: what an energy must look like, what a
// sweep is taken over, what a second-order result is, and the scratch a sweep
// seeds into.  Split out so `numeric.hpp` (runtime callables) and
// `symbolic.hpp` (expression graphs) can each depend on the shapes without
// depending on each other.

#include "dual/dual.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
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
// forward-over-forward Hessian.  Naming it here is what lets an energy that is
// not generic enough fail at its call site rather than inside a sweep.
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

// The point every driver here differentiates at is a `std::span<const double>`,
// and that is deliberately a concrete type rather than a concept.
//
// It reads like it forces the caller's hand and it does not: a span parameter is
// non-deduced, so implicit conversion applies at the call site and a vector, a
// std::array, a C array, and a contiguous view over any of them all bind with
// nothing spelled.  A temporary binds too -- span's range constructor drops the
// borrowed_range requirement when the element type is const, which it is here.
//
// There used to be a `CPointRange` concept plus an `as_point()` helper and a
// second overload of every entry point taking `const P&`, held apart from the
// span one by a `requires(!same_as<..., span>)` clause.  All of it was
// accepting exactly what the span parameter already accepted.  Erasing to the
// span at the boundary also keeps each sweep instantiated once instead of once
// per container type, which the concept version had to hand-arrange.
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
// first call and never again.
//
// Small points -- which is nearly all of them -- are seeded into an inline
// block and never touch the allocator at all.  The block is *raw storage*, and
// that is the whole point: a plain `std::array<D, K>` member was tried first
// and measured worse (7,920 -> 9,581 Ir at n = 4), because Dual's members carry
// `{}` initialisers, so the array is value-initialised on every construction
// and the call trades one allocation for a full zero-fill of the block.  Here
// nothing is constructed until `seed_with` places a seed in a slot, so the
// unused tail costs exactly nothing.  D is required to be trivially
// destructible, which is what makes reusing the block between calls -- and
// simply abandoning it -- well defined.

namespace detail {
// The inline block, or nothing at all.
//
// `alignas(D) std::byte[N]` is an array member, and -fstack-protector-strong
// (on by default on this toolchain) gives a canary to every function holding
// one.  For a D wide enough that the block could never hold a whole point that
// canary -- plus the stack it reserves -- is pure loss, so the block is not
// declared at all there and [[no_unique_address]] gives the empty stand-in no
// footprint.  Measured: carrying an unusable block cost the vector-forward
// driver 3.8% at Dual<VectorDual<8>>, a 192-byte element (that driver has since
// been deleted, but the rule -- do not declare a block a type can never use --
// is what the static_assert-free `inline_floor` gate below encodes).
template <typename D, std::size_t Bytes> struct inline_block {
  alignas(D) std::byte storage[Bytes];
};
struct no_inline_block {};
} // namespace detail

template <Numeric D> struct SweepWorkspace {
  static_assert(std::is_trivially_destructible_v<D>,
                "the inline block is reused and abandoned without destruction");

  // A byte budget, not an element count: this template is instantiated at
  // dual2nd (32 B) and could be instantiated at a much wider D.  A block that
  // cannot hold `inline_floor` seeds is not worth its stack, so it is not
  // declared at all there.
  static constexpr std::size_t inline_bytes = 512;
  static constexpr std::size_t inline_floor = 8;
  static constexpr std::size_t inline_capacity =
      (inline_bytes / sizeof(D) >= inline_floor) ? inline_bytes / sizeof(D) : 0;

  [[no_unique_address]] std::conditional_t<
      (inline_capacity > 0), detail::inline_block<D, inline_bytes>,
      detail::no_inline_block> block;
  std::vector<D> heap;

  // Seed each slot from the point via `make`.  Returns storage valid until the
  // next seed on this workspace.
  //
  // Not const: this fills the scratch.  Not noexcept: growth may allocate.
  template <typename Make>
  [[nodiscard]] D *seed_with(const std::span<const double> x, Make make) {
    if constexpr (inline_capacity > 0) {
      if (x.size() <= inline_capacity) {
        // Raw storage, so a slot may hold no object yet: each seed is a
        // construction rather than an assignment.  D is trivially destructible,
        // so whatever a previous call left in the block simply ends.
        D *const dof = reinterpret_cast<D *>(block.storage);
        for (std::size_t k = 0; k < x.size(); ++k) {
          std::construct_at(dof + k, make(x[k]));
        }
        return dof;
      }
    }
    // Grow-only, and resize() rather than a capacity check: the vector's size
    // is what a later .data() read is entitled to, and resize keeps capacity on
    // shrink anyway.  These slots already hold live objects, so seeding them is
    // plain assignment, and the transform stays exactly what it was before the
    // inline block existed -- the wide instantiations reach only this path and
    // must not pay for a fast path they can never take.
    if (heap.size() < x.size()) {
      heap.resize(x.size());
    }
    std::ranges::transform(x, heap.begin(), make);
    return heap.data();
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

} // namespace diff
