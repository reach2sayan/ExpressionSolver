#pragma once

#include "coupling.hpp"       // compile-time Hessian sparsity + colouring
#include "dual.hpp"
#include "expressions.hpp"    // CExpression
#include "forward_driver.hpp" // HessianResult + scalar hessian() fallback
#include "gradient.hpp"       // fill_cache / node_cache_t for the reverse sweep
#include "seeded_energy.hpp"  // seeded_energy() bridge for expression graphs
#include "vector_dual.hpp"

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace diff {

// An energy callable produced by seeded_energy() (a compile-time expression
// graph bridged into the driver) advertises this tag.  hessian() routes such
// callables to the scalar driver — see the note on SeededExprEnergy.
template <typename F>
concept CSeededExprEnergy =
    requires { requires std::remove_cvref_t<F>::kSeededExprEnergy; };

namespace detail {

// Vector-forward Hessian for a known compile-time capacity N >= m.
//
// One forward sweep per active variable i (m sweeps total).  The *value* level
// of every dof carries the full identity tangent pack (lane t ==
// d/dx_active[t]), so each sweep also yields the gradient; the *outer*
// derivative level carries the scalar seed e_i.  After f(dof):
// r.value() : value-level VectorDual -> value = f(x), grad[t] = df/dx_t
// r.deriv() : outer-deriv VectorDual -> grad[t] = d2f/dx_i dx_t (Hessian row i)

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

  // Build dof once: value level = the identity tangent pack (constant across
  // sweeps), outer-derivative level = all zero.  Across sweeps only the single
  // outer-derivative *value* at k == active[i] toggles 0->1->0, so per sweep we
  // flip that one scalar instead of rewriting all n entries (was O(n*N)/sweep).
  std::vector<D> dof(n);
  std::ranges::transform(base, dof.begin(),
                         [](const V &v) { return D{v, V{}}; });
  for (std::size_t i = 0; i < m; ++i) {
    const std::size_t ai = active[i];
    auto &dof_ai = dof[ai].deriv();

    dof_ai.value = double{1}; // outer-deriv seed e_i
    const D r = f(dof.data());
    dof_ai.value = double{0}; // reset for next sweep

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

// Forward-over-reverse Hessian for a compile-time expression graph.  Seeding
// column j with a first-order tangent and running ONE backward sweep yields
// that whole Hessian column, so the full N x N matrix costs N sweeps against
// the scalar driver's N(N+1)/2 forward-over-forward probes — an O(N) vs O(N^2)
// difference, and the j == 0 sweep hands back the value and gradient for free.
// Only a graph can take this path: a runtime lambda has no tree to sweep
// backward.  The seed array is sized by the symbol set, so N is compile-time.
template <CExpression Expr>
HessianResult hessian_expr_reverse(const Expr &expr,
                                   std::span<const double> x) {
  using E = std::remove_cvref_t<Expr>;
  using T = typename E::value_type;
  using S = dual_scalar_t<T>;
  using Syms = extract_symbols_from_expr_t<E>;
  constexpr std::size_t N = mpl::mp_size(Syms{});

  // Sparsity read off the expression type, then a colouring of it: columns
  // that share no row can be seeded in the SAME sweep, because no row receives
  // a contribution from more than one of them.  A banded Hessian needs a fixed
  // number of colours regardless of n, so this takes the sweep count from O(N)
  // to O(colours) — the chain energy's tridiagonal-plus-corner pattern colours
  // in 5 whether n is 8 or 32.  A dense pattern colours in N and the loop
  // degenerates to one sweep per column, i.e. never worse.
  static constexpr auto kPattern = hessian_pattern<E>();
  static constexpr auto kColors = color_columns<N>(kPattern);
  static constexpr auto kScatter = scatter_targets<N>(kPattern, kColors);

  HessianResult res;
  res.gradient.assign(N, 0.0);
  res.hessian.assign(N * N, 0.0);

  // The value level is the point and is identical on every sweep; only the
  // seeded tangents move between colours.
  std::array<T, N> seeds{};
  std::ranges::transform(x, seeds.begin(),
                         [](double v) { return T{static_cast<S>(v), S{}}; });

  for (std::size_t c = 0; c < kColors.count; ++c) {
    for (auto &&[seed, color] : std::views::zip(seeds, kColors.color)) {
      seed.deriv() = (color == c) ? S{1} : S{};
    }

    std::array<T, N> grads{};
    node_cache_t<E> cache{};
    const T root = fill_cache<0, Syms>(expr, seeds, cache);
    expr.backward(Syms{}, T{1}, grads, cache);

    if (c == 0) {
      // The value and the first-order adjoints do not depend on the tangent
      // seeding, so any one sweep yields both.
      res.value = static_cast<double>(root.template get<0>());
      std::ranges::transform(grads, res.gradient.begin(), [](const T &g) {
        return static_cast<double>(g.template get<0>());
      });
    }

    // Row i received the sum over this colour's columns; the colouring
    // guarantees at most one of them is structurally nonzero in row i, so the
    // sum IS that entry, and which column it belongs to was resolved at compile
    // time.  Entries outside the pattern are never written and stay zero.
    for (auto &&[row, grad, target] :
         std::views::zip(res.hessian | std::views::chunk(N), grads,
                         kScatter[c])) {
      if (target != no_column) {
        row[target] = static_cast<double>(grad.template get<1>());
      }
    }
  }

  // Columns come from independent sweeps, so H(i,j) and H(j,i) can differ in
  // the last ULP.  Same exactly-symmetric contract the other drivers honour.
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = i + 1; j < N; ++j) {
      const double s = 0.5 * (res.hessian[i * N + j] + res.hessian[j * N + i]);
      res.hessian[i * N + j] = s;
      res.hessian[j * N + i] = s;
    }
  }
  return res;
}

// The nonzero VALUES of the Hessian, in the compressed-column order fixed by
// sparse_layout<Expr>() — the same colour-compressed sweeps as above, but
// scattered straight into compressed storage, so the dense N x N matrix is
// never materialised and the allocation is O(nnz) rather than O(N^2).
//
// Deliberately Eigen-free: it returns a plain value buffer, and
// eigen_interop.hpp is what maps that buffer onto an Eigen::SparseMatrix.  This
// keeps every Eigen contact in one header.
//
// No symmetrization pass: the layout already names each entry exactly once, and
// H(i,j) and H(j,i) are separate slots filled from the sweep of their own
// colour, so there is no pair to average.
template <CExpression Expr>
std::vector<double> hessian_values_sparse(const Expr &expr,
                                          std::span<const double> x) {
  using E = std::remove_cvref_t<Expr>;
  using T = typename E::value_type;
  using S = dual_scalar_t<T>;
  using Syms = extract_symbols_from_expr_t<E>;
  constexpr std::size_t N = mpl::mp_size(Syms{});

  static constexpr auto kPattern = hessian_pattern<E>();
  static constexpr auto kColors = color_columns<N>(kPattern);
  static constexpr auto kSlots = sparse_slots<E>();

  std::vector<double> values(hessian_nnz<E>(), 0.0);

  std::array<T, N> seeds{};
  std::ranges::transform(x, seeds.begin(),
                         [](double v) { return T{static_cast<S>(v), S{}}; });

  for (std::size_t c = 0; c < kColors.count; ++c) {
    for (auto &&[seed, color] : std::views::zip(seeds, kColors.color)) {
      seed.deriv() = (color == c) ? S{1} : S{};
    }

    std::array<T, N> grads{};
    node_cache_t<E> cache{};
    fill_cache<0, Syms>(expr, seeds, cache);
    expr.backward(Syms{}, T{1}, grads, cache);

    for (auto &&[grad, slot] : std::views::zip(grads, kSlots[c])) {
      if (slot != no_column) {
        values[slot] = static_cast<double>(grad.template get<1>());
      }
    }
  }
  return values;
}

// A graph can take the forward-over-reverse path only if its numbers carry a
// first-order tangent to seed — that is what makes one backward sweep yield a
// whole Hessian column rather than just a gradient.
template <typename F>
concept CGraphReverseHessian =
    CExpression<F> && DualLike<typename std::remove_cvref_t<F>::value_type>;

// Precondition for handing an expression graph to a probe-based driver.
//
// The bridge built by seeded_energy() reads a fixed window — slots [0, arity)
// of the driver's seed buffer — because a raw `const Dof *` carries no length
// (see SeededExprEnergy::operator()).  The buffer is sized from `x`, so the two
// only line up if the caller supplied at least one value per symbol, and only
// addressed symbols that exist.  A short `x` would read off the end of the
// driver's vector; an out-of-range `active` index would be silently dropped by
// the bridge and come back as a zero row rather than an error.  Both are
// checked here, for the same reason eval() refuses a short range
// (bound.hpp): silently differentiating at the wrong point is worse than
// throwing.
// constexpr, not noexcept, for the reason spelled out in bound.hpp: a throw in
// a constexpr function only matters if it is actually reached, so a bad point
// is a compile error during constant evaluation and an exception at run time.
constexpr void check_graph_point(std::size_t arity, std::span<const double> x,
                                 std::span<const std::size_t> active) {
  if (x.size() < arity) {
    throw std::out_of_range(
        "hessian: an expression graph needs one value per symbol; the point is "
        "shorter than the symbol count (see symbol_order<Expr>())");
  }
  if (std::ranges::any_of(active,
                          [arity](std::size_t i) { return i >= arity; })) {
    throw std::out_of_range(
        "hessian: active index does not name a symbol of this expression "
        "(see symbol_order<Expr>())");
  }
}

// All-variables form: with no `active` the driver differentiates every slot of
// x, so a point LONGER than the symbol set is just as wrong as a short one —
// the surplus rows would come back zero instead of erroring.
constexpr void check_graph_point(std::size_t arity,
                                 std::span<const double> x) {
  if (x.size() != arity) {
    throw std::out_of_range(
        "hessian: an expression graph needs exactly one value per symbol (see "
        "symbol_order<Expr>())");
  }
}

// The graph driver differentiates every symbol, in canonical order.  A caller
// asking for a strict subset (or passing a point that is not one value per
// symbol) keeps the probe-based driver, which can honour `active`.
[[nodiscard]] constexpr bool
reverse_hessian_applies(std::size_t n, std::span<const double> x,
                        std::span<const std::size_t> active) noexcept {
  if (x.size() != n || active.size() != n) {
    return false;
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (active[i] != i) {
      return false;
    }
  }
  return true;
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

// Above this many active variables the scalar probe driver beats vector-forward
// for a raw callable, so hessian() stops widening the pack and switches.  Each
// vector-forward sweep carries a Dual<VectorDual<N>> per node — 576 bytes at
// N=32 — and that per-node width eventually outweighs running O(m) sweeps
// instead of O(m^2) probes.  Measured on this box: at m=32 the scalar driver is
// ~1.7x faster on the sparse energy and ~3.3x on the dense one, while at m=8
// vector-forward still wins both.  8 is the value that gets the most regimes
// right; the true crossover depends on how expensive the energy is per node,
// which the router cannot see, so it is overridable.
#ifndef DIFF_VFORWARD_CROSSOVER
#define DIFF_VFORWARD_CROSSOVER 8
#endif
inline constexpr std::size_t kVForwardCrossover = DIFF_VFORWARD_CROSSOVER;

// Driver selection, decided at compile time from what `f` is:
//
//   expression graph  -> forward-over-reverse, O(N) backward sweeps.  The graph
//                        is the only input with a tree to sweep backward, and
//                        this is the only O(N) driver — everything else is
//                        O(N^2) probes, autodiff's hessian() included.
//   seeded_energy tag -> scalar forward-over-forward (the graph already bridged
//                        by the caller, so no tree is left to sweep).
//   raw callable      -> vector-forward, the small-m winner.
template <typename F>
HessianResult hessian(F &&f, std::span<const double> x,
                      std::span<const std::size_t> active) {
  if constexpr (CExpression<F>) {
    if constexpr (detail::CGraphReverseHessian<F>) {
      constexpr std::size_t N = mpl::mp_size(
          extract_symbols_from_expr_t<std::remove_cvref_t<F>>{});
      if (detail::reverse_hessian_applies(N, x, active)) {
        return detail::hessian_expr_reverse(f, x);
      }
    }
    detail::check_graph_point(
        mpl::mp_size(extract_symbols_from_expr_t<std::remove_cvref_t<F>>{}), x,
        active);
    return detail::hessian_scalar(seeded_energy(static_cast<F &&>(f)), x,
                                  active);
  } else if constexpr (CSeededExprEnergy<F>) {
    detail::check_graph_point(std::remove_cvref_t<F>::arity, x, active);
    return detail::hessian_scalar(static_cast<F &&>(f), x, active);
  } else if (active.size() > kVForwardCrossover) {
    return detail::hessian_scalar(static_cast<F &&>(f), x, active);
  } else {
    return hessian_vforward(static_cast<F &&>(f), x, active);
  }
}
template <typename F> HessianResult hessian(F &&f, std::span<const double> x) {
  if constexpr (CExpression<F>) {
    if constexpr (detail::CGraphReverseHessian<F>) {
      constexpr std::size_t N = mpl::mp_size(
          extract_symbols_from_expr_t<std::remove_cvref_t<F>>{});
      if (x.size() == N) {
        return detail::hessian_expr_reverse(f, x);
      }
    }
    detail::check_graph_point(
        mpl::mp_size(extract_symbols_from_expr_t<std::remove_cvref_t<F>>{}), x);
    return detail::hessian_scalar(seeded_energy(static_cast<F &&>(f)), x);
  } else if constexpr (CSeededExprEnergy<F>) {
    detail::check_graph_point(std::remove_cvref_t<F>::arity, x);
    return detail::hessian_scalar(static_cast<F &&>(f), x);
  } else if (x.size() > kVForwardCrossover) {
    return detail::hessian_scalar(static_cast<F &&>(f), x);
  } else {
    return hessian_vforward(static_cast<F &&>(f), x);
  }
}

} // namespace diff
