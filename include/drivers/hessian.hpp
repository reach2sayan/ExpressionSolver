#pragma once

#include "drivers/coupling.hpp" // compile-time Hessian sparsity + colouring
#include "drivers/forward_driver.hpp" // HessianResult + scalar hessian() fallback
#include "drivers/gradient.hpp" // fill_cache / node_cache_t for the reverse sweep
#include "drivers/seeded_energy.hpp" // seeded_energy() bridge for expression graphs
#include "dual/dual.hpp"
#include "expr/expressions.hpp" // CExpression
#include "util/scope_guard.hpp" // scoped_value — RAII for the per-sweep seed

#include <algorithm>
#include <array>
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

// A raw energy callable the scalar probe driver can sweep.  dual2nd is the only
// number hessian() ever seeds a non-graph callable with, so this is the whole
// requirement.
template <typename F>
concept CRawEnergy = CEnergyOf<F, dual2nd>;

// Everything hessian() knows how to differentiate:
// 1. an expression graph (which it bridges or sweeps in reverse),
// 2. a graph already bridged by seeded_energy(),
// 3. a raw energy callable.
template <typename F>
concept CHessianTarget =
    CExpression<F> || CSeededExprEnergy<F> || CRawEnergy<F>;

namespace detail {

template <CExpression Expr, typename Colors, typename Harvest>
DIFF_ALWAYS_INLINE void color_sweeps(const Expr &expr, std::span<const double> x,
                  const Colors &colors, Harvest &&harvest) {
  using E = std::remove_cvref_t<Expr>;
  using T = typename E::value_type;
  using S = dual_scalar_t<T>;
  using Syms = detail::expr_symbols_t<E>;
  constexpr std::size_t N = mpl::mp_size(Syms{});

  // The value level is the point and is identical on every sweep; only the
  // seeded tangents move between colours.
  std::array<T, N> seeds{};
  std::ranges::transform(x, seeds.begin(),
                         [](double v) { return T{static_cast<S>(v), S{}}; });

  for (std::size_t c = 0; c < colors.count; ++c) {
    for (auto &&[seed, color] : std::views::zip(seeds, colors.color)) {
      seed.deriv() = (color == c) ? S{1} : S{};
    }
    std::array<T, N> grads{};
    const T root = reverse_sweep<Syms>(expr, seeds, grads);
    harvest(c, root, grads);
  }
}

// Forward-over-reverse Hessian for a compile-time expression graph.  Seeding
// column j with a first-order tangent and running ONE backward sweep yields
// that whole Hessian column, so the full N x N matrix costs N sweeps against
// the scalar driver's N(N+1)/2 forward-over-forward probes — an O(N) vs O(N^2)
// difference, and the j == 0 sweep hands back the value and gradient for free.
// Only a graph can take this path: a runtime lambda has no tree to sweep
// backward.  The seed array is sized by the symbol set, so N is compile-time.
template <CExpression Expr>
HessianStatic<mpl::mp_size(detail::expr_symbols_t<std::remove_cvref_t<Expr>>{})>
hessian_expr_reverse(const Expr &expr, std::span<const double> x) {
  using E = std::remove_cvref_t<Expr>;
  using T = typename E::value_type;
  using Syms = detail::expr_symbols_t<E>;
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
  // Value-initialised, and that is load-bearing: the scatter below writes only
  // structurally non-zero entries and relies on everything outside the pattern
  // already being zero.
  HessianStatic<N> res{};
  auto &res_gradient = std::get<1>(res);
  auto &res_hessian = std::get<2>(res);

  color_sweeps(expr, x, kColors, [&](std::size_t c, const T &root,
                                     const auto &grads) {
    static constexpr auto kScatter = scatter_targets<N>(kPattern, kColors);
    if (c == 0) {
      // The value and the first-order adjoints do not depend on the tangent
      // seeding, so any one sweep yields both.
      std::get<0>(res) = static_cast<double>(root.template get<0>());
      std::ranges::transform(grads, res_gradient.begin(), [](const T &g) {
        return static_cast<double>(g.template get<0>());
      });
    }

    // Row i received the sum over this colour's columns; the colouring
    // guarantees at most one of them is structurally nonzero in row i, so the
    // sum IS that entry, and which column it belongs to was resolved at compile
    // time.  Entries outside the pattern are never written and stay zero.
    for (auto &&[row, grad, target] :
         std::views::zip(res_hessian | std::views::chunk(N), grads,
                         kScatter[c])) {
      if (target != no_column) {
        // A chunk view indexes by its signed difference_type, not by size_t.
        row[static_cast<std::ranges::range_difference_t<decltype(row)>>(
            target)] = static_cast<double>(grad.template get<1>());
      }
    }
  });

  // Columns come from independent sweeps, so mirrored entries can differ in
  // the last ULP.
  detail::symmetrize(res_hessian.data(), N);
  return res;
}

// The nonzero VALUES of the Hessian, in the compressed-column order fixed by
// sparse_layout<Expr>() — the same colour-compressed sweeps as above, but
// scattered straight into compressed storage, so the dense N x N matrix is
// never materialised and the buffer is O(nnz) rather than O(N^2).
// No symmetrization pass: the layout already names each entry exactly once, and
// H(i,j) and H(j,i) are separate slots filled from the sweep of their own
// colour, so there is no pair to average.
//
// NNZ rides along as a defaulted template parameter — the same shape as
// sparse_layout() in coupling.hpp — because hessian_nnz is consteval: the size
// of this buffer is a property of Expr, so it is an array and not a vector, and
// the sparse path does not allocate.  std::vector is for the runtime-extent
// drivers above, where the active-variable count is a span and not a type.
template <CExpression Expr,
          std::size_t NNZ = hessian_nnz<std::remove_cvref_t<Expr>>()>
std::array<double, NNZ + 1> hessian_values_sparse(const Expr &expr,
                                                  std::span<const double> x) {
  using E = std::remove_cvref_t<Expr>;
  using T = typename E::value_type;
  using Syms = detail::expr_symbols_t<E>;
  constexpr std::size_t N = mpl::mp_size(Syms{});

  static constexpr auto kPattern = hessian_pattern<E>();
  static constexpr auto kColors = color_columns<N>(kPattern);

  // One cell past the nonzeros: layout_sparse_pattern maps every structural
  // zero onto that last slot, so H[i, j] answers for indices the pattern
  // excludes without the caller checking first.  Eigen reads exactly nnz
  // entries from values.data(), so the extra cell is invisible to it.
  //
  // Value-initialised, which zeroes it: the sweeps below only write the slots
  // the pattern names, and every other cell — the sink included — has to read
  // back as exactly 0.0.
  std::array<double, NNZ + 1> values{};

  color_sweeps(expr, x, kColors,
               [&](std::size_t c, const T &, const auto &grads) {
                 static constexpr auto kSlots = sparse_slots<E>();
                 for (auto &&[grad, slot] : std::views::zip(grads, kSlots[c])) {
                   if (slot != no_column) {
                     values[slot] = static_cast<double>(grad.template get<1>());
                   }
                 }
               });
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
constexpr void check_graph_point(std::size_t arity, std::span<const double> x,
                                 CIndexRange auto &&active) {
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
                        CIndexRange auto &&active) noexcept {
  if (x.size() != n || std::ranges::size(active) != n) {
    return false;
  }

  return std::ranges::equal(active, std::views::iota(std::size_t{0}, n));
  return true;
}

} // namespace detail

// Driver selection.  There are two Hessian drivers left, and only one kind of
// input can choose between them:
//
//   expression graph  -> forward-over-reverse, O(N) backward sweeps.  A graph is
//                        the only input carrying a tree to sweep backward, and
//                        this is the only O(N) driver -- everything else is
//                        O(N^2) probes, autodiff's hessian() included.  When it
//                        does not apply (a strict `active` subset, a mismatched
//                        point, a non-dual graph) the graph is bridged and falls
//                        through to the probes.
//   anything else     -> the O(N^2) probe driver, full stop.
//
// A `seeded_energy()`-tagged callable is a graph the caller already bridged, so
// there is no tree left and it is "anything else" -- it differs from a raw
// lambda only in that its arity is known, which is worth an argument check.
// That check is the entire remainder of the dispatch.
//
// This used to be a three-way choice with a size threshold in it: raw callables
// went to a vector-forward driver below kVForwardCrossover.  That driver was
// deleted on 2026-08-20 for losing at every size, and the branch went with it.
namespace detail {
// The arity a bridged or raw callable advertises, for the argument check; 0 if
// it advertises none (a raw lambda, whose arity is whatever the point says).
template <typename F> constexpr std::size_t declared_arity() noexcept {
  if constexpr (CSeededExprEnergy<F>) {
    return std::remove_cvref_t<F>::arity;
  } else {
    return 0;
  }
}
} // namespace detail

template <CHessianTarget F, CIndexRange R>
auto hessian(F &&f, std::span<const double> x, R &&active) {
  if constexpr (CExpression<F>) {
    if constexpr (detail::CGraphReverseHessian<F>) {
      if (detail::reverse_hessian_applies(detail::expr_arity_v<F>, x, active)) {
        // This overload can also reach the runtime-extent probe driver below,
        // and a function has one return type, so the static result is widened.
        return detail::to_owned(detail::hessian_expr_reverse(f, x));
      }
    }
    detail::check_graph_point(detail::expr_arity_v<F>, x, active);
    return detail::hessian(seeded_energy(static_cast<F &&>(f)), x, active);
  } else {
    if constexpr (detail::declared_arity<F>() > 0) {
      detail::check_graph_point(detail::declared_arity<F>(), x, active);
    }
    return detail::hessian(static_cast<F &&>(f), x, active);
  }
}

// Every symbol, in canonical order.  Distinct from the overload above because
// the extent is then a compile-time constant for a graph, which lets the whole
// sweep run out of a std::array and allocate nothing but the result.
template <CHessianTarget F>
auto hessian(F &&f, std::span<const double> x) {
  if constexpr (CExpression<F>) {
    if constexpr (detail::CGraphReverseHessian<F>) {
      if (x.size() == detail::expr_arity_v<F>) {
        return detail::hessian_expr_reverse(f, x);
      }
    }
    detail::check_graph_point(detail::expr_arity_v<F>, x);
    return detail::hessian_static<detail::expr_arity_v<F>>(
        seeded_energy(static_cast<F &&>(f)), x);
  } else if constexpr (detail::declared_arity<F>() > 0) {
    detail::check_graph_point(detail::declared_arity<F>(), x);
    return detail::hessian_static<detail::declared_arity<F>()>(
        static_cast<F &&>(f), x);
  } else {
    return detail::hessian(static_cast<F &&>(f), x);
  }
}

// Any contiguous sized range of doubles -- vector, array, C array, span.  One
// conversion, then the span form; no driver is re-instantiated per container.
template <CHessianTarget F, CPointRange P>
  requires(!std::same_as<std::remove_cvref_t<P>, std::span<const double>>)
auto hessian(F &&f, const P &x) {
  return hessian(static_cast<F &&>(f), as_point(x));
}

template <CHessianTarget F, CPointRange P, CIndexRange R>
  requires(!std::same_as<std::remove_cvref_t<P>, std::span<const double>>)
auto hessian(F &&f, const P &x, R &&active) {
  return hessian(static_cast<F &&>(f), as_point(x),
                 static_cast<R &&>(active));
}

} // namespace diff
