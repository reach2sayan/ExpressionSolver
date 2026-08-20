#pragma once

// The public `hessian()` entry point, and nothing else: it owns the choice
// between the two Hessian drivers and the argument checks that choice needs.
//
//   drivers/numeric.hpp   detail::hessian  -- O(N^2) probes over a callable
//   drivers/symbolic.hpp  detail::hessian_expr_reverse -- O(N) sweeps of a graph

#include "drivers/numeric.hpp"
#include "drivers/seeded_energy.hpp" // seeded_energy() bridge for expression graphs
#include "drivers/symbolic.hpp"
#include "dual/dual.hpp"
#include "expr/expressions.hpp" // CExpression

#include <cstddef>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>

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
auto hessian(F &&f, const std::span<const double> x, R &&active) {
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
auto hessian(F &&f, const std::span<const double> x) {
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

} // namespace diff
