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

// The tag seeded_energy() puts on a bridged expression graph; hessian() routes
// such callables to the scalar driver.
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

// Precondition for handing an expression graph to a probe-based driver.  The
// bridge reads slots [0, arity) of a buffer sized from `x`, so a short `x` would
// read off the end and an out-of-range `active` index would come back as a zero
// row rather than an error.  Silently differentiating at the wrong point is
// worse than throwing.
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

// All-variables form: every slot of x is differentiated, so a point LONGER than
// the symbol set is as wrong as a short one -- the surplus rows would be zero.
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
}

} // namespace detail

// Driver selection.  Only one kind of input can choose:
//
//   expression graph  -> forward-over-reverse, O(N) backward sweeps.  A graph is
//                        the only input carrying a tree to sweep backward.  When
//                        that does not apply (a strict `active` subset, a
//                        mismatched point, a non-dual graph) the graph is bridged
//                        and falls through to the probes.
//   anything else     -> the O(N^2) probe driver.
//
// A seeded_energy()-tagged callable is a graph already bridged, so it is
// "anything else" -- it differs from a raw lambda only in advertising its arity,
// which is worth an argument check.  That check is the rest of the dispatch.
namespace detail {
// The arity a bridged or raw callable advertises; 0 if it advertises none.
template <CHessianTarget F> constexpr std::size_t declared_arity() noexcept {
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
        // This overload can also reach the runtime-extent probe driver, and a
        // function has one return type, so the static result is widened.
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
// the extent is then compile-time for a graph, so the sweep runs out of a
// std::array and allocates nothing but the result.
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
