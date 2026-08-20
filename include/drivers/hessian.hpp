#pragma once

// The public hessian() entry point: it chooses between the O(N^2) probe driver
// (numeric.hpp) and the O(N) reverse graph sweep (symbolic.hpp).

#include "drivers/numeric.hpp"
#include "drivers/seeded_energy.hpp"
#include "drivers/symbolic.hpp"
#include "dual/dual.hpp"
#include "expr/expressions.hpp" // CExpression

#include <cstddef>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace diff::impl {

// The tag seeded_energy() puts on a bridged expression graph.
template <typename F>
concept CSeededExprEnergy =
    requires { requires std::remove_cvref_t<F>::kSeededExprEnergy; };

// An expression graph, a graph already bridged by seeded_energy(), or a raw
// energy callable.
template <typename F>
concept CHessianTarget =
    CExpression<F> || CSeededExprEnergy<F> || CEnergyOf<F, dual2nd>;

namespace detail {

template <typename F>
concept CGraphReverseHessian =
    CExpression<F> && DualLike<typename std::remove_cvref_t<F>::value_type>;

// The bridge reads slots [0, arity) of a buffer sized from `x`, so a short `x`
// reads off the end and a stray `active` index yields a zero row, not an error.
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

// All-variables form: a point longer than the symbol set is as wrong as a short
// one.
constexpr void check_graph_point(std::size_t arity,
                                 std::span<const double> x) {
  if (x.size() != arity) {
    throw std::out_of_range(
        "hessian: an expression graph needs exactly one value per symbol (see "
        "symbol_order<Expr>())");
  }
}

// The graph driver differentiates every symbol; a strict subset keeps the probe
// driver, which can honour `active`.
[[nodiscard]] constexpr bool
reverse_hessian_applies(std::size_t n, std::span<const double> x,
                        CIndexRange auto &&active) noexcept {
  if (x.size() != n || std::ranges::size(active) != n) {
    return false;
  }

  return std::ranges::equal(active, std::views::iota(std::size_t{0}, n));
}

} // namespace detail

namespace detail {
// The arity a bridged callable advertises; 0 if it advertises none.
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
        // One return type for both drivers, so widen the static result.
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

// Every symbol, in canonical order: the extent is compile-time for a graph, so
// this allocates nothing but the result.
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

} // namespace diff::impl
