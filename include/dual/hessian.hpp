#pragma once

// Chooses between the O(N^2) probe driver (numeric.hpp) and the O(N) reverse
// graph sweep (symbolic.hpp).

#include "dual/dual.hpp"
#include "dual/numeric.hpp"
#include "symbolic/expressions.hpp" // CExpression
#include "symbolic/seeded_energy.hpp"
#include "symbolic/sweep.hpp"
#include "util/error.hpp"

#include <cstddef>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>

namespace ddx::impl {

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
[[nodiscard]] constexpr result<void>
check_graph_point(std::size_t arity, std::span<const double> x,
                  CIndexRange auto &&active) noexcept {
  if (x.size() < arity) {
    return fail(errc::short_point);
  } else if (std::ranges::any_of(
                 active, [arity](std::size_t i) { return i >= arity; })) {
    return fail(errc::index_out_of_range);
  }
  return {};
}

// All-variables form: a point longer than the symbol set is as wrong as a short
// one.
[[nodiscard]] constexpr result<void>
check_graph_point(std::size_t arity, std::span<const double> x) noexcept {
  return (x.size() != arity) ? fail(errc::wrong_arity) : result<void>{};
}

// The graph driver differentiates every symbol; a strict subset keeps the probe
// driver, which honours `active`.
[[nodiscard]] constexpr bool
reverse_hessian_applies(std::size_t n, std::span<const double> x,
                        CIndexRange auto &&active) noexcept {
  return x.size() == n &&
         std::ranges::equal(active, std::views::iota(std::size_t{0}, n));
}

// The arity a bridged callable advertises, if it advertises one.
template <CHessianTarget F>
constexpr std::optional<std::size_t> declared_arity() noexcept {
  if constexpr (CSeededExprEnergy<F>) {
    return std::remove_cvref_t<F>::arity;
  } else {
    return std::nullopt;
  }
}

// The point checks both entry points share.  Free to hoist:
// reverse_hessian_applies is strictly stronger than check_graph_point, and a
// callable advertising no arity has nothing to check against.
template <CHessianTarget F>
[[nodiscard]] constexpr result<void> check_target(std::span<const double> x,
                                                  CIndexRange auto &&...active)
  requires(sizeof...(active) <= 1)
{
  if constexpr (CExpression<F>) {
    return check_graph_point(expr_arity_v<F>, x, active...);
  } else if constexpr (declared_arity<F>().has_value()) {
    return check_graph_point(*declared_arity<F>(), x, active...);
  } else {
    return {};
  }
}
} // namespace detail

// The point is a span, so its length only arrives with the call: both entry
// points answer with result<T>.
auto hessian(CHessianTarget auto &&f, const std::span<const double> x,
             CIndexRange auto &&active) {
  using F = std::remove_cvref_t<decltype(f)>;
  return detail::check_target<F>(x, active).transform([&] {
    if constexpr (CExpression<F>) {
      if constexpr (detail::CGraphReverseHessian<F>) {
        if (detail::reverse_hessian_applies(detail::expr_arity_v<F>, x,
                                            active)) {
          // One return type for both drivers, so widen the static result.
          return detail::to_owned(detail::hessian_expr_reverse(f, x));
        }
      }
      return detail::hessian(seeded_energy(std::forward<F>(f)), x, active);
    } else {
      return detail::hessian(std::forward<F>(f), x, active);
    }
  });
}

// Every symbol, in canonical order.  The extent is compile-time for a graph, so
// this allocates nothing but the result and a constant evaluation can run it;
// the subset-taking form widens to the owning shape and cannot.
constexpr auto hessian(CHessianTarget auto &&f,
                       const std::span<const double> x) {
  using F = std::remove_cvref_t<decltype(f)>;
  return detail::check_target<F>(x).transform([&] {
    if constexpr (CExpression<F>) {
      if constexpr (detail::CGraphReverseHessian<F>) {
        if (x.size() == detail::expr_arity_v<F>) {
          return detail::hessian_expr_reverse(f, x);
        }
      }
      return detail::hessian_static<detail::expr_arity_v<F>>(
          seeded_energy(std::forward<F>(f)), x);
    } else if constexpr (detail::declared_arity<F>().has_value()) {
      return detail::hessian_static<*detail::declared_arity<F>()>(
          std::forward<F>(f), x);
    } else {
      return detail::hessian(std::forward<F>(f), x);
    }
  });
}

} // namespace ddx::impl
