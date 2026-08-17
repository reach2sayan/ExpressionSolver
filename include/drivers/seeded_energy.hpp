#pragma once

#include "expr/expressions.hpp" // CExpression, Numeric
#include "expr/traits.hpp"      // extract_symbols_from_expr_t

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

#include "util/mpl.hpp" // mp_size

namespace diff {

// Bridges a compile-time expression *graph* (Variable / + / * / log / exp ...)
// into the runtime numeric Hessian driver.  The driver hands us seeded dual
// dofs in sorted symbol order; we pack them and traverse the graph once via
// eval_seeded.  Holding the expression by value keeps the dofs alive for the
// driver and lets the resulting callable be stored or returned safely.
//
// The wrapper carries a static tag (kSeededExprEnergy) that the public
// hessian() router keys on: an expression-template energy is dispatched to the
// *scalar* O(n^2) forward-over-forward driver instead of vector-forward.  A
// graph node carries wide Dual<VectorDual<N>> intermediates whose per-node cost
// scales with pack width and dwarfs the fewer-sweeps saving the vector-forward
// driver is built for.  Raw arithmetic energy lambdas
// carry no tag, so the router keeps routing those to vector-forward, where they
// win at small n.
template <CExpression Expr> class SeededExprEnergy {
  static_assert(!std::is_reference_v<Expr>,
                "SeededExprEnergy stores the expression by value on purpose — a "
                "reference parameter would leave the callable holding a "
                "dangling graph.  Use seeded_energy(), which decays.");
  Expr expr_;

public:
  using symbols = extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  static constexpr std::size_t arity = mp::mp_size(symbols{});

  // Marker the hessian() router dispatches on (see vforward_driver.hpp).
  static constexpr bool kSeededExprEnergy = true;

  explicit constexpr SeededExprEnergy(Expr expr) noexcept
      : expr_(static_cast<Expr &&>(expr)) {}

  template <Numeric Dof>
  [[nodiscard]] constexpr auto operator()(const Dof *dof) const noexcept {
    using T = std::remove_cvref_t<Dof>;
    // Built by copy, not value-initialized then overwritten: this runs once per
    // Hessian probe, and for a wide dof type zeroing first doubles the writes.
    const auto s = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return std::array<T, arity>{dof[I]...};
    }(std::make_index_sequence<arity>{});
    return expr_.template eval_seeded<symbols>(s);
  }
};

template <CExpression Expr>
[[nodiscard]] constexpr auto seeded_energy(Expr &&expr) noexcept {
  return SeededExprEnergy<std::remove_cvref_t<Expr>>(
      static_cast<Expr &&>(expr));
}

} // namespace diff
