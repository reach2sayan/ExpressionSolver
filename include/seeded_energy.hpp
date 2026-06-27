#pragma once

#include "traits.hpp" // extract_symbols_from_expr_t

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

#include <boost/mp11/list.hpp> // mp_size

namespace diff {

// Bridges a compile-time expression *graph* (Variable / + / * / log / exp ...)
// into the runtime numeric Hessian driver.  The driver hands us seeded dual
// dofs in sorted symbol order; we pack them and traverse the graph once via
// eval_seeded_as.  Holding the expression by value keeps the dofs alive for the
// driver and lets the resulting callable be stored or returned safely.
//
// The wrapper carries a static tag (kSeededExprEnergy) that the public
// hessian() router keys on: an expression-template energy is dispatched to the
// *scalar* O(n^2) forward-over-forward driver instead of vector-forward.  A
// graph node carries wide Dual<VectorDual<N>> intermediates whose per-node cost
// scales with pack width and dwarfs the fewer-sweeps saving the vector-forward
// driver is built for (BENCHMARKS.md: expression-template n=4 measures
// 4,095 ns vector-forward vs 723 ns scalar).  Raw arithmetic energy lambdas
// carry no tag, so the router keeps routing those to vector-forward, where they
// win at small n.
template <typename Expr> class SeededExprEnergy {
  Expr expr_;

public:
  using symbols = extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  static constexpr std::size_t arity = mp::mp_size<symbols>::value;

  // Marker the hessian() router dispatches on (see vforward_driver.hpp).
  static constexpr bool kSeededExprEnergy = true;

  explicit constexpr SeededExprEnergy(Expr expr) noexcept
      : expr_(static_cast<Expr &&>(expr)) {}

  template <typename Dof>
  [[nodiscard]] constexpr auto operator()(const Dof *dof) const noexcept {
    using T = std::remove_cvref_t<Dof>;
    std::array<T, arity> s{};
    std::copy_n(dof, arity, s.begin());
    return expr_.template eval_seeded_as<T, symbols>(s);
  }
};

template <typename Expr>
[[nodiscard]] constexpr auto seeded_energy(Expr &&expr) noexcept {
  return SeededExprEnergy<std::remove_cvref_t<Expr>>(
      static_cast<Expr &&>(expr));
}

} // namespace diff
