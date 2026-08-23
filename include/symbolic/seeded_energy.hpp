#pragma once

#include "symbolic/expressions.hpp" // CExpression, Numeric
#include "symbolic/traits.hpp"      // extract_symbols_from_expr_t

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

#include "symbolic/symbol.hpp" // mp_size

namespace ddx::impl {

// Bridges an expression graph into the numeric Hessian driver: seeded dofs
// arrive in sorted symbol order, and the graph is walked once by eval_seeded.
template <CExpression Expr> class SeededExprEnergy {
  static_assert(
      !std::is_reference_v<Expr>,
      "SeededExprEnergy stores the expression by value on purpose — a "
      "reference parameter would leave the callable holding a "
      "dangling graph.  Use seeded_energy(), which decays.");
  Expr expr_;

public:
  using symbols = detail::expr_symbols_t<std::remove_cvref_t<Expr>>;
  static constexpr std::size_t arity = mp::mp_size<symbols>::value;

  static constexpr bool kSeededExprEnergy = true;

  explicit constexpr SeededExprEnergy(Expr expr) noexcept
      : expr_(static_cast<Expr &&>(expr)) {}

  template <Numeric Dof>
  [[nodiscard]] constexpr auto operator()(const Dof *dof) const noexcept {
    using T = std::remove_cvref_t<Dof>;
    // Built by copy: zeroing first would double the writes per probe.
    const auto s = index_apply<arity>(
        [&]<std::size_t... I>() { return std::array<T, arity>{dof[I]...}; });
    return expr_.template eval_seeded<symbols>(s);
  }
};

// The tag seeded_energy() puts on a bridged graph.  With the class it detects
// rather than the driver that reads it: "is this already seeded?" is a question
// about this header's type.
template <typename F>
concept CSeededExprEnergy =
    requires { requires std::remove_cvref_t<F>::kSeededExprEnergy; };

template <CExpression Expr>
[[nodiscard]] constexpr auto seeded_energy(Expr &&expr) noexcept {
  return SeededExprEnergy<std::remove_cvref_t<Expr>>(
      static_cast<Expr &&>(expr));
}

} // namespace ddx::impl
