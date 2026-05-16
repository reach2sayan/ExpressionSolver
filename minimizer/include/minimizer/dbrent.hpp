#pragma once

#include "bracketmethod.hpp"
#include "detail.hpp"
#include "gradient.hpp"
#include <limits>

namespace diff::min {

// NR §10.4 — Brent's method with first-derivative information.
//
// Mirrors Brent but replaces parabolic interpolation with secant interpolation
// on f′, giving superlinear convergence near a smooth minimum.  The derivative
// is obtained for free from the expression's reverse-mode AD.
//
// Requires a single-variable expression (same restriction as Bracketmethod).
template <diff::CExpression Expr>
struct Dbrent : Bracketmethod<Expr> {
  using Base = Bracketmethod<Expr>;
  using value_type = typename Base::value_type;
  using Syms = typename Base::Syms;
  using Base::ax;
  using Base::bx;
  using Base::cx;
  using Base::bracket;
  using Base::eval_at;
  using Base::expr;

  static constexpr value_type ZEPS =
      std::numeric_limits<value_type>::epsilon() * static_cast<value_type>(1.0e-3);
  static constexpr int ITMAX = 100;

  value_type xmin{};
  value_type fmin{};
  const value_type tol;

  constexpr explicit Dbrent(Expr e,
                            value_type tol_ = static_cast<value_type>(3.0e-8))
      : Base(std::move(e)), tol(tol_) {}

  value_type minimize() {
    struct Funcd {
      Dbrent &self;
      value_type operator()(value_type t) { return self.eval_at(t); }
      value_type df(value_type t) {
        std::array<value_type, 1> v{t};
        self.expr.update(Syms{}, v);
        const auto g = diff::gradient<diff::DiffMode::Reverse>(self.expr);
        return g[0];
      }
    };
    Funcd fc{*this};
    xmin = detail::dbrent(fc, ax, bx, cx, tol, ZEPS, ITMAX);
    fmin = eval_at(xmin);
    return xmin;
  }

  value_type minimize(const value_type &ax0, const value_type &bx0) {
    bracket(ax0, bx0);
    return minimize();
  }
};

template <diff::CExpression Expr> Dbrent(Expr) -> Dbrent<Expr>;
template <diff::CExpression Expr, typename T> Dbrent(Expr, T) -> Dbrent<Expr>;

} // namespace diff::min
