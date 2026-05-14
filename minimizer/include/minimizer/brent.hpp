#pragma once

#include "bracketmethod.hpp"
#include "detail.hpp"
#include <limits>

namespace diff::min {

// NR §10.3 — Brent's method: parabolic interpolation with golden section
// fallback.  Algorithm lives in detail::brent; this struct wires it to the
// expression-template eval_at protocol.
template <diff::CExpression Expr> struct Brent : Bracketmethod<Expr> {
  using Base = Bracketmethod<Expr>;
  using value_type = typename Base::value_type;
  using Base::ax;
  using Base::bracket;
  using Base::bx;
  using Base::cx;
  using Base::eval_at;

  static constexpr diff::Constant<value_type> ZEPS{
      std::numeric_limits<value_type>::epsilon() * 1.0e-3};
  static constexpr int ITMAX = 100;

  value_type xmin{};
  value_type fmin{};
  const value_type tol;

  constexpr explicit Brent(Expr e,
                           value_type tol_ = static_cast<value_type>(3.0e-8))
      : Base(std::move(e)), tol(tol_) {}

  constexpr value_type minimize() {
    auto f = [this](const value_type &x) { return eval_at(x); };
    xmin = detail::brent(f, ax, bx, cx, tol, ZEPS.get(), ITMAX);
    fmin = eval_at(xmin);
    return xmin;
  }

  constexpr value_type minimize(const value_type &ax0, const value_type &bx0) {
    bracket(ax0, bx0);
    return minimize();
  }
};

template <diff::CExpression Expr> Brent(Expr) -> Brent<Expr>;
template <diff::CExpression Expr, typename T> Brent(Expr, T) -> Brent<Expr>;

} // namespace diff::min
