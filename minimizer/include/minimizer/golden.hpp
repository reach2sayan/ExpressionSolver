#pragma once

#include "bracketmethod.hpp"
#include <cmath>

namespace diff::min {

// NR §10.2 — Golden section search built on Bracketmethod.
//
// Each iteration reduces the bracket by the golden factor 0.61803.
// Convergence is linear; tol should not be smaller than sqrt(machine epsilon)
// (~3e-8 for double). The minimum abscissa is in xmin, function value in fmin.
template <diff::CExpression Expr> struct Golden : Bracketmethod<Expr> {
  using Base = Bracketmethod<Expr>;
  using value_type = typename Base::value_type;
  using Base::ax;
  using Base::bracket;
  using Base::bx;
  using Base::cx;
  using Base::eval_at;
  using Base::fa;
  using Base::fb;
  using Base::fc;

  // R = 1/φ ≈ 0.61803 — contraction ratio (NOT φ itself)
  static constexpr diff::Constant<value_type> R{1.0 /
                                                std::numbers::phi_v<double>};
  static constexpr diff::Constant<value_type> C{
      1.0 - 1.0 / std::numbers::phi_v<double>};

  value_type xmin{};
  value_type fmin{};
  const value_type tol;

  constexpr explicit Golden(Expr e,
                            value_type tol_ = static_cast<value_type>(3.0e-8))
      : Base(std::move(e)), tol(tol_) {}

  // Perform golden section search on the already-bracketed triplet
  // (ax, bx, cx).  Call bracket() first, or set the triplet manually.
  constexpr value_type minimize() {
    using std::abs;

    value_type x0 = ax, x3 = cx, x1{}, x2{};
    if (abs(cx - bx) > abs(bx - ax)) {
      x1 = bx;
      x2 = bx + C * (cx - bx);
    } else {
      x2 = bx;
      x1 = bx - C * (bx - ax);
    }

    auto f1 = eval_at(x1);
    auto f2 = eval_at(x2);

    while (abs(x3 - x0) > tol * (abs(x1) + abs(x2))) {
      if (f2 < f1) {
        x0 = x1;
        x1 = x2;
        x2 = R * x2 + C * x3;
        f1 = f2;
        f2 = eval_at(x2);
      } else {
        x3 = x2;
        x2 = x1;
        x1 = R * x1 + C * x0;
        f2 = f1;
        f1 = eval_at(x1);
      }
    }

    if (f1 < f2) {
      xmin = x1;
      fmin = f1;
    } else {
      xmin = x2;
      fmin = f2;
    }
    return xmin;
  }

  // Convenience: bracket from (ax0, bx0) then minimize.
  constexpr value_type minimize(const value_type &ax0, const value_type &bx0) {
    bracket(ax0, bx0);
    return minimize();
  }
};

template <diff::CExpression Expr> Golden(Expr) -> Golden<Expr>;
template <diff::CExpression Expr, typename T> Golden(Expr, T) -> Golden<Expr>;

} // namespace diff::min
