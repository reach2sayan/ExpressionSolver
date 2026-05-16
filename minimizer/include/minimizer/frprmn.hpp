#pragma once

#include "dlinmin.hpp"
#include "gradient.hpp"
#include "linmin.hpp"
#include <Eigen/Dense>
#include <limits>
#include <utility>

namespace diff::min {

enum class CGMethod { FletcherReeves, PolakRibiere };

// NR §10.6 — Conjugate Gradient (Fletcher-Reeves / Polak-Ribière).
//
// Gradient ∇f is obtained for free via reverse-mode AD on the expression tree.
// Line minimization is delegated to LM (default: LinMin/Brent; use DLinMin for
// derivative-aware Dbrent line searches).
template <diff::CExpression Expr,
          CGMethod Method = CGMethod::PolakRibiere,
          template <diff::CExpression> class LM = LinMin>
struct Frprmn {
  using LMType = LM<Expr>;
  using value_type = typename LMType::value_type;
  using Point = typename LMType::Point;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = LMType::N;

  static constexpr diff::Constant<value_type> ZEPS{
      std::numeric_limits<value_type>::epsilon() * 1.0e-3};
  static constexpr int ITMAX = 200;

  LMType lm;
  value_type fret{};
  int iter{};
  const value_type ftol;

  constexpr explicit Frprmn(Expr e,
                            value_type ftol_ = static_cast<value_type>(3.0e-8))
      : lm(std::move(e), ftol_), ftol(ftol_) {}

  constexpr value_type eval_at(const Point &p) { return lm.eval_at(p); }

  // Returns {f(p), ∇f(p)}, updating expr state to p.
  constexpr std::pair<value_type, Point> eval_grad(const Point &p) {
    lm.expr.update(Syms{}, p);
    const auto g_arr = diff::gradient<diff::DiffMode::Reverse>(lm.expr);
    Point g = Eigen::Map<const Point>(g_arr.data());
    return {lm.expr.eval(), std::move(g)};
  }

  constexpr Point minimize(Point p) {
    using std::abs;

    auto [fp, g] = eval_grad(p);
    fret = fp;

    Point xi = -g;
    Point h = xi;

    for (iter = 0; iter < ITMAX; ++iter) {
      lm.minimize(p, xi);
      fret = lm.fret;

      if (value_type{2} * abs(fret - fp) <=
          ftol * (abs(fret) + abs(fp)) + ZEPS.get()) {
        return p;
      }

      fp = fret;
      auto [_, g_new] = eval_grad(p);

      const value_type gg = g.squaredNorm();
      const value_type dgg = [&] {
        if constexpr (Method == CGMethod::PolakRibiere) {
          return (g_new - g).dot(g_new);
        } else {
          return g_new.squaredNorm();
        }
      }();

      if (gg == value_type{}) {
        return p;
      }
      const value_type gam = dgg / gg;

      h = -g_new + gam * h;
      xi = h;
      g = std::move(g_new);
    }
    return p;
  }
};

template <diff::CExpression Expr> Frprmn(Expr) -> Frprmn<Expr>;
template <diff::CExpression Expr, diff::Numeric T>
Frprmn(Expr, T) -> Frprmn<Expr>;

// Derivative-aware CG: uses Dbrent line search via DLinMin.
template <diff::CExpression Expr, CGMethod Method = CGMethod::PolakRibiere>
using DFrprmn = Frprmn<Expr, Method, DLinMin>;

} // namespace diff::min
