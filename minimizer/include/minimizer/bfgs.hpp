#pragma once

#include "dlinmin.hpp"
#include "gradient.hpp"
#include "linmin.hpp"
#include <Eigen/Dense>
#include <limits>
#include <utility>

namespace diff::min {

// NR §10.7 — BFGS quasi-Newton minimization.
//
// Maintains an N×N approximation to the inverse Hessian, updated via the
// rank-2 BFGS formula after each step.  Line minimization is delegated to
// LM (default: LinMin/Brent; use DLinMin for derivative-aware Dbrent).
// Convergence criterion: max |∂f/∂xᵢ| · max(|xᵢ|,1) / max(|f|,1) < gtol.
template <diff::CExpression Expr,
          template <diff::CExpression> class LM = LinMin>
struct BFGS {
  using LMType = LM<Expr>;
  using value_type = typename LMType::value_type;
  using Point = typename LMType::Point;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = LMType::N;
  using Hessian = Eigen::Matrix<value_type, static_cast<int>(N), static_cast<int>(N)>;

  static constexpr int ITMAX = 200;

  LMType lm;
  value_type fret{};
  int iter{};
  const value_type gtol;

  constexpr explicit BFGS(Expr e,
                          value_type gtol_ = static_cast<value_type>(1.0e-8))
      : lm(std::move(e), gtol_), gtol(gtol_) {}

  value_type eval_at(const Point &p) { return lm.eval_at(p); }

  // Returns {f(p), ∇f(p)}, updating expr state to p.
  std::pair<value_type, Point> eval_grad(const Point &p) {
    lm.expr.update(Syms{}, p);
    const auto g_arr = diff::gradient<diff::DiffMode::Reverse>(lm.expr);
    Point g = Eigen::Map<const Point>(g_arr.data());
    return {lm.expr.eval(), std::move(g)};
  }

  Point minimize(Point p) {
    using std::abs, std::max;
    constexpr auto EPS = std::numeric_limits<value_type>::epsilon();

    Hessian hsn = Hessian::Identity();
    auto [fp, g] = eval_grad(p);
    fret = fp;
    Point xi = -g;

    for (iter = 0; iter < ITMAX; ++iter) {
      lm.minimize(p, xi); // xi ← actual step Δp;  p ← p_new
      fret = lm.fret;

      auto [_, g_new] = eval_grad(p);

      // Gradient-based convergence (NR §10.7)
      const value_type den = max(abs(fret), value_type{1});
      const value_type test =
          (g_new.cwiseAbs().array() *
           p.cwiseAbs().cwiseMax(value_type{1}).array())
              .maxCoeff() /
          den;
      if (test < gtol)
        return p;

      Point dg = g_new - g;
      const Point hdg = hsn * dg;

      value_type fac = dg.dot(xi);
      const value_type fae = dg.dot(hdg);
      const value_type sumdg = dg.squaredNorm();
      const value_type sumxi = xi.squaredNorm();

      // BFGS rank-2 update — skip if curvature condition fails
      if (fac > value_type{} && fac * fac > EPS * sumdg * sumxi) {
        fac = value_type{1} / fac;
        const value_type fad = value_type{1} / fae;

        dg = fac * xi - fad * hdg; // u = fac·Δp − fad·H·Δg

        hsn += fac * xi * xi.transpose();
        hsn -= fad * hdg * hdg.transpose();
        hsn += fae * dg * dg.transpose();
      }

      xi = -(hsn * g_new);
      g = std::move(g_new);
      fp = fret;
    }
    return p;
  }
};

template <diff::CExpression Expr> BFGS(Expr) -> BFGS<Expr>;
template <diff::CExpression Expr, typename T> BFGS(Expr, T) -> BFGS<Expr>;

// Derivative-aware BFGS: uses Dbrent line search via DLinMin.
template <diff::CExpression Expr>
using DBFGS = BFGS<Expr, DLinMin>;

} // namespace diff::min
