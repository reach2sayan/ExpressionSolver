#pragma once

#include "gradient.hpp"
#include "linmin.hpp"
#include <algorithm>
#include <limits>
#include <ranges>
#include <utility>

namespace diff::min {

// NR §10.7 — BFGS quasi-Newton minimization.
//
// Maintains an N×N approximation to the inverse Hessian, updated via the
// rank-2 BFGS formula after each step.  Line minimization along the Newton
// direction is delegated to LinMin (Brent); the step vector it returns is
// exactly Δp, which feeds directly into the Hessian update.
// Convergence criterion: max |∂f/∂xᵢ| · max(|xᵢ|,1) / max(|f|,1) < gtol.
template <diff::CExpression Expr> struct BFGS {
  using value_type = typename LinMin<Expr>::value_type;
  using Point = typename LinMin<Expr>::Point;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  using Hessian = std::array<Point, LinMin<Expr>::N>;
  static constexpr std::size_t N = LinMin<Expr>::N;

  static constexpr int ITMAX = 200;

  LinMin<Expr> lm;
  value_type fret{};
  int iter{};
  const value_type gtol;

  constexpr explicit BFGS(Expr e,
                          value_type gtol_ = static_cast<value_type>(1.0e-8))
      : lm(std::move(e), gtol_), gtol(gtol_) {}

  constexpr value_type eval_at(const Point &p) { return lm.eval_at(p); }

  // Returns {f(p), ∇f(p)}, updating expr state to p.
  constexpr std::pair<value_type, Point> eval_grad(const Point &p) {
    lm.expr.update(Syms{}, p);
    return {lm.expr.eval(), diff::gradient<diff::DiffMode::Reverse>(lm.expr)};
  }

  constexpr Point minimize(Point p) {
    using std::abs, std::max;
    constexpr auto EPS = std::numeric_limits<value_type>::epsilon();

    Hessian hsn{};
    for (auto i : std::views::iota(0uz, N)) {
      hsn[i][i] = value_type{1};
    }

    auto [fp, g] = eval_grad(p);
    fret = fp;

    Point xi{};
    std::ranges::transform(g, xi.begin(), std::negate<>{});

    for (iter = 0; iter < ITMAX; ++iter) {
      lm.minimize(p, xi); // xi ← actual step Δp;  p ← p_new
      fret = lm.fret;

      auto [_, g_new] = eval_grad(p);

      // Gradient-based convergence (NR §10.7)
      value_type test{};
      const value_type den = max(abs(fret), value_type{1});
      for (auto &&[pj, gnj] : std::views::zip(p, g_new)) {
        test = max(test, abs(gnj) * max(abs(pj), value_type{1}) / den);
      }
      if (test < gtol) {
        return p;
      }

      // dg = Δg = g_new − g
      Point dg{};
      std::ranges::transform(g_new, g, dg.begin(), std::minus<>{});

      // hdg = H · dg
      Point hdg = detail::mat_vec(hsn, dg);

      value_type fac = detail::dot(dg, xi);
      const value_type fae = detail::dot(dg, hdg);
      const value_type sumdg = detail::norm_sq(dg);
      const value_type sumxi = detail::norm_sq(xi);

      // BFGS rank-2 update — skip if curvature condition fails (avoids sqrt)
      if (fac > value_type{} && fac * fac > EPS * sumdg * sumxi) {
        fac = value_type{1} / fac;
        const value_type fad = value_type{1} / fae;

        // u = fac·Δp − fad·(H·Δg), stored in dg
        std::ranges::transform(xi, hdg, dg.begin(),
                               [fac, fad](const auto &xij, const auto &hdgj) {
                                 return fac * xij - fad * hdgj;
                               });

        // H ← H + fac·(Δp⊗Δp) − fad·(HΔg⊗HΔg) + fae·(u⊗u)
        detail::rank1_update(hsn, fac, xi, xi);
        detail::rank1_update(hsn, -fad, hdg, hdg);
        detail::rank1_update(hsn, fae, dg, dg);
      }

      // New direction: xi = −H · g_new
      xi = detail::mat_vec(hsn, g_new);
      std::ranges::transform(xi, xi.begin(), std::negate<>{});

      g = std::move(g_new);
      fp = fret;
    }
    return p;
  }
};

template <diff::CExpression Expr> BFGS(Expr) -> BFGS<Expr>;
template <diff::CExpression Expr, typename T> BFGS(Expr, T) -> BFGS<Expr>;

} // namespace diff::min
