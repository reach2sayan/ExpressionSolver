#pragma once

#include "gradient.hpp"
#include "linmin.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

namespace diff::min {

enum class CGMethod { FletcherReeves, PolakRibiere };

// NR §10.6 — Conjugate Gradient (Fletcher-Reeves / Polak-Ribière).
//
// Gradient ∇f is obtained for free via reverse-mode AD on the expression tree.
// Line minimization along each conjugate direction is delegated to LinMin
// (Brent). Default method is Polak-Ribière; NR recommends it over
// Fletcher-Reeves.
template <diff::CExpression Expr, CGMethod Method = CGMethod::PolakRibiere>
struct Frprmn {
  using value_type = typename LinMin<Expr>::value_type;
  using Point = typename LinMin<Expr>::Point;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = LinMin<Expr>::N;

  static constexpr diff::Constant<value_type> ZEPS{
      std::numeric_limits<value_type>::epsilon() * 1.0e-3};
  static constexpr int ITMAX = 200;

  LinMin<Expr> lm;
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
    return {lm.expr.eval(), diff::gradient<diff::DiffMode::Reverse>(lm.expr)};
  }

  constexpr Point minimize(Point p) {
    using std::abs;

    auto [fp, g] = eval_grad(p);
    fret = fp;

    Point xi{}, h{};
    std::ranges::transform(g, xi.begin(), std::negate<>{});
    h = xi;

    for (iter = 0; iter < ITMAX; ++iter) {
      lm.minimize(p, xi);
      fret = lm.fret;

      if (value_type{2} * abs(fret - fp) <=
          ftol * (abs(fret) + abs(fp)) + ZEPS.get()) {
        return p;
      }

      fp = fret;
      auto [_, g_new] = eval_grad(p);

      const value_type gg =
          std::inner_product(g.begin(), g.end(), g.begin(), value_type{});
      const value_type dgg = [&] {
        if constexpr (Method == CGMethod::PolakRibiere) {
          return std::inner_product(
              g_new.begin(), g_new.end(), g.begin(), value_type{},
              std::plus<>{},
              [](const auto &gnj, const auto &gj) { return (gnj - gj) * gnj; });
        }
        else {
          return std::inner_product(g_new.begin(), g_new.end(), g_new.begin(),
                                    value_type{});
        }
      }();

      if (gg == value_type{}) {
        return p;
      }
      const value_type gam = dgg / gg;

      // h = −g_new + γ·h;  xi = h;  g = g_new
      std::ranges::transform(
          h, g_new, h.begin(),
          [gam](const auto &hj, const auto &gnj) { return -gnj + gam * hj; });
      xi = h;
      g = std::move(g_new);
    }
    return p;
  }
};

template <diff::CExpression Expr> Frprmn(Expr) -> Frprmn<Expr>;
template <diff::CExpression Expr, diff::Numeric T>
Frprmn(Expr, T) -> Frprmn<Expr>;

} // namespace diff::min
