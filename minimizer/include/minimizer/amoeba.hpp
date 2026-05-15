#pragma once

#include "expressions.hpp"
#include "traits.hpp"
#include <algorithm>
#include <array>
#include <boost/mp11/list.hpp>
#include <ranges>

namespace diff::min {

namespace mp = boost::mp11;

// NR §10.4 — Nelder-Mead downhill simplex method.
//
// Derivative-free N-dim minimizer.  Maintains a simplex of N+1 vertices,
// progressively shrinking it toward the minimum via reflection, expansion,
// and contraction moves.  No line search; no gradient required.
//
// amotry convention (NR §10.4):  psum = sum of all N+1 vertices; fac=-1
// reflects, fac=2 expands, fac=0.5 contracts.  Derivation:
//   centroid c = (psum − s[ihi]) / N
//   ptry = psum·fac1 − s[ihi]·fac2,  fac1=(1−fac)/N, fac2=fac1−fac
// which gives ptry = c + fac·(c − s[ihi]) for each sign of fac.
template <diff::CExpression Expr> struct Amoeba {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = std::array<value_type, N>;
  using Simplex = std::array<Point, N + 1>;
  using FVals = std::array<value_type, N + 1>;

  static constexpr value_type TINY{1.0e-10};
  static constexpr int ITMAX = 5000;

  Expr expr;
  value_type fret{};
  int iter{};
  const value_type ftol;

  constexpr explicit Amoeba(Expr e,
                            value_type ftol_ = static_cast<value_type>(3.0e-8))
      : expr(std::move(e)), ftol(ftol_) {}

  constexpr value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

  // Build simplex from a single starting point and uniform step size.
  constexpr Point minimize(const Point &p, const value_type &delta) {
    Simplex s;
    s[0] = p;
    for (auto i : std::views::iota(0uz, N)) {
      s[i + 1] = p;
      s[i + 1][i] += delta;
    }
    return minimize(std::move(s));
  }

  constexpr Point minimize(Simplex s) {
    FVals y;
    for (auto &&[yi, si] : std::views::zip(y, s)) {
      yi = eval_at(si);
    }

    Point psum{};
    for (const auto &si : s) {
      std::ranges::transform(psum, si, psum.begin(), std::plus<>{});
    }

    for (iter = 0; iter < ITMAX; ++iter) {
      // Indices of best (ilo), worst (ihi), second-worst (inhi)
      const std::size_t ilo =
          static_cast<std::size_t>(std::ranges::min_element(y) - y.begin());

      std::size_t ihi = (y[0] > y[1]) ? 0uz : 1uz;
      std::size_t inhi = 1uz - ihi;
      for (auto i : std::views::iota(2uz, N + 1)) {
        if (y[i] > y[ihi]) {
          inhi = ihi;
          ihi = i;
        } else if (y[i] > y[inhi] && i != ihi) {
          inhi = i;
        }
      }

      // Convergence: relative spread between best and worst
      if (value_type{2} * std::abs(y[ihi] - y[ilo]) /
              (std::abs(y[ihi]) + std::abs(y[ilo]) + TINY) <
          ftol) {
        fret = y[ilo];
        return s[ilo];
      }

      value_type ytry = amotry(s, y, psum, ihi, value_type{-1});
      if (ytry <= y[ilo]) {
        amotry(s, y, psum, ihi,
               value_type{2}); // reflection good — try expansion
      } else if (ytry >= y[inhi]) {
        const value_type ysave = y[ihi];
        ytry = amotry(s, y, psum, ihi, value_type{0.5}); // contraction
        if (ytry >= ysave) {
          // Contraction failed — shrink whole simplex toward best
          for (auto [i, si] : std::views::enumerate(s)) {
            if (i != ilo) {
              std::ranges::transform(si, s[ilo], si.begin(),
                                     [](const auto &a, const auto &b) {
                                       return value_type{0.5} * (a + b);
                                     });
              y[i] = eval_at(si);
            }
          }
          psum = {};
          for (const auto &si : s) {
            std::ranges::transform(psum, si, psum.begin(), std::plus<>{});
          }
        }
      }
    }
    const auto ilo_it = std::ranges::min_element(y);
    fret = *ilo_it;
    return s[static_cast<std::size_t>(ilo_it - y.begin())];
  }

private:
  constexpr value_type amotry(Simplex &s, FVals &y, Point &psum,
                              const std::size_t ihi, const value_type &fac) {
    const value_type fac1 = (value_type{1} - fac) / static_cast<value_type>(N);
    const value_type fac2 = fac1 - fac;
    Point ptry{};
    std::ranges::transform(psum, s[ihi], ptry.begin(),
                           [fac1, fac2](const auto &ps, const auto &sh) {
                             return fac1 * ps - fac2 * sh;
                           });
    const value_type ytry = eval_at(ptry);
    if (ytry < y[ihi]) {
      std::ranges::transform(psum, ptry, psum.begin(), std::plus<>{});
      std::ranges::transform(psum, s[ihi], psum.begin(), std::minus<>{});
      s[ihi] = ptry;
      y[ihi] = ytry;
    }
    return ytry;
  }
};

template <diff::CExpression Expr> Amoeba(Expr) -> Amoeba<Expr>;
template <diff::CExpression Expr, typename T> Amoeba(Expr, T) -> Amoeba<Expr>;

} // namespace diff::min
