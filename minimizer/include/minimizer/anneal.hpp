#pragma once

#include "detail.hpp"
#include "expressions.hpp"
#include "traits.hpp"
#include <algorithm>
#include <array>
#include <boost/mp11/list.hpp>
#include <cmath>
#include <limits>
#include <random>
#include <ranges>

namespace diff::min {

namespace mp = boost::mp11;

// NR §10.12.2 — Simulated annealing by modified downhill simplex, followed by
// an exact Amoeba cold refinement from the best SA point.
//
// Hot phase: each vertex's comparison value yy[i] = y[i] + T·log(U), U~Uniform.
// Since log(U) ≤ 0 the noise shrinks stored values; as T→0 this vanishes.
// Uphill moves are accepted with probability ∝ exp(−ΔE/T), enabling escape from
// shallow local minima.  Schedule: T ← T·cooling every epoch_steps iterations.
//
// Cold phase: once T < TINY the simplex is reset at the best SA point with
// delta = cold_delta and standard Amoeba (ftol convergence) takes over.
// This matches NR's recommendation to "restart at the claimed minimum."
template <diff::CExpression Expr> struct SimAnneal {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = std::array<value_type, N>;
  using Simplex = std::array<Point, N + 1>;
  using FVals = std::array<value_type, N + 1>;

  static constexpr value_type TINY{1.0e-10};
  static constexpr int NMAX = 200'000;

  Expr expr;
  value_type fret{};
  int iter{};
  value_type temperature;
  const value_type cooling;
  const int epoch_steps;
  const value_type ftol;
  const value_type cold_delta;

  explicit SimAnneal(Expr e, value_type T0 = value_type{1},
                     value_type cool = static_cast<value_type>(0.9),
                     int epoch = 100,
                     value_type ftol_ = static_cast<value_type>(3.0e-8),
                     value_type cold_delta_ = static_cast<value_type>(0.1))
      : expr(std::move(e)), temperature(T0), cooling(cool), epoch_steps(epoch),
        ftol(ftol_), cold_delta(cold_delta_) {}

  constexpr value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

  constexpr Point minimize(const Point &p, const value_type &delta) {
    return minimize(detail::make_simplex(p, delta));
  }

  Point minimize(Simplex s) {
    std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<value_type> udist{
        std::numeric_limits<value_type>::epsilon(), value_type{1}};

    auto bolt = [&]() -> value_type {
      return temperature * std::log(udist(rng));
    };

    // ── Hot SA phase ──────────────────────────────────────────────────────
    FVals y{}, yy{};
    for (auto i : std::views::iota(0uz, N + 1)) {
      y[i] = eval_at(s[i]);
      yy[i] = y[i] + bolt();
    }

    std::size_t ib =
        static_cast<std::size_t>(std::ranges::min_element(y) - y.begin());
    value_type ybest = y[ib];
    Point pbest = s[ib];

    Point psum{};
    for (const auto &si : s)
      detail::zip_inplace(psum, si, std::plus<>{});

    for (iter = 0; iter < NMAX && temperature > TINY; ++iter) {
      if (iter > 0 && iter % epoch_steps == 0) {
        temperature *= cooling;
        for (auto i : std::views::iota(0uz, N + 1))
          yy[i] = y[i] + bolt();
      }

      const std::size_t ilo =
          static_cast<std::size_t>(std::ranges::min_element(yy) - yy.begin());
      std::size_t ihi = (yy[0] > yy[1]) ? 0uz : 1uz;
      std::size_t inhi = 1uz - ihi;
      for (auto i : std::views::iota(2uz, N + 1)) {
        if (yy[i] > yy[ihi]) {
          inhi = ihi;
          ihi = i;
        } else if (yy[i] > yy[inhi] && i != ihi) {
          inhi = i;
        }
      }

      for (auto i : std::views::iota(0uz, N + 1)) {
        if (y[i] < ybest) {
          ybest = y[i];
          pbest = s[i];
        }
      }

      value_type ytry = amotry(s, y, yy, psum, ihi, bolt, value_type{-1});
      if (ytry <= yy[ilo]) {
        amotry(s, y, yy, psum, ihi, bolt, value_type{2});
      } else if (ytry >= yy[inhi]) {
        const value_type ysave = yy[ihi];
        ytry = amotry(s, y, yy, psum, ihi, bolt, value_type{0.5});
        if (ytry >= ysave) {
          for (auto [i, si] : std::views::enumerate(s)) {
            if (static_cast<std::size_t>(i) != ilo) {
              detail::zip_inplace(si, s[ilo], [](const auto &a, const auto &b) {
                return value_type{0.5} * (a + b);
              });
              y[i] = eval_at(si);
              yy[i] = y[i] + bolt();
            }
          }
          psum = {};
          for (const auto &si : s)
            detail::zip_inplace(psum, si, std::plus<>{});
        }
      }
    }

    // ── Cold Amoeba refinement from best SA point ─────────────────────────
    // Rebuild a fresh, non-degenerate simplex at pbest to avoid the
    // collapsed-simplex false-convergence that the hot phase can cause.
    s = detail::make_simplex(pbest, cold_delta);
    for (auto i : std::views::iota(0uz, N + 1))
      y[i] = eval_at(s[i]);

    psum = {};
    for (const auto &si : s)
      detail::zip_inplace(psum, si, std::plus<>{});

    static constexpr value_type ATINY{1.0e-20};
    for (int cold = 0; cold < NMAX; ++cold, ++iter) {
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

      if (y[ilo] < ybest) {
        ybest = y[ilo];
        pbest = s[ilo];
      }

      const value_type denom = std::abs(y[ihi]) + std::abs(y[ilo]) + ATINY;
      if (value_type{2} * std::abs(y[ihi] - y[ilo]) / denom < ftol)
        break;

      value_type ytry = amotry_cold(s, y, psum, ihi, value_type{-1});
      if (ytry <= y[ilo]) {
        amotry_cold(s, y, psum, ihi, value_type{2});
      } else if (ytry >= y[inhi]) {
        const value_type ysave = y[ihi];
        ytry = amotry_cold(s, y, psum, ihi, value_type{0.5});
        if (ytry >= ysave) {
          for (auto [i, si] : std::views::enumerate(s)) {
            if (static_cast<std::size_t>(i) != ilo) {
              detail::zip_inplace(si, s[ilo], [](const auto &a, const auto &b) {
                return value_type{0.5} * (a + b);
              });
              y[i] = eval_at(si);
            }
          }
          psum = {};
          for (const auto &si : s)
            detail::zip_inplace(psum, si, std::plus<>{});
        }
      }
    }

    const std::size_t ilo_f =
        static_cast<std::size_t>(std::ranges::min_element(y) - y.begin());
    if (y[ilo_f] < ybest) {
      ybest = y[ilo_f];
      pbest = s[ilo_f];
    }

    fret = ybest;
    return pbest;
  }

private:
  template <std::invocable<> Bolt>
  value_type amotry(Simplex &s, FVals &y, FVals &yy, Point &psum,
                    const std::size_t ihi, Bolt &bolt, const value_type &fac) {
    const value_type fac1 = (value_type{1} - fac) / static_cast<value_type>(N);
    const value_type fac2 = fac1 - fac;
    Point ptry{};
    std::ranges::transform(psum, s[ihi], ptry.begin(),
                           [fac1, fac2](const auto &ps, const auto &sh) {
                             return fac1 * ps - fac2 * sh;
                           });
    const value_type ytry_real = eval_at(ptry);
    const value_type ytry = ytry_real + bolt();
    if (ytry < yy[ihi]) {
      detail::zip_inplace(psum, ptry, std::plus<>{});
      detail::zip_inplace(psum, s[ihi], std::minus<>{});
      s[ihi] = ptry;
      y[ihi] = ytry_real;
      yy[ihi] = ytry;
    }
    return ytry;
  }

  value_type amotry_cold(Simplex &s, FVals &y, Point &psum,
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
      detail::zip_inplace(psum, ptry, std::plus<>{});
      detail::zip_inplace(psum, s[ihi], std::minus<>{});
      s[ihi] = ptry;
      y[ihi] = ytry;
    }
    return ytry;
  }
};

template <diff::CExpression Expr> SimAnneal(Expr) -> SimAnneal<Expr>;
template <diff::CExpression Expr, typename T>
SimAnneal(Expr, T) -> SimAnneal<Expr>;
template <diff::CExpression Expr, typename T1, typename T2>
SimAnneal(Expr, T1, T2) -> SimAnneal<Expr>;
template <diff::CExpression Expr, typename T1, typename T2, typename I>
SimAnneal(Expr, T1, T2, I) -> SimAnneal<Expr>;
template <diff::CExpression Expr, typename T1, typename T2, typename I,
          typename T3>
SimAnneal(Expr, T1, T2, I, T3) -> SimAnneal<Expr>;
template <diff::CExpression Expr, typename T1, typename T2, typename I,
          typename T3, typename T4>
SimAnneal(Expr, T1, T2, I, T3, T4) -> SimAnneal<Expr>;

} // namespace diff::min
