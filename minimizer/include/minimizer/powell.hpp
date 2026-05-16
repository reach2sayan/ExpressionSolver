#pragma once

#include "linmin.hpp"

namespace diff::min {

// NR §10.5 — Powell's method for N-dimensional minimization.
//
// Successively line-minimizes along N conjugate directions, initially the
// coordinate axes.  Each outer iteration replaces the direction of largest
// decrease with the net step direction, progressively building a set of
// mutually conjugate directions.  No derivatives required.
template <diff::CExpression Expr> struct Powell {
  using value_type = typename LinMin<Expr>::value_type;
  using Point = typename LinMin<Expr>::Point;
  static constexpr std::size_t N = LinMin<Expr>::N;
  // Each column is one search direction; starts as identity (coordinate axes).
  using Dirs = Eigen::Matrix<value_type, static_cast<int>(N), static_cast<int>(N)>;

  static constexpr diff::Constant<value_type> FTINY{1.0e-25};
  static constexpr int ITMAX = 200;

  LinMin<Expr> lm;
  value_type fret{};
  int iter{};
  const value_type ftol;

  constexpr explicit Powell(Expr e,
                            value_type ftol_ = static_cast<value_type>(3.0e-8))
      : lm(std::move(e), ftol_), ftol(ftol_) {}

  constexpr value_type eval_at(const Point &p) { return lm.eval_at(p); }

  // Minimize from p using coordinate-axis initial directions.
  constexpr Point minimize(Point p) {
    return minimize(std::move(p), Dirs::Identity());
  }

  // Minimize from p with explicit initial direction set xi (columns = directions).
  constexpr Point minimize(Point p, Dirs xi) {
    using std::abs;
    fret = eval_at(p);
    Point pt = p;

    for (iter = 0; iter < ITMAX; ++iter) {
      const value_type fp = fret;
      int ibig = 0;
      value_type del{};

      for (int i = 0; i < static_cast<int>(N); ++i) {
        const value_type fptt = fret;
        lm.minimize(p, xi.col(i));
        fret = lm.fret;
        if (abs(fptt - fret) > del) {
          del = abs(fptt - fret);
          ibig = i;
        }
      }

      if (value_type{2} * abs(fp - fret) <=
          ftol * (abs(fp) + abs(fret)) + FTINY.get()) {
        return p;
      }

      const Point ptt = value_type{2} * p - pt;
      Point xit = p - pt;
      pt = p;

      const value_type fptt = eval_at(ptt);
      if (fptt < fp) {
        const value_type a = fp - fret - del;
        const value_type b = fp - fptt;
        if (value_type{2} * (fp - value_type{2} * fret + fptt) * a * a <
            b * b * del) {
          lm.minimize(p, xit);
          fret = lm.fret;
          xi.col(ibig) = xi.col(static_cast<int>(N) - 1);
          xi.col(static_cast<int>(N) - 1) = xit;
        }
      }
    }
    return p;
  }
};

template <diff::CExpression Expr> Powell(Expr) -> Powell<Expr>;
template <diff::CExpression Expr, typename T> Powell(Expr, T) -> Powell<Expr>;

} // namespace diff::min
