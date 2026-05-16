#pragma once

#include "detail.hpp"
#include "expressions.hpp"
#include "gradient.hpp"
#include "traits.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <limits>

namespace diff::min {

namespace mp = boost::mp11;

// Derivative-aware line minimizer — uses detail::dbrent (secant on f′)
// instead of detail::brent (parabolic interpolation).
//
// The directional derivative df/dt along dir is obtained via reverse-mode AD:
//   df/dt = ∇f(p + t·dir) · dir
//
// Drop-in replacement for LinMin: same Point type, same minimize() signature.
template <diff::CExpression Expr> struct DLinMin {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;

  static constexpr value_type ZEPS =
      std::numeric_limits<value_type>::epsilon() * static_cast<value_type>(1.0e-3);
  static constexpr int ITMAX = 100;

  Expr expr;
  value_type fret{};
  const value_type tol;

  constexpr explicit DLinMin(Expr e,
                             value_type tol_ = static_cast<value_type>(3.0e-8))
      : expr(std::move(e)), tol(tol_) {}

  value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

  void minimize(Point &p, Eigen::Ref<Point> dir) {
    // Capture the base point before modification; dir may be a matrix column.
    const Point p0 = p;
    const Point d0 = dir;  // evaluated copy of direction

    struct Funcd {
      DLinMin &self;
      const Point &p0;
      const Point &d0;

      value_type operator()(value_type t) {
        self.expr.update(Syms{}, p0 + t * d0);
        return self.expr.eval();
      }
      value_type df(value_type t) {
        self.expr.update(Syms{}, p0 + t * d0);
        const auto g_arr = diff::gradient<diff::DiffMode::Reverse>(self.expr);
        return Eigen::Map<const Point>(g_arr.data()).dot(d0);
      }
    };
    Funcd fc{*this, p0, d0};

    value_type ax{0}, bx{1}, cx;
    value_type fa = fc(ax), fb = fc(bx), fc_val;
    detail::bracket(fc, ax, bx, cx, fa, fb, fc_val);
    const value_type xmin = detail::dbrent(fc, ax, bx, cx, tol, ZEPS, ITMAX);

    dir *= xmin;
    p   += dir;
    fret = eval_at(p);
  }
};

template <diff::CExpression Expr> DLinMin(Expr) -> DLinMin<Expr>;
template <diff::CExpression Expr, typename T> DLinMin(Expr, T) -> DLinMin<Expr>;

} // namespace diff::min
