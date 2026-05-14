#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "expressions.hpp"
#include "traits.hpp"
#include <boost/mp11/list.hpp>

namespace diff::min {

namespace mp = boost::mp11;

// NR §10.1 — Bracketmethod adapted for expression templates.
//
// Instead of a functor f(x), the expression tree itself is the callable.
// eval_at(x) updates the single Variable in the tree via the existing
// .update(Syms{}, vals) + .eval() protocol, where Syms is deduced at
// compile time from extract_symbols_from_expr_t<Expr>.
template <diff::CExpression Expr> struct Bracketmethod {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;

  static_assert(mp::mp_size<Syms>::value == 1,
                "Bracketmethod requires a single-variable expression");

  Expr expr;
  value_type ax{}, bx{}, cx{};
  value_type fa{}, fb{}, fc{};

  explicit Bracketmethod(Expr e) : expr(std::move(e)) {}

  value_type eval_at(value_type x) {
    std::array<value_type, 1> v{x};
    expr.update(Syms{}, v);
    return expr.eval();
  }

  // Step downhill from (ax0, bx0) until the minimum is bracketed.
  void bracket(value_type ax0, value_type bx0) {
    constexpr value_type GOLD =
        static_cast<value_type>(std::numbers::phi_v<double>);
    constexpr value_type GLIMIT = static_cast<value_type>(100.0);
    constexpr value_type TINY = static_cast<value_type>(1.0e-20);

    ax = ax0;
    bx = bx0;
    fa = eval_at(ax);
    fb = eval_at(bx);

    if (fb > fa) {
      using std::swap;
      swap(ax, bx);
      swap(fa, fb);
    }

    cx = bx + GOLD * (bx - ax);
    fc = eval_at(cx);

    while (fb > fc) {
      value_type r = (bx - ax) * (fb - fc);
      value_type q = (bx - cx) * (fb - fa);

      // Sign-safe denominator to avoid division-by-zero (NR: TINY guard).
      value_type qdiff = q - r;
      value_type denom = static_cast<value_type>(2.0) *
                         (qdiff >= value_type{} ? static_cast<value_type>(1)
                                                : static_cast<value_type>(-1)) *
                         std::max(std::abs(qdiff), TINY);

      value_type u = bx - ((bx - cx) * q - (bx - ax) * r) / denom;
      value_type ulim = bx + GLIMIT * (cx - bx);

      value_type fu{};
      if ((bx - u) * (u - cx) > value_type{}) {
        fu = eval_at(u);
        if (fu < fc) {
          ax = bx;
          fa = fb;
          bx = u;
          fb = fu;
          return;
        }
        if (fu > fb) {
          cx = u;
          fc = fu;
          return;
        }
        u = cx + GOLD * (cx - bx);
        fu = eval_at(u);
      } else if ((cx - u) * (u - ulim) > value_type{}) {
        fu = eval_at(u);
        if (fu < fc) {
          bx = cx;
          fb = fc;
          cx = u;
          fc = fu;
          u = cx + GOLD * (cx - bx);
          fu = eval_at(u);
        }
      } else if ((u - ulim) * (ulim - cx) >= value_type{}) {
        u = ulim;
        fu = eval_at(u);
      } else {
        u = cx + GOLD * (cx - bx);
        fu = eval_at(u);
      }

      ax = bx;
      bx = cx;
      cx = u;
      fa = fb;
      fb = fc;
      fc = fu;
    }
  }
};

template <diff::CExpression Expr> Bracketmethod(Expr) -> Bracketmethod<Expr>;

} // namespace diff::min
