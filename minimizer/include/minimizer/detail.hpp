#pragma once

#include "expressions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace diff::min::detail {

// Build an (N+1)-vertex simplex: s[0]=p, s[i+1]=p with s[i+1][i]+=delta
template <typename T, int N>
std::array<Eigen::Vector<T, N>, N + 1>
constexpr make_simplex(const Eigen::Vector<T, N> &p, const T &delta) noexcept {
  std::array<Eigen::Vector<T, N>, N + 1> s;
  std::fill(s.begin(), s.end(), p);
  for (int i = 0; i < N; ++i)
    s[i + 1][i] += delta;
  return s;
}

// NR §10.1 bracket algorithm for any callable T(T).
// fa = f(ax) and fb = f(bx) must be set by the caller before entry.
template <diff::Numeric T, std::invocable<T> F>
constexpr void bracket(F &f, T &ax, T &bx, T &cx, T &fa, T &fb, T &fc) {
  using std::abs;
  using std::max;
  using std::swap;
  constexpr T GOLD = static_cast<T>(std::numbers::phi_v<double>);
  constexpr T GLIMIT{100};
  constexpr T TINY{1.0e-20};

  if (fb > fa) {
    swap(ax, bx);
    swap(fa, fb);
  }
  cx = bx + GOLD * (bx - ax);
  fc = f(cx);

  while (fb > fc) {
    const T r = (bx - ax) * (fb - fc);
    const T q = (bx - cx) * (fb - fa);
    const T qdiff = q - r;
    const T denom =
        T{2} * (qdiff >= T{} ? T{1} : T{-1}) * max(abs(qdiff), TINY);
    T u = bx - ((bx - cx) * q - (bx - ax) * r) / denom;
    T ulim = bx + GLIMIT * (cx - bx);
    T fu;

    if ((bx - u) * (u - cx) > T{}) {
      fu = f(u);
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
      fu = f(u);
    } else if ((cx - u) * (u - ulim) > T{}) {
      fu = f(u);
      if (fu < fc) {
        bx = cx;
        fb = fc;
        cx = u;
        fc = fu;
        u = cx + GOLD * (cx - bx);
        fu = f(u);
      }
    } else if ((u - ulim) * (ulim - cx) >= T{}) {
      u = ulim;
      fu = f(u);
    } else {
      u = cx + GOLD * (cx - bx);
      fu = f(u);
    }
    ax = bx;
    bx = cx;
    cx = u;
    fa = fb;
    fb = fc;
    fc = fu;
  }
}

// NR §10.3 Brent's method for any callable T(T).
template <diff::Numeric T, std::invocable<T> F>
constexpr T brent(F &f, const T &ax, const T &bx, const T &cx, const T &tol,
                  const T &zeps, const int itmax = 100) {
  using std::abs;
  using std::max;
  using std::min;
  constexpr T CGOLD = static_cast<T>(1.0 - 1.0 / std::numbers::phi_v<double>);

  T a = min(ax, cx), b = max(ax, cx);
  T x = bx, w = bx, v = bx;
  T fx = f(x), fw = fx, fv = fx;
  T d{}, e{};

  for (int i = 0; i < itmax; ++i) {
    const T xm = T{0.5} * (a + b);
    const T tol1 = tol * abs(x) + zeps;
    const T tol2 = T{2} * tol1;

    if (abs(x - xm) <= tol2 - T{0.5} * (b - a)) {
      return x;
    }

    T u;
    if (abs(e) > tol1) {
      T r = (x - w) * (fx - fv);
      T q = (x - v) * (fx - fw);
      T p = (x - v) * q - (x - w) * r;
      q = T{2} * (q - r);
      if (q > T{}) {
        p = -p;
      }
      q = abs(q);
      const T etemp = e;
      e = d;
      if (abs(p) >= abs(T{0.5} * q * etemp) || p <= q * (a - x) ||
          p >= q * (b - x)) {
        e = (x >= xm ? a - x : b - x);
        d = CGOLD * e;
      } else {
        d = p / q;
        u = x + d;
        if (u - a < tol2 || b - u < tol2) {
          d = (xm >= x ? tol1 : -tol1);
        }
      }
    } else {
      e = (x >= xm ? a - x : b - x);
      d = CGOLD * e;
    }

    u = (abs(d) >= tol1 ? x + d : x + (d >= T{} ? tol1 : -tol1));
    const T fu = f(u);

    if (fu <= fx) {
      if (u < x) {
        b = x;
      } else {
        a = x;
      }
      v = w;
      fv = fw;
      w = x;
      fw = fx;
      x = u;
      fx = fu;
    } else {
      if (u < x) {
        a = u;
      } else {
        b = u;
      }
      if (fu <= fw || w == x) {
        v = w;
        fv = fw;
        w = u;
        fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u;
        fv = fu;
      }
    }
  }
  return x;
}

// NR §10.4 Dbrent — Brent's method with derivative information.
// Functor F must expose operator()(T) (value) and df(T) (derivative).
// Uses secant interpolation on f′; bisects toward the zero-crossing side
// when the secant step is outside the bracket or not improving.
template <diff::Numeric T, typename F>
constexpr T dbrent(F &f, const T &ax, const T &bx, const T &cx,
         const T &tol, const T &zeps, int itmax = 100) {
  using std::abs;

  T a = std::min(ax, cx), b = std::max(ax, cx);
  T x = bx, w = bx, v = bx;
  T fx = f(bx), fw = fx, fv = fx;
  T dx = f.df(bx), dw = dx, dv = dx;
  T d{}, e{};

  for (int i = 0; i < itmax; ++i) {
    const T xm   = T{0.5} * (a + b);
    const T tol1 = tol * abs(x) + zeps;
    const T tol2 = T{2} * tol1;
    if (abs(x - xm) <= tol2 - T{0.5} * (b - a))
      return x;

    if (abs(e) > tol1) {
      // Attempt secant steps using (x,w) and (x,v) pairs
      T d1 = T{2} * (b - a), d2 = d1;
      if (dw != dx) d1 = (w - x) * dx / (dx - dw);
      if (dv != dx) d2 = (v - x) * dx / (dx - dv);
      const T u1 = x + d1, u2 = x + d2;
      const bool ok1 = (a - u1) * (u1 - b) > T{} && dx * d1 <= T{};
      const bool ok2 = (a - u2) * (u2 - b) > T{} && dx * d2 <= T{};
      const T olde = e;
      e = d;
      if (ok1 || ok2) {
        d = (ok1 && ok2) ? (abs(d1) < abs(d2) ? d1 : d2)
                         : (ok1 ? d1 : d2);
        if (abs(d) <= abs(T{0.5} * olde)) {
          const T u = x + d;
          if (u - a < tol2 || b - u < tol2)
            d = (xm >= x ? tol1 : -tol1);
        } else {
          e = (dx >= T{} ? a - x : b - x);
          d = T{0.5} * e;
        }
      } else {
        e = (dx >= T{} ? a - x : b - x);
        d = T{0.5} * e;
      }
    } else {
      e = (dx >= T{} ? a - x : b - x);
      d = T{0.5} * e;
    }

    const T u  = (abs(d) >= tol1 ? x + d : x + (d >= T{} ? tol1 : -tol1));
    const T fu = f(u);
    const T du = f.df(u);

    if (fu <= fx) {
      (u < x ? b : a) = x;
      v = w; fv = fw; dv = dw;
      w = x; fw = fx; dw = dx;
      x = u; fx = fu; dx = du;
    } else {
      (u < x ? a : b) = u;
      if (fu <= fw || w == x) {
        v = w; fv = fw; dv = dw;
        w = u; fw = fu; dw = du;
      } else if (fu <= fv || v == x || v == w) {
        v = u; fv = fu; dv = du;
      }
    }
  }
  return x;
}

} // namespace diff::min::detail
