#pragma once

#include "expressions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <numeric>
#include <ranges>

namespace diff::min::detail {

template <typename T, std::size_t N>
using Matrix = std::array<std::array<T, N>, N>;

template <typename T, std::size_t N>
constexpr T dot(const std::array<T, N> &a, const std::array<T, N> &b) noexcept {
  return std::inner_product(a.begin(), a.end(), b.begin(), T{});
}

template <typename T, std::size_t N>
constexpr T norm_sq(const std::array<T, N> &a) noexcept {
  return dot(a, a);
}

// Build an (N+1)-vertex simplex: s[0]=p, s[i+1]=p with s[i+1][i]+=delta
template <typename T, std::size_t N>
constexpr std::array<std::array<T, N>, N + 1>
make_simplex(const std::array<T, N> &p, const T &delta) noexcept {
  std::array<std::array<T, N>, N + 1> s{};
  s[0] = p;
  for (std::size_t i = 0; i < N; ++i) {
    s[i + 1] = p;
    s[i + 1][i] += delta;
  }
  return s;
}

// a[i] = op(a[i], b[i])  — in-place element-wise binary op
template <typename T, std::size_t N, typename Op>
constexpr void zip_inplace(std::array<T, N> &a, const std::array<T, N> &b,
                           Op op) noexcept {
  std::ranges::transform(a, b, a.begin(), op);
}

template <typename T, std::size_t N>
constexpr std::array<T, N> add(const std::array<T, N> &a,
                               const std::array<T, N> &b) noexcept {
  std::array<T, N> r{};
  std::ranges::transform(a, b, r.begin(), std::plus<>{});
  return r;
}

// a ⊗ b — rank-1 outer product
template <typename T, std::size_t N>
constexpr Matrix<T, N> outer(const std::array<T, N> &a,
                             const std::array<T, N> &b) noexcept {
  Matrix<T, N> r{};
  for (auto [i, ri] : std::views::enumerate(r))
    std::ranges::transform(b, ri.begin(),
                           [ai = a[i]](const T &bj) { return ai * bj; });
  return r;
}

template <typename T, std::size_t N>
constexpr Matrix<T, N> mat_add(const Matrix<T, N> &A,
                               const Matrix<T, N> &B) noexcept {
  Matrix<T, N> r{};
  for (auto [ri, ai, bi] : std::views::zip(r, A, B))
    std::ranges::transform(ai, bi, ri.begin(), std::plus<>{});
  return r;
}

// M · v
template <typename T, std::size_t N>
constexpr std::array<T, N> mat_vec(const Matrix<T, N> &M,
                                   const std::array<T, N> &v) noexcept {
  std::array<T, N> r{};
  for (auto [i, mi] : std::views::enumerate(M))
    r[i] = dot(mi, v);
  return r;
}

// M += s * (a ⊗ b)  — in-place rank-1 update, no temporaries
template <typename T, std::size_t N>
constexpr void rank1_update(Matrix<T, N> &M, const T &s,
                            const std::array<T, N> &a,
                            const std::array<T, N> &b) noexcept {
  for (auto [i, mi] : std::views::enumerate(M))
    for (auto j : std::views::iota(0uz, N))
      mi[j] += s * a[i] * b[j];
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

} // namespace diff::min::detail
