#pragma once

#include "dual.hpp"
#include <array>
#include <cmath>
#include <utility>

namespace diff {

// TaylorDual<S, N> — flat N+1-coefficient type for univariate higher-order AD.
//
// Stores normalized Taylor coefficients: c[k] = f^(k)(x) / k!
// Multiply is truncated polynomial convolution — O(N²) vs the O(2^N) of
// nested Dual<Dual<...<S>...>>.  Extract the N-th derivative as c[N] * N!.
//
// Use univariate_derivative<N>(expr [, x0]) for a clean public API.

template <Numeric S, std::size_t N>
struct TaylorDual {
  std::array<S, N + 1> c{};
  constexpr TaylorDual() noexcept = default;
  constexpr explicit TaylorDual(S val) noexcept : c{} { c[0] = val; }

  constexpr TaylorDual operator+(const TaylorDual &o) const noexcept {
    TaylorDual r;
    for (std::size_t k = 0; k <= N; ++k) {
      r.c[k] = c[k] + o.c[k];
    }
    return r;
  }
  constexpr TaylorDual operator-(const TaylorDual &o) const noexcept {
    TaylorDual r;
    for (std::size_t k = 0; k <= N; ++k) {
      r.c[k] = c[k] - o.c[k];
    }
    return r;
  }
  constexpr TaylorDual operator-() const noexcept {
    TaylorDual r;
    for (std::size_t k = 0; k <= N; ++k) {
      r.c[k] = -c[k];
    }
    return r;
  }

  // Truncated polynomial multiplication: (uv)[k] = Σ_{j=0}^{k} u[j]*v[k-j]
  constexpr TaylorDual operator*(const TaylorDual &o) const noexcept {
    TaylorDual r;
    for (std::size_t i = 0; i <= N; ++i) {
      for (std::size_t j = 0; j + i <= N; ++j) {
        r.c[i + j] += c[i] * o.c[j];
      }
    }
    return r;
  }

  // Polynomial division via r*o = this:
  //   r[k] = (c[k] - Σ_{j=0}^{k-1} r[j]*o[k-j]) / o[0]
  constexpr TaylorDual operator/(const TaylorDual &o) const noexcept {
    TaylorDual r;
    for (std::size_t k = 0; k <= N; ++k) {
      S sum = c[k];
      for (std::size_t j = 0; j < k; ++j) {
        sum -= r.c[j] * o.c[k - j];
      }
      r.c[k] = sum / o.c[0];
    }
    return r;
  }

  constexpr TaylorDual &operator+=(const TaylorDual &o) noexcept {
    return *this = *this + o;
  }
  constexpr TaylorDual &operator-=(const TaylorDual &o) noexcept {
    return *this = *this - o;
  }
  constexpr TaylorDual &operator*=(const TaylorDual &o) noexcept {
    return *this = *this * o;
  }
  constexpr TaylorDual &operator/=(const TaylorDual &o) noexcept {
    return *this = *this / o;
  }

  constexpr TaylorDual &operator++() noexcept {
    ++c[0];
    return *this;
  }

  template <std::size_t I> [[nodiscard]] constexpr S get() const noexcept {
    static_assert(I <= N, "TaylorDual::get<I>: index out of range");
    return c[I];
  }

  // -------------------------------------------------------------------------
  // Transcendentals — all use recurrences on the normalized c[k] = f^(k)/k!
  // -------------------------------------------------------------------------

  // exp: w[0]=exp(u[0]),  k*w[k] = Σ_{j=1}^{k} j*u[j]*w[k-j]
  [[nodiscard]] friend TaylorDual exp(const TaylorDual &u) noexcept {
    using std::exp;
    TaylorDual w;
    w.c[0] = exp(u.c[0]);
    for (std::size_t k = 1; k <= N; ++k) {
      S s{};
      for (std::size_t j = 1; j <= k; ++j) {
        s += static_cast<S>(j) * u.c[j] * w.c[k - j];
      }
      w.c[k] = s / static_cast<S>(k);
    }
    return w;
  }

  // log: w[0]=log(u[0]),  w[k] = (u[k] - (1/k)Σ_{j=1}^{k-1} j*w[j]*u[k-j]) /
  // u[0]
  [[nodiscard]] friend TaylorDual log(const TaylorDual &u) noexcept {
    using std::log;
    TaylorDual w;
    w.c[0] = log(u.c[0]);
    for (std::size_t k = 1; k <= N; ++k) {
      auto s = u.c[k];
      for (std::size_t j = 1; j < k; ++j) {
        s -= static_cast<S>(j) * w.c[j] * u.c[k - j] / static_cast<S>(k);
      }
      w.c[k] = s / u.c[0];
    }
    return w;
  }

  // sin & cos coupled: k*s[k] = Σ j*u[j]*c[k-j],  k*c[k] = -Σ j*u[j]*s[k-j]
  [[nodiscard]] friend std::pair<TaylorDual, TaylorDual>
  sincos_td(const TaylorDual &u) noexcept {
    using std::sin, std::cos;
    TaylorDual s, c;
    s.c[0] = sin(u.c[0]);
    c.c[0] = cos(u.c[0]);
    for (std::size_t k = 1; k <= N; ++k) {
      S sv{}, cv{};
      for (std::size_t j = 1; j <= k; ++j) {
        const S ju = static_cast<S>(j) * u.c[j];
        sv += ju * c.c[k - j];
        cv -= ju * s.c[k - j];
      }
      s.c[k] = sv / static_cast<S>(k);
      c.c[k] = cv / static_cast<S>(k);
    }
    return {s, c};
  }

  [[nodiscard]] constexpr friend TaylorDual sin(const TaylorDual &u) noexcept {
    return sincos_td(u).first;
  }
  [[nodiscard]] constexpr friend TaylorDual cos(const TaylorDual &u) noexcept {
    return sincos_td(u).second;
  }
  [[nodiscard]] constexpr friend TaylorDual tan(const TaylorDual &u) noexcept {
    auto [s, c] = sincos_td(u);
    return s / c;
  }

  // sqrt: w[0]=sqrt(u[0]),  w[k] = (u[k] - Σ_{j=1}^{k-1} w[j]*w[k-j]) /
  // (2*w[0])
  [[nodiscard]] constexpr friend TaylorDual sqrt(const TaylorDual &u) noexcept {
    using std::sqrt;
    TaylorDual w;
    w.c[0] = sqrt(u.c[0]);
    const auto inv2w0 = S{1} / (S{2} * w.c[0]);
    for (std::size_t k = 1; k <= N; ++k) {
      auto s = u.c[k];
      for (std::size_t j = 1; j < k; ++j) {
        s -= w.c[j] * w.c[k - j];
      }
      w.c[k] = s * inv2w0;
    }
    return w;
  }

  [[nodiscard]] constexpr friend TaylorDual abs(const TaylorDual &u) noexcept {
    return u.c[0] >= S{} ? u : -u;
  }

  // sinh & cosh coupled: k*sh[k] = Σ j*u[j]*ch[k-j],  k*ch[k] = Σ
  // j*u[j]*sh[k-j]
  [[nodiscard]] constexpr friend std::pair<TaylorDual, TaylorDual>
  sinhcosh_td(const TaylorDual &u) noexcept {
    using std::sinh, std::cosh;
    TaylorDual sh, ch;
    sh.c[0] = sinh(u.c[0]);
    ch.c[0] = cosh(u.c[0]);
    for (std::size_t k = 1; k <= N; ++k) {
      S sv{}, cv{};
      for (std::size_t j = 1; j <= k; ++j) {
        const auto ju = static_cast<S>(j) * u.c[j];
        sv += ju * ch.c[k - j];
        cv += ju * sh.c[k - j];
      }
      sh.c[k] = sv / static_cast<S>(k);
      ch.c[k] = cv / static_cast<S>(k);
    }
    return {sh, ch};
  }

  [[nodiscard]] constexpr friend TaylorDual sinh(const TaylorDual &u) noexcept {
    return sinhcosh_td(u).first;
  }
  [[nodiscard]] constexpr friend TaylorDual cosh(const TaylorDual &u) noexcept {
    return sinhcosh_td(u).second;
  }
  [[nodiscard]] constexpr friend TaylorDual tanh(const TaylorDual &u) noexcept {
    const auto [sh, ch] = sinhcosh_td(u);
    return sh / ch;
  }

  // asin: sqrt(1-u²)*w' = 1  →  s[0]*k*w[k] = [k==1] - Σ_{j=1}^{k-1}
  // s[j]*(k-j)*w[k-j]
  [[nodiscard]] constexpr friend TaylorDual asin(const TaylorDual &u) noexcept {
    using std::asin;
    TaylorDual w;
    w.c[0] = asin(u.c[0]);
    TaylorDual one;
    one.c[0] = S{1};
    const TaylorDual s = sqrt(one - u * u);
    for (std::size_t k = 1; k <= N; ++k) {
      auto rhs = (k == 1) ? S{1} : S{};
      for (std::size_t j = 1; j < k; ++j) {
        rhs -= s.c[j] * static_cast<S>(k - j) * w.c[k - j];
      }
      w.c[k] = rhs / (static_cast<S>(k) * s.c[0]);
    }
    return w;
  }

  // acos = pi/2 - asin  →  same derivative coefficients, negated for k>=1
  [[nodiscard]] friend TaylorDual acos(const TaylorDual &u) noexcept {
    using std::acos;
    TaylorDual w = asin(u);
    w.c[0] = acos(u.c[0]);
    for (std::size_t k = 1; k <= N; ++k) {
      w.c[k] = -w.c[k];
    }
    return w;
  }

  // atan: (1+u²)*w' = 1  →  p[0]*k*w[k] = [k==1] - Σ_{j=1}^{k-1}
  // p[j]*(k-j)*w[k-j]
  [[nodiscard]] friend TaylorDual atan(const TaylorDual &u) noexcept {
    using std::atan;
    TaylorDual w;
    w.c[0] = atan(u.c[0]);
    TaylorDual p = u * u;
    p.c[0] += S{1};
    for (std::size_t k = 1; k <= N; ++k) {
      auto rhs = (k == 1) ? S{1} : S{};
      for (std::size_t j = 1; j < k; ++j) {
        rhs -= p.c[j] * static_cast<S>(k - j) * w.c[k - j];
      }
      w.c[k] = rhs / (static_cast<S>(k) * p.c[0]);
    }
    return w;
  }
};

// TaylorDual satisfies Numeric — verified by static_assert in gradient.hpp.

template <typename T> inline constexpr bool is_taylor_dual_v = false;
template <typename S, std::size_t N>
inline constexpr bool is_taylor_dual_v<TaylorDual<S, N>> = true;

template <typename T, std::size_t N>
auto scalar_base_impl(std::type_identity<TaylorDual<T, N>>) -> T;

// dual_depth_v<TaylorDual<S,N>> = N  (mirrors nth_dual_t<S,N> depth)
template <typename S, std::size_t N>
inline constexpr std::size_t dual_depth_v<TaylorDual<S, N>> = N;

// ---------------------------------------------------------------------------
// ConstantEmbedder specialization — lets Constant::eval_seeded_as produce
// a zero-derivative TaylorDual for constant nodes.
// ---------------------------------------------------------------------------

template <typename S, std::size_t N> struct ConstantEmbedder<TaylorDual<S, N>> {
  static constexpr TaylorDual<S, N> embed(S val) noexcept {
    TaylorDual<S, N> t;
    t.c[0] = val;
    return t;
  }
};

} // namespace diff
