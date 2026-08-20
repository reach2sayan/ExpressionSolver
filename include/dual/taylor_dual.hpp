#pragma once

#include "dual/dual.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numbers>
#include <ranges>
#include <utility>

namespace ddx::impl {

// TaylorDual<S, N>: flat N+1 normalized coefficients c[k] = f^(k)(x)/k! for
// univariate higher-order AD.  Multiply is truncated convolution, O(N^2)
// against the O(2^N) of nested Dual.  Reach it via univariate_derivative<N>().

// The coefficient product is a convolution, sum_i a_i b_(k-i), so it commutes
// exactly when the scalar underneath does.
template <Numeric S, std::size_t N> struct TaylorDual;
template <Numeric S, std::size_t N>
inline constexpr bool is_commutative_multiply_v<TaylorDual<S, N>> =
    is_commutative_multiply_v<S>;

template <Numeric S, std::size_t N> struct TaylorDual {
  std::array<S, N + 1> c{};

  // 1/k for the recurrences below: folded once rather than issued as a
  // ~13-cycle vdivsd on the dependency chain.
  static constexpr std::array<S, N + 1> inv_k = []() consteval {
    std::array<S, N + 1> r{};
    for (std::size_t k = 1; k <= N; ++k) {
      r[k] = S{1} / static_cast<S>(k);
    }
    return r;
  }();

  constexpr TaylorDual() noexcept = default;
  constexpr explicit TaylorDual(S val) noexcept : c{} { c[0] = val; }

  constexpr TaylorDual operator+(const TaylorDual &o) const noexcept {
    TaylorDual r;
    std::ranges::transform(c, o.c, r.c.begin(), std::plus{});
    return r;
  }
  constexpr TaylorDual operator-(const TaylorDual &o) const noexcept {
    TaylorDual r;
    std::ranges::transform(c, o.c, r.c.begin(), std::minus{});
    return r;
  }
  constexpr TaylorDual operator-() const noexcept {
    TaylorDual r;
    std::ranges::transform(c, r.c.begin(), std::negate{});
    return r;
  }

  // Truncated polynomial multiplication: (uv)[k] = Σ_{j=0}^{k} u[j]*v[k-j]
  constexpr TaylorDual operator*(const TaylorDual &o) const noexcept {
    TaylorDual r;
    for (std::size_t k = 0; k <= N; ++k) {
      S s{};
      for (std::size_t j = 0; j <= k; ++j) {
        s += c[j] * o.c[k - j];
      }
      r.c[k] = s;
    }
    return r;
  }

  // Polynomial division via r*o = this:
  //   r[k] = (c[k] - Σ_{j=0}^{k-1} r[j]*o[k-j]) / o[0]
  constexpr TaylorDual operator/(const TaylorDual &o) const noexcept {
    TaylorDual r;
    const S inv0 = S{1} / o.c[0];
    for (std::size_t k = 0; k <= N; ++k) {
      S sum = c[k];
      for (std::size_t j = 0; j < k; ++j) {
        sum -= r.c[j] * o.c[k - j];
      }
      r.c[k] = sum * inv0;
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

  // On the value coefficient, as Dual's comparisons use val().  The branching
  // kernels in operations.hpp need it.
  [[nodiscard]] friend constexpr auto
  operator<=>(const TaylorDual &a, const TaylorDual &b) noexcept {
    return a.c[0] <=> b.c[0];
  }
  [[nodiscard]] friend constexpr bool
  operator==(const TaylorDual &a, const TaylorDual &b) noexcept {
    return a.c[0] == b.c[0];
  }

  // Solve g·w' = u' for w, given w[0].  Differentiating and matching
  // coefficients gives, for k >= 1:
  //   g[0]·k·w[k] = k·u[k] − Σ_{j=1}^{k-1} (k-j)·g[j]·w[k-j]
  //
  // Composite-safe: stated in terms of the whole series g, so it holds when u
  // is a composite rather than a bare seed.  Every inverse function below is
  // this recurrence with a different g.
  [[nodiscard]] static constexpr TaylorDual
  implicit_recurrence(const TaylorDual &u, S w0, const TaylorDual &g) noexcept {
    TaylorDual w;
    w.c[0] = w0;
    const S invg0 = S{1} / g.c[0];
    for (std::size_t k = 1; k <= N; ++k) {
      auto rhs = static_cast<S>(k) * u.c[k];
      for (std::size_t j = 1; j < k; ++j) {
        rhs -= g.c[j] * static_cast<S>(k - j) * w.c[k - j];
      }
      w.c[k] = rhs * inv_k[k] * invg0;
    }
    return w;
  }

  // exp: w[0]=exp(u[0]),  k*w[k] = Σ_{j=1}^{k} j*u[j]*w[k-j]
  [[nodiscard]] friend constexpr TaylorDual exp(const TaylorDual &u) noexcept {
    using std::exp;
    TaylorDual w;
    w.c[0] = exp(u.c[0]);
    for (std::size_t k = 1; k <= N; ++k) {
      S s{};
      for (std::size_t j = 1; j <= k; ++j) {
        s += static_cast<S>(j) * u.c[j] * w.c[k - j];
      }
      w.c[k] = s * inv_k[k];
    }
    return w;
  }

  // log: w[0]=log(u[0]),  w[k] = (u[k] - (1/k)Σ_{j=1}^{k-1} j*w[j]*u[k-j]) /
  // u[0]
  [[nodiscard]] friend constexpr TaylorDual log(const TaylorDual &u) noexcept {
    using std::log;
    TaylorDual w;
    w.c[0] = log(u.c[0]);
    const S invu0 = S{1} / u.c[0];
    for (std::size_t k = 1; k <= N; ++k) {
      S s{};
      for (std::size_t j = 1; j < k; ++j) {
        s += static_cast<S>(j) * w.c[j] * u.c[k - j];
      }
      w.c[k] = (u.c[k] - s * inv_k[k]) * invu0;
    }
    return w;
  }

  // The coupled sine/cosine recurrence; circular and hyperbolic differ in one
  // sign:
  //   k*s[k] = Σ j*u[j]*c[k-j],   k*c[k] = ±Σ j*u[j]*s[k-j]
  // Circular = -1, hyperbolic = +1.  Either series needs both, so they are
  // computed together.
  template <int Sign>
  [[nodiscard]] static constexpr std::pair<TaylorDual, TaylorDual>
  coupled_pair(const TaylorDual &u, S s0, S c0) noexcept {
    static_assert(Sign == 1 || Sign == -1);
    TaylorDual s, cc;
    s.c[0] = s0;
    cc.c[0] = c0;
    for (std::size_t k = 1; k <= N; ++k) {
      S sv{}, cv{};
      for (std::size_t j = 1; j <= k; ++j) {
        const S ju = static_cast<S>(j) * u.c[j];
        sv += ju * cc.c[k - j];
        if constexpr (Sign < 0) {
          cv -= ju * s.c[k - j];
        } else {
          cv += ju * s.c[k - j];
        }
      }
      s.c[k] = sv * inv_k[k];
      cc.c[k] = cv * inv_k[k];
    }
    return {s, cc};
  }

  [[nodiscard]] friend constexpr std::pair<TaylorDual, TaylorDual>
  sincos_td(const TaylorDual &u) noexcept {
    using std::sin, std::cos;
    return coupled_pair<-1>(u, sin(u.c[0]), cos(u.c[0]));
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

  [[nodiscard]] constexpr friend std::pair<TaylorDual, TaylorDual>
  sinhcosh_td(const TaylorDual &u) noexcept {
    using std::sinh, std::cosh;
    return coupled_pair<1>(u, sinh(u.c[0]), cosh(u.c[0]));
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

  // asin: sqrt(1-u²)·w' = u'  (general, composite-safe):
  //   s[0]·k·w[k] = k·u[k] − Σ_{j=1}^{k-1} (k-j)·s[k-j]·w[k-j]
  [[nodiscard]] constexpr friend TaylorDual asin(const TaylorDual &u) noexcept {
    using std::asin;
    TaylorDual one;
    one.c[0] = S{1};
    return implicit_recurrence(u, asin(u.c[0]), sqrt(one - u * u));
  }

  // acos = pi/2 - asin  →  same derivative coefficients, negated for k>=1
  [[nodiscard]] friend constexpr TaylorDual acos(const TaylorDual &u) noexcept {
    using std::acos;
    TaylorDual w = asin(u);
    w.c[0] = acos(u.c[0]);
    // Negate the derivative coefficients (k >= 1); value coefficient stays.
    std::ranges::transform(w.c | std::views::drop(1), w.c.begin() + 1,
                           std::negate{});
    return w;
  }

  // atan: (1+u²)·w' = u'  (general, composite-safe):
  //   p[0]·k·w[k] = k·u[k] − Σ_{j=1}^{k-1} (k-j)·p[k-j]·w[k-j]
  [[nodiscard]] friend constexpr TaylorDual atan(const TaylorDual &u) noexcept {
    using std::atan;
    TaylorDual p = u * u;
    p.c[0] += S{1};
    return implicit_recurrence(u, atan(u.c[0]), p);
  }

  // Real-power series for w = u^p via the ODE  u·w' = p·u'·w, given w[0].
  //   n·u[0]·w[n] = p·Σ_{j=1}^{n} j·u[j]·w[n-j] − Σ_{j=1}^{n-1} j·u[n-j]·w[j]
  [[nodiscard]] static constexpr TaylorDual powser(const TaylorDual &u, S p,
                                                   S w0) noexcept {
    TaylorDual w;
    w.c[0] = w0;
    const S invu0 = S{1} / u.c[0];
    for (std::size_t n = 1; n <= N; ++n) {
      S acc{};
      for (std::size_t j = 1; j <= n; ++j) {
        acc += p * static_cast<S>(j) * u.c[j] * w.c[n - j];
      }
      for (std::size_t j = 1; j < n; ++j) {
        acc -= static_cast<S>(j) * w.c[j] * u.c[n - j];
      }
      w.c[n] = acc * inv_k[n] * invu0;
    }
    return w;
  }

  // pow(u, v) = exp(v·log u), requires u[0] > 0.  A constant exponent instead
  // goes through powser, which needs no logarithm and allows u[0] < 0.
  [[nodiscard]] friend constexpr TaylorDual pow(const TaylorDual &u,
                                                const TaylorDual &v) noexcept {
    using std::pow;
    if (std::ranges::all_of(v.c | std::views::drop(1),
                            [](const S &e) { return e == S{}; })) {
      return powser(u, v.c[0], pow(u.c[0], v.c[0]));
    }
    return exp(v * log(u));
  }

  // cbrt via the same recurrence; w[0] = std::cbrt handles a negative argument.
  [[nodiscard]] friend constexpr TaylorDual cbrt(const TaylorDual &u) noexcept {
    using std::cbrt;
    return powser(u, S{1} / S{3}, cbrt(u.c[0]));
  }

  // log10(u) = log(u) / ln(10); every coefficient scales by log10(e).
  [[nodiscard]] friend constexpr TaylorDual log10(const TaylorDual &u) noexcept {
    TaylorDual w = log(u);
    const S log10e = static_cast<S>(std::numbers::log10e);
    std::ranges::transform(w.c, w.c.begin(),
                           [log10e](S x) noexcept { return x * log10e; });
    return w;
  }

  // asinh: sqrt(1+u²)·w' = u'  (mirrors asin with s = sqrt(1 + u²)).
  [[nodiscard]] friend constexpr TaylorDual asinh(const TaylorDual &u) noexcept {
    using std::asinh;
    TaylorDual one;
    one.c[0] = S{1};
    return implicit_recurrence(u, asinh(u.c[0]), sqrt(one + u * u));
  }

  // acosh: sqrt(u²-1)·w' = u'  (requires u[0] > 1).
  [[nodiscard]] friend constexpr TaylorDual acosh(const TaylorDual &u) noexcept {
    using std::acosh;
    TaylorDual one;
    one.c[0] = S{1};
    return implicit_recurrence(u, acosh(u.c[0]), sqrt(u * u - one));
  }

  // atanh: (1-u²)·w' = u'  (mirrors atan with p = 1 - u²).
  [[nodiscard]] friend constexpr TaylorDual atanh(const TaylorDual &u) noexcept {
    using std::atanh;
    TaylorDual p = -(u * u);
    p.c[0] += S{1};
    return implicit_recurrence(u, atanh(u.c[0]), p);
  }

  // erf: erf' = (2/√π)·exp(-u²); integrate w' = (2/√π)·exp(-u²)·u'.
  [[nodiscard]] friend constexpr TaylorDual erf(const TaylorDual &u) noexcept {
    using std::erf;
    TaylorDual w;
    w.c[0] = erf(u.c[0]);
    constexpr S cc = static_cast<S>(S{2} * std::numbers::inv_sqrtpi);
    const TaylorDual g = exp(-(u * u)); // exp(-u²)
    for (std::size_t k = 1; k <= N; ++k) {
      // coefficient (k-1) of g·u' : Σ_{i=0}^{k-1} g[i]·(k-i)·u[k-i]
      S deriv{};
      for (std::size_t i = 0; i < k; ++i) {
        deriv += g.c[i] * static_cast<S>(k - i) * u.c[k - i];
      }
      w.c[k] = cc * deriv * inv_k[k];
    }
    return w;
  }

  // atan2(y, x) and atan(y/x) differ by a piecewise constant, so only the value
  // coefficient differs; atan is composite-safe, so this is exact.
  [[nodiscard]] friend constexpr TaylorDual atan2(const TaylorDual &y,
                                        const TaylorDual &x) noexcept {
    using std::atan2;
    TaylorDual w = atan(y / x);
    w.c[0] = atan2(y.c[0], x.c[0]);
    return w;
  }

  [[nodiscard]] friend constexpr TaylorDual hypot(const TaylorDual &x,
                                        const TaylorDual &y) noexcept {
    return sqrt(x * x + y * y);
  }

};

static_assert(Numeric<TaylorDual<double, 3>>,
              "TaylorDual must satisfy Numeric to nest in an expression");

template <Numeric T, std::size_t N>
auto scalar_base_impl(std::type_identity<TaylorDual<T, N>>) -> T;

// dual_depth_v<TaylorDual<S,N>> = N  (mirrors nth_dual_t<S,N> depth)
template <Numeric S, std::size_t N>
inline constexpr std::size_t dual_depth_v<TaylorDual<S, N>> = N;

// Lets Constant::eval_seeded produce a zero-derivative TaylorDual.

template <Numeric S, std::size_t N> struct ConstantEmbedder<TaylorDual<S, N>> {
  static constexpr TaylorDual<S, N> embed(S val) noexcept {
    TaylorDual<S, N> t;
    t.c[0] = val;
    return t;
  }
};

namespace detail {
template <Numeric S, std::size_t N>
inline constexpr bool is_dual_family_v<TaylorDual<S, N>> = true;
} // namespace detail

} // namespace ddx::impl

// The shared series renderer over the coefficient array.  These are the
// normalized coefficients f^(k)/k!; univariate_derivative puts the factorial
// back, not this.
template <ddx::impl::Numeric S, std::size_t N>
struct std::formatter<ddx::impl::TaylorDual<S, N>, char>
    : ddx::impl::detail::dual_formatter_base<S> {
  auto format(const ddx::impl::TaylorDual<S, N> &t, std::format_context &ctx) const {
    this->series(ctx, t.c);
    return ctx.out();
  }
};
