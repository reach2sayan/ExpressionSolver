#pragma once

#include "dual.hpp"        // Dual (for the dual_vforward alias)
#include "expressions.hpp" // Numeric

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace diff {

// VectorDual<N> — a forward-vector dual number: one value plus a fixed-capacity
// pack of N first-order partial derivatives, propagated *elementwise* through
// every operation.  It is a drop-in "scalar" for the Dual<> machinery, so
// Dual<VectorDual<N>> is a second-order, vector-forward-over-forward number: a
// single sweep with the inner pack seeded to the identity yields a whole row of
// the Hessian at once (see vforward_driver.hpp), replacing the O(n^2) scalar
// sweep in forward_driver.hpp with O(n) sweeps.
//
// N is a compile-time capacity.  The driver picks the smallest bucket with
// N >= m (the active-variable count) and only the first m lanes carry meaning;
// the trailing lanes stay zero and ride along for free.  The lane loops are
// plain `for (k < N)` over a contiguous std::array so the compiler
// auto-vectorizes them to SIMD under -O2/-O3 (no Eigen dependency is pulled
// into this otherwise dependency-free, constexpr header library).
template <std::size_t N> struct VectorDual {
  double value{};
  // Over-align the partial pack so the elementwise lane loops below vectorize
  // without a scalar peel prologue: under -march=x86-64-v3 (AVX2, 32-byte
  // vectors) an 8-byte-aligned array forces the compiler to emit an alignment
  // peel + remainder.  The Hessian driver buckets N to powers of two >= 4
  // (vforward_pick), so a 32-byte start makes each loop a clean run of whole
  // 256-bit stores.  std::vector uses aligned new for the over-aligned element.
  alignas(32) std::array<double, N> grad{}; // partials, lanes [0, N)

  constexpr VectorDual() noexcept = default;
  constexpr VectorDual(double v, const std::array<double, N> &g) noexcept
      : value(v), grad(g) {}

  // Lift a bare scalar to a zero-derivative constant (grad value-initialised).
  // Non-explicit to mirror Dual's implicit double->dual lift, which the
  // promotion helpers in dual.hpp rely on via functional/static casts.
  constexpr VectorDual(double v) noexcept : value(v) {}
  template <typename U>
    requires(std::is_arithmetic_v<U> && !std::is_same_v<U, double>)
  constexpr VectorDual(U s) noexcept : value(static_cast<double>(s)) {}

  constexpr VectorDual operator-() const noexcept {
    VectorDual r;
    r.value = -value;
    for (std::size_t k = 0; k < N; ++k) {
      r.grad[k] = -grad[k];
    }
    return r;
  }
  constexpr VectorDual operator+(const VectorDual &o) const noexcept {
    VectorDual r;
    r.value = value + o.value;
    for (std::size_t k = 0; k < N; ++k) {
      r.grad[k] = grad[k] + o.grad[k];
    }
    return r;
  }
  constexpr VectorDual operator-(const VectorDual &o) const noexcept {
    VectorDual r;
    r.value = value - o.value;
    for (std::size_t k = 0; k < N; ++k) {
      r.grad[k] = grad[k] - o.grad[k];
    }
    return r;
  }
  constexpr VectorDual operator*(const VectorDual &o) const noexcept {
    VectorDual r;
    r.value = value * o.value;
    for (std::size_t k = 0; k < N; ++k) {
      r.grad[k] = grad[k] * o.value + value * o.grad[k];
    }
    return r;
  }
  constexpr VectorDual operator/(const VectorDual &o) const noexcept {
    VectorDual r;
    const double inv = double{1} / o.value;
    r.value = value * inv;
    for (std::size_t k = 0; k < N; ++k) {
      r.grad[k] = (grad[k] - r.value * o.grad[k]) * inv;
    }
    return r;
  }

  constexpr VectorDual &operator+=(const VectorDual &o) noexcept {
    return *this = *this + o;
  }
  constexpr VectorDual &operator-=(const VectorDual &o) noexcept {
    return *this = *this - o;
  }
  constexpr VectorDual &operator*=(const VectorDual &o) noexcept {
    return *this = *this * o;
  }
  constexpr VectorDual &operator/=(const VectorDual &o) noexcept {
    return *this = *this / o;
  }

  // ADL math functions used by the templated energy stack (log/exp) and by the
  // Dual<> friends that wrap them.  Derivative of g(value) is g'(value)*grad.
  [[nodiscard]] friend constexpr VectorDual log(const VectorDual &d) noexcept {
    using std::log;
    VectorDual r;
    r.value = log(d.value);
    const double inv = double{1} / d.value;
    for (std::size_t k = 0; k < N; ++k) {
      r.grad[k] = d.grad[k] * inv;
    }
    return r;
  }
  [[nodiscard]] friend constexpr VectorDual exp(const VectorDual &d) noexcept {
    using std::exp;
    const double e = exp(d.value);
    VectorDual r;
    r.value = e;
    for (std::size_t k = 0; k < N; ++k) {
      r.grad[k] = e * d.grad[k];
    }
    return r;
  }
  [[nodiscard]] friend constexpr VectorDual sin(const VectorDual &d) noexcept {
    using std::cos, std::sin;
    VectorDual r;
    r.value = sin(d.value);
    const double c = cos(d.value);
    for (std::size_t k = 0; k < N; ++k) {
      r.grad[k] = c * d.grad[k];
    }
    return r;
  }
  [[nodiscard]] friend constexpr VectorDual cos(const VectorDual &d) noexcept {
    using std::cos, std::sin;
    VectorDual r;
    r.value = cos(d.value);
    const double ms = -sin(d.value);
    for (std::size_t k = 0; k < N; ++k) {
      r.grad[k] = ms * d.grad[k];
    }
    return r;
  }
  [[nodiscard]] friend constexpr VectorDual tan(const VectorDual &d) noexcept {
    using std::tan;
    const double t = tan(d.value);
    VectorDual r;
    r.value = t;
    const double sec2 = double{1} + t * t; // sec^2 = 1 + tan^2
    for (std::size_t k = 0; k < N; ++k) {
      r.grad[k] = sec2 * d.grad[k];
    }
    return r;
  }
  [[nodiscard]] friend constexpr VectorDual sqrt(const VectorDual &d) noexcept {
    using std::sqrt;
    const double s = sqrt(d.value);
    VectorDual r;
    r.value = s;
    const double f = double{1} / (double{2} * s);
    for (std::size_t k = 0; k < N; ++k) {
      r.grad[k] = d.grad[k] * f;
    }
    return r;
  }
  // a^b.  d(a^b) = a^b (b' ln a + b a'/a).  Needed by Dual<VectorDual>'s own
  // pow(), which calls pow on the inner scalar (e.g. VolumeContribution's
  // press_shape raises an RK base to a non-integer exponent).
  [[nodiscard]] friend constexpr VectorDual pow(const VectorDual &a,
                                                const VectorDual &b) noexcept {
    using std::log, std::pow;
    const double p = pow(a.value, b.value);
    const double la = log(a.value);
    const double inva = double{1} / a.value;
    VectorDual r;
    r.value = p;
    for (std::size_t k = 0; k < N; ++k) {
      r.grad[k] = p * (b.grad[k] * la + b.value * a.grad[k] * inva);
    }
    return r;
  }
};

// val(): recover the underlying scalar.  Lets the dual.hpp val()/to_double()/
// comparison overloads peel a Dual<VectorDual<N>> down to its base double.
template <std::size_t N> constexpr double val(const VectorDual<N> &d) noexcept {
  return d.value;
}

static_assert(Numeric<VectorDual<1>>);
static_assert(Numeric<VectorDual<8>>);

#ifndef DIFF_VFORWARD_CAPACITY
#define DIFF_VFORWARD_CAPACITY 32
#endif
inline constexpr std::size_t kVForwardN = DIFF_VFORWARD_CAPACITY;
static_assert(kVForwardN > 0, "DIFF_VFORWARD_CAPACITY must be positive");
using dual_vforward = Dual<VectorDual<kVForwardN>>;

} // namespace diff
