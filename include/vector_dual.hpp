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
  std::array<double, N> grad{}; // partials, lanes [0, N)

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
};

// val(): recover the underlying scalar.  Lets the dual.hpp val()/to_double()/
// comparison overloads peel a Dual<VectorDual<N>> down to its base double.
template <std::size_t N>
constexpr double val(const VectorDual<N> &d) noexcept {
  return d.value;
}

static_assert(Numeric<VectorDual<1>>);
static_assert(Numeric<VectorDual<8>>);

// Single fixed VectorDual capacity for the vector-forward Hessian driver.
// Covers the realistic active-variable count of every model (multi-sublattice
// orderings, ionic phases, MQMQA quadruplets all sit comfortably below 32);
// larger problems fall back to the scalar O(m^2) hessian().  A single capacity
// (rather than a 4/8/16/32 bucket ladder) keeps to ONE Dual<VectorDual<N>>
// element type, which matters because every consumer that defines its energy
// template in a .cpp must explicitly instantiate it for this type.
inline constexpr std::size_t kVForwardN = 32;

// The forward-dual element type the vector-forward driver evaluates `f` with.
// Consumers explicitly instantiating their energy template should add an
// instantiation for diff::dual_vforward alongside their diff::dual2nd one.
using dual_vforward = Dual<VectorDual<kVForwardN>>;

} // namespace diff
