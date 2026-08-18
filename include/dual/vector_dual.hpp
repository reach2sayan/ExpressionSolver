#pragma once

#include "expr/expressions.hpp" // Numeric
#include "expr/unary_math.hpp"  // detail::<Op>Fn descriptors, DIFF_UNARY_MATH_TABLE
#include "util/fmt.hpp"         // dual_formatter_base

#include <array>
#include <cmath>
#include <cstddef>
#include <format>
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
  // peel + remainder.  A 32-byte start makes each loop a clean run of whole
  // 256-bit stores.  std::vector uses aligned new for the over-aligned element.
  //
  // Confirmed with -fopt-info-vec on GCC 15 at -O3 -march=native: at N = 32
  // every lane loop in this file vectorizes to 32-byte vectors.  It buys
  // nothing at the bottom of the ladder, though -- vforward_pick starts at
  // N = 1 and doubles, and at N = 1 or 2 the pack is smaller than one vector,
  // so no loop vectorizes and the alignas is pure padding there.  That is
  // accepted: those buckets are for tiny problems where it does not matter.
  //
  // Hand-written intrinsics were measured against this and rejected -- there is
  // nothing for them to win here, and no AVX-512 on the targets to reach for.
  alignas(32) std::array<double, N> grad{}; // partials, lanes [0, N)

  constexpr VectorDual() noexcept = default;
  constexpr VectorDual(double v, const std::array<double, N> &g) noexcept
      : value(v), grad(g) {}

  // Lift a bare scalar to a zero-derivative constant (grad value-initialised).
  // Non-explicit to mirror Dual's implicit double->dual lift, which the
  // promotion helpers in dual.hpp rely on via functional/static casts.
  constexpr VectorDual(double v) noexcept : value(v) {}
  template <CArithmetic U>
    requires(!std::same_as<U, double>)
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

  // Comparisons run on the materialized value, exactly as Dual's do: the
  // derivative pack carries no ordering.  Needed by every branching kernel --
  // abs_combine's sign test, and detail::max_impl/min_impl's `a < b` -- which
  // otherwise cannot compile for a Dual<VectorDual<N>> sweep.  Hidden friends
  // rather than free templates so N comes from the class and the implicit
  // double lift above applies to either operand (`v > 0.0` works).
  [[nodiscard]] friend constexpr auto operator<=>(const VectorDual &a,
                                                  const VectorDual &b) noexcept {
    return a.value <=> b.value;
  }
  [[nodiscard]] friend constexpr bool operator==(const VectorDual &a,
                                                 const VectorDual &b) noexcept {
    return a.value == b.value;
  }

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

  // atan2(y, x): the first operand is the numerator y.
  //   d atan2 = (x*dy - y*dx) / (x^2 + y^2)
  // Same formula as Dual's (dual.hpp), applied per lane with the scalar
  // denominator hoisted out of the loop.  Without these two, an expression
  // using atan2 or hypot compiles at two variables and fails at three, where
  // derivative_tensor<1> switches to the VectorDual fast path.
  [[nodiscard]] friend constexpr VectorDual atan2(const VectorDual &y,
                                                  const VectorDual &x) noexcept {
    using std::atan2;
    const double inv = double{1} / (x.value * x.value + y.value * y.value);
    VectorDual r;
    r.value = atan2(y.value, x.value);
    for (std::size_t k = 0; k < N; ++k) {
      r.grad[k] = (x.value * y.grad[k] - y.value * x.grad[k]) * inv;
    }
    return r;
  }

  // hypot(x, y) = sqrt(x^2 + y^2).  d hypot = (x*dx + y*dy) / hypot.
  [[nodiscard]] friend constexpr VectorDual hypot(const VectorDual &x,
                                                  const VectorDual &y) noexcept {
    using std::hypot;
    const double h = hypot(x.value, y.value);
    const double inv = double{1} / h;
    VectorDual r;
    r.value = h;
    for (std::size_t k = 0; k < N; ++k) {
      r.grad[k] = (x.value * x.grad[k] + y.value * y.grad[k]) * inv;
    }
    return r;
  }
};

// Chain rule for a unary math function, one lane at a time.
//
// The descriptor is instantiated at *double*, not at VectorDual: g(value) and
// g'(value) are scalars, so the whole derivative pack is one scalar scaling a
// contiguous run of lanes.  Instantiating it at VectorDual instead would
// propagate a full pack through the derivative formula as well and cost O(N)
// extra arithmetic per node for an answer that is scalar by construction.
//
// This is the same chain rule unary_dual_combine applies in dual.hpp, including
// the deriv_from_value reuse that computes the primal once when the descriptor
// can express g' in terms of g(u) (exp, tan, tanh, sqrt, cbrt).
template <template <typename> class Fn, std::size_t N>
[[nodiscard]] constexpr VectorDual<N>
vector_unary(const VectorDual<N> &d) noexcept {
  VectorDual<N> r;
  double df;
  if constexpr (detail::has_deriv_from_value_v<Fn<double>, double>) {
    r.value = Fn<double>{}(d.value);
    df = Fn<double>::deriv_from_value(d.value, r.value);
  } else {
    r.value = Fn<double>{}(d.value);
    df = Fn<double>::deriv(d.value);
  }
  for (std::size_t k = 0; k < N; ++k) {
    r.grad[k] = df * d.grad[k];
  }
  return r;
}

// Found by ADL on VectorDual, which is what lets the descriptors in
// unary_math.hpp -- and so every op node and every Dual<VectorDual<N>> sweep --
// call these unqualified.  Generated from the registry, so VectorDual supports
// the same function set as Dual rather than a subset of it.
#define DIFF_VECTOR_UNARY(FN, OP, LABEL)                                       \
  template <std::size_t N>                                                     \
  [[nodiscard]] constexpr VectorDual<N> FN(const VectorDual<N> &d) noexcept {  \
    return vector_unary<detail::OP##Fn>(d);                                    \
  }
DIFF_UNARY_MATH_TABLE(DIFF_VECTOR_UNARY)
#undef DIFF_VECTOR_UNARY

// abs is not in the registry: its derivative is a sign rather than a function
// of the primal, and it is only piecewise differentiable -- the derivative at 0
// is taken as 0, matching abs_combine in dual.hpp.
template <std::size_t N>
[[nodiscard]] constexpr VectorDual<N> abs(const VectorDual<N> &d) noexcept {
  using std::abs;
  VectorDual<N> r;
  r.value = abs(d.value);
  const double sign = d.value > 0.0 ? 1.0 : d.value < 0.0 ? -1.0 : 0.0;
  for (std::size_t k = 0; k < N; ++k) {
    r.grad[k] = sign * d.grad[k];
  }
  return r;
}

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

namespace detail {
template <std::size_t N>
inline constexpr bool is_dual_family_v<VectorDual<N>> = true;
} // namespace detail

} // namespace diff

// `v+[g0, g1, ...]e` -- Dual's `v+de` with the derivative slot widened to the
// whole pack, since that is exactly what VectorDual is.  All N lanes print,
// including the unused tail: the capacity is bucketed to a power of two by
// vforward_pick, so a short pack that printed only its live lanes would hide
// which bucket a sweep actually ran in.
template <std::size_t N>
struct std::formatter<diff::VectorDual<N>, char>
    : diff::detail::dual_formatter_base<double> {
  auto format(const diff::VectorDual<N> &d, std::format_context &ctx) const {
    this->num(ctx, d.value);
    this->put(ctx, "+");
    this->pack(ctx, d.grad);
    this->put(ctx, "e");
    return ctx.out();
  }
};
