#pragma once

#include "expressions.hpp"
#include <cmath>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace diff {

template <typename T> class Dual {
private:
  T val{};
  T deriv{};

public:
  constexpr Dual() noexcept = default;
  constexpr explicit Dual(T v, T d = T{}) noexcept : val(v), deriv(d) {}

  constexpr Dual operator+(const Dual &o) const noexcept {
    return Dual{val + o.val, deriv + o.deriv};
  }
  constexpr Dual operator-(const Dual &o) const noexcept {
    return Dual{val - o.val, deriv - o.deriv};
  }
  constexpr Dual operator*(const Dual &o) const noexcept {
    return Dual{val * o.val, deriv * o.val + val * o.deriv};
  }
  constexpr Dual operator/(const Dual &o) const noexcept {
    return Dual{val / o.val, (deriv * o.val - val * o.deriv) / (o.val * o.val)};
  }

  // Mixed Dual/scalar arithmetic: a bare scalar is promoted to a
  // zero-derivative Dual.  Hidden friends, so they only participate in overload
  // resolution via ADL when a Dual operand is present.
  friend constexpr Dual operator+(const Dual &a, const T &s) noexcept {
    return a + Dual{s};
  }
  friend constexpr Dual operator+(const T &s, const Dual &a) noexcept {
    return Dual{s} + a;
  }
  friend constexpr Dual operator-(const Dual &a, const T &s) noexcept {
    return a - Dual{s};
  }
  friend constexpr Dual operator-(const T &s, const Dual &a) noexcept {
    return Dual{s} - a;
  }
  friend constexpr Dual operator*(const Dual &a, const T &s) noexcept {
    return a * Dual{s};
  }
  friend constexpr Dual operator*(const T &s, const Dual &a) noexcept {
    return Dual{s} * a;
  }
  friend constexpr Dual operator/(const Dual &a, const T &s) noexcept {
    return a / Dual{s};
  }
  friend constexpr Dual operator/(const T &s, const Dual &a) noexcept {
    return Dual{s} / a;
  }

  constexpr Dual &operator+=(const Dual &o) noexcept {
    val += o.val;
    deriv += o.deriv;
    return *this;
  }
  constexpr Dual &operator-=(const Dual &o) noexcept {
    val -= o.val;
    deriv -= o.deriv;
    return *this;
  }
  constexpr Dual &operator*=(const Dual &o) noexcept {
    *this = *this * o;
    return *this;
  }
  constexpr Dual &operator/=(const Dual &o) noexcept {
    *this = *this / o;
    return *this;
  }

  friend constexpr Dual &operator+=(Dual &a, const T &s) noexcept {
    return a += Dual{s};
  }
  friend constexpr Dual &operator-=(Dual &a, const T &s) noexcept {
    return a -= Dual{s};
  }
  friend constexpr Dual &operator*=(Dual &a, const T &s) noexcept {
    return a *= Dual{s};
  }
  friend constexpr Dual &operator/=(Dual &a, const T &s) noexcept {
    return a /= Dual{s};
  }

  constexpr Dual operator-() const noexcept { return Dual{-val, -deriv}; }
  constexpr Dual &operator++() noexcept {
    ++val;
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &out, const Dual &d) {
    return out << d.val << "+" << d.deriv << "e";
  }

  template <std::size_t Index>
  [[nodiscard]] constexpr auto get() const noexcept {
    static_assert(Index < 2, "Dual index out of bounds");
    if constexpr (Index == 0) {
      return val;
    } else {
      return deriv;
    }
  }

  [[nodiscard]] friend constexpr Dual sin(const Dual &d) noexcept {
    using std::sin, std::cos;
    return Dual{sin(d.val), cos(d.val) * d.deriv};
  }
  [[nodiscard]] friend constexpr Dual cos(const Dual &d) noexcept {
    using std::sin, std::cos;
    return Dual{cos(d.val), -sin(d.val) * d.deriv};
  }
  [[nodiscard]] friend constexpr Dual exp(const Dual &d) noexcept {
    using std::exp;
    const T e = exp(d.val);
    return Dual{e, e * d.deriv};
  }
  [[nodiscard]] friend constexpr Dual tan(const Dual &d) noexcept {
    using std::tan, std::cos;
    const T c = cos(d.val);
    return Dual{tan(d.val), d.deriv / (c * c)};
  }
  [[nodiscard]] friend constexpr Dual log(const Dual &d) noexcept {
    using std::log;
    return Dual{log(d.val), d.deriv / d.val};
  }
  [[nodiscard]] friend constexpr Dual sqrt(const Dual &d) noexcept {
    using std::sqrt;
    const T s = sqrt(d.val);
    return Dual{s, d.deriv / (T{2} * s)};
  }
  [[nodiscard]] friend constexpr Dual abs(const Dual &d) noexcept {
    using std::abs;
    const T sign = d.val > T{} ? T{1} : d.val < T{} ? T{-1} : T{};
    return Dual{abs(d.val), sign * d.deriv};
  }
  [[nodiscard]] friend constexpr Dual asin(const Dual &d) noexcept {
    using std::asin, std::sqrt;
    return Dual{asin(d.val), d.deriv / sqrt(T{1} - d.val * d.val)};
  }
  [[nodiscard]] friend constexpr Dual acos(const Dual &d) noexcept {
    using std::acos, std::sqrt;
    return Dual{acos(d.val), -d.deriv / sqrt(T{1} - d.val * d.val)};
  }
  [[nodiscard]] friend constexpr Dual atan(const Dual &d) noexcept {
    using std::atan;
    return Dual{atan(d.val), d.deriv / (T{1} + d.val * d.val)};
  }
  [[nodiscard]] friend constexpr Dual sinh(const Dual &d) noexcept {
    using std::sinh, std::cosh;
    return Dual{sinh(d.val), cosh(d.val) * d.deriv};
  }
  [[nodiscard]] friend constexpr Dual cosh(const Dual &d) noexcept {
    using std::sinh, std::cosh;
    return Dual{cosh(d.val), sinh(d.val) * d.deriv};
  }
  [[nodiscard]] friend constexpr Dual tanh(const Dual &d) noexcept {
    using std::tanh, std::cosh;
    const T c = cosh(d.val);
    return Dual{tanh(d.val), d.deriv / (c * c)};
  }
};

static_assert(Numeric<Dual<double>>);
static_assert(Numeric<Dual<float>>);

namespace {
template <typename T> T dual_scalar_impl(T &&);
template <typename T> T dual_scalar_impl(Dual<T> &&);
template <typename T> consteval bool is_dual_impl(std::type_identity<T>) {
  return false;
}
template <typename T> consteval bool is_dual_impl(std::type_identity<Dual<T>>) {
  return true;
}
} // anonymous namespace

template <typename T>
inline constexpr bool is_dual_v = is_dual_impl(std::type_identity<T>{});
template <typename T>
using dual_scalar_t = decltype(dual_scalar_impl(std::declval<T>()));

// nth_dual_t<T, N> = Dual<Dual<...<T>...>> nested N times
template <typename T, std::size_t N> consteval auto nth_dual_impl() noexcept {
  if constexpr (N == 0) {
    return std::type_identity<T>{};
  } else {
    using Inner = typename decltype(nth_dual_impl<T, N - 1>())::type;
    return std::type_identity<Dual<Inner>>{};
  }
}

template <typename T, std::size_t N>
using nth_dual_t = typename decltype(nth_dual_impl<T, N>())::type;

// How many Dual<> layers wrap T
template <typename T> inline constexpr std::size_t dual_depth_v = 0;
template <typename T>
inline constexpr std::size_t dual_depth_v<Dual<T>> = 1 + dual_depth_v<T>;

template <typename T> auto scalar_base_impl(std::type_identity<T>) -> T;
template <typename T>
auto scalar_base_impl(std::type_identity<Dual<T>>)
    -> decltype(scalar_base_impl(std::type_identity<T>{}));

template <typename T>
using scalar_base_t = decltype(scalar_base_impl(std::type_identity<T>{}));

// embed_constant: lift a base scalar into nth_dual_t<T,N> with zero dual parts.
template <typename T, std::size_t N>
constexpr nth_dual_t<T, N> embed_constant(T val) noexcept {
  if constexpr (N == 0) {
    return val;
  } else {
    return nth_dual_t<T, N>{embed_constant<T, N - 1>(val),
                            nth_dual_t<T, N - 1>{}};
  }
}

// ConstantEmbedder<U>: creates a "zero-derivative" U from a base scalar.
// Specialise for custom numeric types (e.g. TaylorDual) to extend
// eval_seeded_as.
template <typename U> struct ConstantEmbedder {
  static constexpr U embed(scalar_base_t<U> val) noexcept {
    return embed_constant<scalar_base_t<U>, dual_depth_v<U>>(val);
  }
};

// get_real_part: peel N Dual<> layers to recover the base scalar.
template <std::size_t N, typename T>
constexpr auto get_real_part(const T &x) noexcept {
  if constexpr (N == 0) {
    return x;
  } else {
    return get_real_part<N - 1>(x.template get<0>());
  }
}

// ===========================================================================
// autodiff-compatible scalar contract for Dual<>.
//
// These additions let a nested Dual (e.g. nth_dual_t<double, 2>) be used as a
// freestanding numeric scalar inside ordinary templated code — the same role
// autodiff::dual2nd plays.  They cover: value extraction (val/to_double),
// mixing with plain arithmetic scalars at any nesting depth, pow/max/min,
// comparisons (branch-on-value), and isfinite.
// ===========================================================================

// Public aliases matching autodiff's spelling.
using dual = nth_dual_t<double, 1>;      // first-order forward dual
using dual2nd = nth_dual_t<double, 2>;   // second-order (Hessian-capable) dual

// X is a dual or a plain arithmetic scalar — the set of operands these
// helpers accept.
template <typename X>
concept DualOrArithmetic = is_dual_v<std::remove_cvref_t<X>> ||
                           std::is_arithmetic_v<std::remove_cvref_t<X>>;

// val(): recursively peel every Dual<> layer to the underlying base scalar.
template <typename T>
  requires std::is_arithmetic_v<T>
constexpr T val(T x) noexcept {
  return x;
}
template <typename T> constexpr auto val(const Dual<T> &d) noexcept {
  return val(d.template get<0>());
}

// to_double(): val() coerced to double, for branch decisions.
template <typename X> constexpr double to_double(const X &x) noexcept {
  return static_cast<double>(val(x));
}

namespace detail {
// Lift a plain arithmetic scalar into the (possibly nested) dual type D, with
// all derivative parts zero.
template <typename D, typename U>
constexpr D as_constant(U s) noexcept {
  using Base = scalar_base_t<D>;
  return embed_constant<Base, dual_depth_v<D>>(static_cast<Base>(s));
}
} // namespace detail

// Mixed Dual/arithmetic arithmetic.  The in-class hidden friends only mix with
// the *inner* type T; these handle Dual<Dual<...>> combined with a bare scalar
// (e.g. dual2nd * 2.0), promoting the scalar through every layer.  The
// !is_same<U, T> guard keeps the innermost Dual<base> ∘ base case on the
// existing in-class overloads.
#define DIFF_DUAL_SCALAR_OP(OP)                                                 \
  template <typename T, typename U>                                            \
    requires(std::is_arithmetic_v<U> && !std::is_same_v<U, T>)                 \
  constexpr Dual<T> operator OP(const Dual<T> &a, U s) noexcept {              \
    return a OP detail::as_constant<Dual<T>>(s);                               \
  }                                                                            \
  template <typename T, typename U>                                            \
    requires(std::is_arithmetic_v<U> && !std::is_same_v<U, T>)                 \
  constexpr Dual<T> operator OP(U s, const Dual<T> &a) noexcept {              \
    return detail::as_constant<Dual<T>>(s) OP a;                              \
  }
DIFF_DUAL_SCALAR_OP(+)
DIFF_DUAL_SCALAR_OP(-)
DIFF_DUAL_SCALAR_OP(*)
DIFF_DUAL_SCALAR_OP(/)
#undef DIFF_DUAL_SCALAR_OP

// At least one operand is a Dual (plain scalar/scalar pairs stay built-in).
template <typename A, typename B>
concept DualMix = DualOrArithmetic<A> && DualOrArithmetic<B> &&
                  (is_dual_v<std::remove_cvref_t<A>> ||
                   is_dual_v<std::remove_cvref_t<B>>);

// Comparisons decided on the fully-reduced base value.  Only <=> and == are
// written; C++20 synthesises <, >, <=, >=, != from them.
template <typename A, typename B>
  requires DualMix<A, B>
constexpr auto operator<=>(const A &a, const B &b) noexcept {
  return val(a) <=> val(b);
}
template <typename A, typename B>
  requires DualMix<A, B>
constexpr bool operator==(const A &a, const B &b) noexcept {
  return val(a) == val(b);
}

// pow(a, b) = a^b.  d(a^b) = a^b (b' ln a + b a'/a).  Works at any nesting
// depth via ADL (the inner pow/log resolve to the inner Dual's overloads).
template <typename T>
constexpr Dual<T> pow(const Dual<T> &a, const Dual<T> &b) noexcept {
  using std::log, std::pow;
  const T av = a.template get<0>(), ad = a.template get<1>();
  const T bv = b.template get<0>(), bd = b.template get<1>();
  const T p = pow(av, bv);
  return Dual<T>{p, p * (bd * log(av) + bv * ad / av)};
}

// max/min: select the larger/smaller operand by value (derivative follows the
// winner; a constant winner contributes zero derivative).
template <typename T>
constexpr Dual<T> max(const Dual<T> &a, const Dual<T> &b) noexcept {
  return val(a) < val(b) ? b : a;
}
template <typename T>
constexpr Dual<T> min(const Dual<T> &a, const Dual<T> &b) noexcept {
  return val(b) < val(a) ? b : a;
}

// For every binary that has a Dual/Dual core above, generate the two
// scalar-mixing overloads by promoting the bare scalar to a constant Dual and
// forwarding.  One macro covers pow, max and min uniformly.
#define DIFF_PROMOTE_BINARY(NAME)                                              \
  template <typename T, typename U>                                           \
    requires std::is_arithmetic_v<U>                                          \
  constexpr Dual<T> NAME(const Dual<T> &a, U s) noexcept {                     \
    return NAME(a, detail::as_constant<Dual<T>>(s));                           \
  }                                                                           \
  template <typename T, typename U>                                           \
    requires std::is_arithmetic_v<U>                                          \
  constexpr Dual<T> NAME(U s, const Dual<T> &a) noexcept {                     \
    return NAME(detail::as_constant<Dual<T>>(s), a);                           \
  }
DIFF_PROMOTE_BINARY(pow)
DIFF_PROMOTE_BINARY(max)
DIFF_PROMOTE_BINARY(min)
#undef DIFF_PROMOTE_BINARY

// isfinite on the reduced value.
template <typename T> constexpr bool isfinite(const Dual<T> &d) noexcept {
  using std::isfinite;
  return isfinite(val(d));
}

} // namespace diff

namespace std {
template <typename T>
struct tuple_size<diff::Dual<T>> : integral_constant<std::size_t, 2> {};
template <typename T, std::size_t N> struct tuple_element<N, diff::Dual<T>> {
  using type = T;
};
} // namespace std
