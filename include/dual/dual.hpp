#pragma once

#include "expr/expressions.hpp"
#include "expr/unary_math.hpp"
#include "util/config.hpp" // DDX_ALWAYS_INLINE
#include "util/fmt.hpp"
#include <array>
#include <cmath>
#include <format>
#include <tuple>
#include <type_traits>

namespace ddx::impl {

// (a+be)(c+de) = ac + (ad+bc)e, so a dual commutes exactly when the scalar
// underneath it does.
template <Numeric T> class Dual;
template <Numeric T>
inline constexpr bool is_commutative_multiply_v<Dual<T>> =
    is_commutative_multiply_v<T>;

// Ref: Clifford, Proc. LMS s1-4 (1873) 381 -- adjoin ε with ε² = 0, so the ε
// part of a product is the product rule and no truncation is ever taken.
template <Numeric T> class Dual {
private:
  T val_{};
  T deriv_{};

public:
  constexpr Dual() noexcept = default;
  constexpr explicit Dual(T v, T d = T{}) noexcept : val_(v), deriv_(d) {}
  constexpr Dual(CArithmetic auto s) noexcept : val_(T(s)), deriv_(T{}) {}

  constexpr Dual &operator+=(const Numeric auto &o) noexcept {
    return *this = *this + o;
  }
  constexpr Dual &operator-=(const Numeric auto &o) noexcept {
    return *this = *this - o;
  }
  constexpr Dual &operator*=(const Numeric auto &o) noexcept {
    return *this = *this * o;
  }
  constexpr Dual &operator/=(const Numeric auto &o) noexcept {
    return *this = *this / o;
  }

  constexpr Dual &operator++() noexcept {
    ++val_;
    return *this;
  }

  [[nodiscard]] constexpr const T &value() const noexcept { return val_; }
  [[nodiscard]] constexpr T &value() noexcept { return val_; }
  [[nodiscard]] constexpr const T &deriv() const noexcept { return deriv_; }
  [[nodiscard]] constexpr T &deriv() noexcept { return deriv_; }

  template <std::size_t Index>
  [[nodiscard]] constexpr const T &get() const noexcept {
    static_assert(Index < 2, "Dual index out of bounds");
    if constexpr (Index == 0) {
      return val_;
    } else {
      return deriv_;
    }
  }
  template <std::size_t Index> [[nodiscard]] constexpr T &get() noexcept {
    static_assert(Index < 2, "Dual index out of bounds");
    if constexpr (Index == 0) {
      return val_;
    } else {
      return deriv_;
    }
  }
};

template <typename T> inline constexpr bool is_dual_v = false;
template <Numeric T> inline constexpr bool is_dual_v<Dual<T>> = true;

namespace detail {
template <Numeric T> inline constexpr bool is_dual_family_v<Dual<T>> = true;
} // namespace detail

template <typename X>
concept DualLike = is_dual_v<std::remove_cvref_t<X>>;

template <Numeric T> struct dual_scalar_type {
  using type = T;
};
template <Numeric T> struct dual_scalar_type<Dual<T>> {
  using type = T;
};
template <Numeric T> using dual_scalar_t = typename dual_scalar_type<T>::type;

template <DualLike X> struct dual_value_type;
template <Numeric T> struct dual_value_type<Dual<T>> {
  using type = T;
};
template <DualLike X>
using dual_value_t = typename dual_value_type<std::remove_cvref_t<X>>::type;

template <typename A, typename B>
concept DualCompatible = DualLike<A> && DualLike<B> &&
                         std::same_as<dual_value_t<A>, dual_value_t<B>>;

} // namespace ddx::impl

namespace std {
template <ddx::impl::Numeric T>
struct tuple_size<ddx::impl::Dual<T>> : integral_constant<std::size_t, 2> {};
template <ddx::impl::Numeric T, std::size_t N>
struct tuple_element<N, ddx::impl::Dual<T>> {
  using type = T;
};
} // namespace std

namespace ddx::impl {

// N of them nested: the Nth-order forward dual.  It has 2^N components, one
// per subset of the ε's, and the all-ones component is the Nth derivative --
// which is what extract_nth reads and make_mixed_seed seeds for.
// Ref: Fike & Alonso, AIAA 2011-886 (N = 2, and the generalisation);
// docs/hyperdual_nth_order_by_example.md draws the lattice.
template <Numeric T, std::size_t N> consteval auto nth_dual_impl() noexcept {
  if constexpr (N == 0) {
    return std::type_identity<T>{};
  } else {
    using Inner = typename decltype(nth_dual_impl<T, N - 1>())::type;
    return std::type_identity<Dual<Inner>>{};
  }
}

template <Numeric T, std::size_t N>
using nth_dual_t = typename decltype(nth_dual_impl<T, N>())::type;

template <Numeric T> inline constexpr std::size_t dual_depth_v = 0;
template <Numeric T>
inline constexpr std::size_t dual_depth_v<Dual<T>> = 1 + dual_depth_v<T>;

template <Numeric T> auto scalar_base_impl(std::type_identity<T>) -> T;
template <Numeric T>
auto scalar_base_impl(std::type_identity<Dual<T>>)
    -> decltype(scalar_base_impl(std::type_identity<T>{}));

template <Numeric T>
using scalar_base_t = decltype(scalar_base_impl(std::type_identity<T>{}));

template <Numeric T, std::size_t N>
constexpr nth_dual_t<T, N> embed_constant(T val) noexcept {
  if constexpr (N == 0) {
    return val;
  } else {
    return nth_dual_t<T, N>{embed_constant<T, N - 1>(val),
                            nth_dual_t<T, N - 1>{}};
  }
}

template <Numeric U> struct ConstantEmbedder {
  static constexpr U embed(scalar_base_t<U> val) noexcept {
    return embed_constant<scalar_base_t<U>, dual_depth_v<U>>(val);
  }
};

template <std::size_t N, Numeric T>
constexpr auto get_real_part(const T &x) noexcept {
  if constexpr (N == 0) {
    return x;
  } else {
    return get_real_part<N - 1>(x.template get<0>());
  }
}

template <typename X>
concept DualOrArithmetic = DualLike<X> || CArithmetic<X>;

template <CArithmetic T> constexpr T val(T x) noexcept { return x; }
template <Numeric T> constexpr auto val(const Dual<T> &d) noexcept {
  return val(d.template get<0>());
}

// Zero in every component; operator== compares val() alone and cannot say
// this.
template <CArithmetic T> constexpr bool all_zero(T x) noexcept {
  return x == T{};
}
template <Numeric T> constexpr bool all_zero(const Dual<T> &d) noexcept {
  return all_zero(d.template get<0>()) && all_zero(d.template get<1>());
}

template <Numeric X> constexpr double to_double(const X &x) noexcept {
  return static_cast<double>(val(x));
}

// A zero-derivative operand for the dual A.
template <typename C, typename A>
concept ConstOperand =
    CArithmetic<C> || std::same_as<std::remove_cvref_t<C>, dual_value_t<A>>;

// Anything other than Dual<T>: selects the (Dual, scalar) formulas below.
template <typename C, typename T>
concept ScalarOperand = !std::same_as<std::remove_cvref_t<C>, Dual<T>>;

template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_add(const Dual<T> &a,
                                             const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  return Dual<T>{av + bv, ad + bd};
}

template <Numeric T, ScalarOperand<T> C>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_add(const Dual<T> &a,
                                             const C &s) noexcept {
  const auto &[av, ad] = a;
  return Dual<T>{av + s, ad};
}
template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_sub(const Dual<T> &a,
                                             const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  return Dual<T>{av - bv, ad - bd};
}
template <Numeric T, ScalarOperand<T> C>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_sub(const Dual<T> &a,
                                             const C &s) noexcept {
  const auto &[av, ad] = a;
  return Dual<T>{av - s, ad};
}
template <Numeric T, ScalarOperand<T> C>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_sub(const C &s,
                                             const Dual<T> &a) noexcept {
  const auto &[av, ad] = a; // s - a == -(a - s);
  return Dual<T>{-(av - s), -ad};
}
template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_mul(const Dual<T> &a,
                                             const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  return Dual<T>{av * bv, ad * bv + av * bd};
}
template <Numeric T, ScalarOperand<T> C>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_mul(const Dual<T> &a,
                                             const C &s) noexcept {
  const auto &[av, ad] = a; // scalar distributes; no zero-derivative term
  return Dual<T>{av * s, ad * s};
}

// Reciprocal form: one hardware division per nesting level instead of two.
template <Numeric T>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_div(const Dual<T> &a,
                                             const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  const T inv = T{1} / bv;
  const T q = av * inv; // value = a / b
  return Dual<T>{q, (ad - q * bd) * inv};
}
template <Numeric T, ScalarOperand<T> C>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_div(const Dual<T> &a,
                                             const C &s) noexcept {
  const auto &[av, ad] = a; // s is a zero-derivative constant
  const T inv = T{1} / T(s);
  return Dual<T>{av * inv, ad * inv};
}
template <Numeric T, ScalarOperand<T> C>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_div(const C &s,
                                             const Dual<T> &a) noexcept {
  const auto &[av, ad] = a; // s / a; inner kept T-on-left (wide-scalar-safe)
  const T inv = T{1} / av;
  const T q = T{s} * inv; // value = s / a
  return Dual<T>{q, -(q * ad) * inv};
}

// All three shapes of each operator: (Dual, Dual), (Dual, scalar) and
// (scalar, Dual).  LEFT spells the last, the only shape that differs between
// operators.  The scalar shapes are separate kernels, never a promotion to a
// zero-derivative Dual: promotion leaves an `ad + 0` IEEE will not let the
// compiler fold.
#define DDX_DUAL_BINOP(OP, COMB, LEFT)                                         \
  template <DualLike A, DualCompatible<A> B>                                   \
  constexpr auto operator OP(A &&a, B &&b) noexcept {                          \
    return COMB(a, b);                                                         \
  }                                                                            \
  template <DualLike A, ConstOperand<A> C>                                     \
  constexpr auto operator OP(A &&a, C &&s) noexcept {                          \
    return COMB(a, s);                                                         \
  }                                                                            \
  template <DualLike A, ConstOperand<A> C>                                     \
  constexpr auto operator OP(C &&s, A &&a) noexcept {                          \
    return LEFT;                                                               \
  }
DDX_DUAL_BINOP(+, dual_add, dual_add(a, s))
DDX_DUAL_BINOP(-, dual_sub, dual_sub(s, a))
DDX_DUAL_BINOP(*, dual_mul, dual_mul(a, s))
DDX_DUAL_BINOP(/, dual_div, dual_div(s, a))
#undef DDX_DUAL_BINOP

// ---- unary minus + math functions (eager) ---------------------------------
template <DualLike A> constexpr auto operator-(A &&a) noexcept {
  const auto &[v, d] = a;
  using DT = std::remove_cvref_t<A>;
  return DT{-v, -d};
}

// Chain rule for a unary math node; the primal is reused when the descriptor
// can express its derivative in terms of f(u).
template <template <Numeric> class Fn> struct unary_dual_combine {
  DDX_ALWAYS_INLINE constexpr auto
  operator()(const DualLike auto &x) const noexcept {
    const auto &[v, d] = x;
    // Comp types the primal and the result, and must stay a Dual at depth >= 2
    // or fv truncates.  S is the base scalar, and the only thing the descriptor
    // may be instantiated at -- at Comp its constants would become
    // zero-derivative duals and leave a `0.0 - x` behind.
    using Comp = std::remove_cvref_t<decltype(v)>;
    using DT = std::remove_cvref_t<decltype(x)>;
    using S = scalar_base_t<Comp>;
    if constexpr (detail::has_deriv_from_value_v<Fn<S>, Comp>) {
      const Comp fv = Fn<S>{}(v);
      return DT{fv, Fn<S>::deriv_from_value(v, fv) * d};
    } else {
      return DT{Fn<S>{}(v), Fn<S>::deriv(v) * d};
    }
  }
};
// Not a unary_dual_combine: the derivative is a sign, and at 0 it is taken as
// 0.  v - v reaches only ±0 and NaN: 0 for the first, NaN for the second, so
// a NaN value carries a NaN derivative.
struct abs_combine {
  constexpr auto operator()(const DualLike auto &x) const noexcept {
    using std::abs;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    const T sign = v > T{} ? T{1} : v < T{} ? T{-1} : T{v - v};
    return DT{abs(v), sign * d};
  }
};
// One chain-rule overload per unary math function, from the registry.
#define DDX_DUAL_UNARY(FN, OP, LABEL)                                          \
  template <DualLike A> constexpr auto FN(A &&a) noexcept {                    \
    return unary_dual_combine<detail::OP##Fn>{}(a);                            \
  }
DDX_UNARY_MATH_TABLE(DDX_DUAL_UNARY)
#undef DDX_DUAL_UNARY

template <DualLike A> constexpr auto abs(A &&a) noexcept {
  return abs_combine{}(a);
}

// ---- comparisons (operate on materialized values) -------------------------
template <typename A, typename B>
concept DualComparable =
    DualOrArithmetic<A> && DualOrArithmetic<B> && (DualLike<A> || DualLike<B>);

template <DualOrArithmetic A, DualComparable<A> B>
constexpr auto operator<=>(const A &a, const B &b) noexcept {
  return val(a) <=> val(b);
}
template <DualOrArithmetic A, DualComparable<A> B>
constexpr bool operator==(const A &a, const B &b) noexcept {
  return val(a) == val(b);
}

// Each all-dual form is followed by its scalar-mixed forms, for the same reason
// as the operators above.  The inner call is unqualified, so ADL makes these
// work at any nesting depth.

// pow(a, b) = a^b.  d(a^b) = a^b (b' ln a + b a'/a).
template <DualLike A, DualCompatible<A> B>
constexpr auto pow(A &&a, B &&b) noexcept {
  using std::log, std::pow;
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  using DT = std::remove_cvref_t<A>;
  using T = std::remove_cvref_t<decltype(av)>;
  const T p = pow(av, bv);
  // A constant exponent arrives here too, embedded with a zero derivative --
  // eval_seeded promotes every leaf.  Its b' ln a term would still be
  // 0 * log(0) = NaN at av <= 0, so the direct form takes over exactly when
  // b' is identically zero.
  if (all_zero(bd)) {
    return DT{p, pow(av, bv - T{1}) * (bv * ad)};
  }
  return DT{p, p * (bd * log(av) + bv * ad / av)};
}

// pow(a, s), s a constant exponent.  d(a^s) = s a^(s-1) a', spent as a second
// pow rather than folded into a^s (s a'/a): the quotient form is 0/0 at
// av == 0 where this one is exact -- d(x^2)/dx at 0 is 0.  The b' ln a term is
// identically absent, so there is no `log` at any depth, and at av < 0 with an
// integral exponent the derivative stays finite where `0 * log(-2)` would be
// NaN.
template <DualLike A, CArithmetic U> constexpr auto pow(A &&a, U s) noexcept {
  using std::pow;
  if constexpr (std::unsigned_integral<std::remove_cvref_t<U>>) {
    // Signed, so the s - 1 below cannot wrap at s == 0.
    return pow(static_cast<A &&>(a), static_cast<long long>(s));
  } else {
    const auto &[av, ad] = a;
    using DT = std::remove_cvref_t<A>;
    using T = std::remove_cvref_t<decltype(av)>;
    const T p = pow(av, s);
    return DT{p, pow(av, s - U{1}) * (s * ad)};
  }
}

// pow(s, a), s a constant base.  d(s^a) = s^a ln(s) a'.  ln(s) is a scalar log
// of a constant, so one libm call however deeply the dual is nested.
template <DualLike A, CArithmetic U> constexpr auto pow(U s, A &&a) noexcept {
  using std::log, std::pow;
  const auto &[av, ad] = a;
  using DT = std::remove_cvref_t<A>;
  using T = std::remove_cvref_t<decltype(av)>;
  const T p = pow(s, av);
  return DT{p, p * (log(s) * ad)};
}

template <DualLike A, DualCompatible<A> B>
constexpr auto max(A &&a, B &&b) noexcept {
  using DT = std::remove_cvref_t<A>;
  // A tie averages the operands -- value unchanged, derivative the mean of
  // the two subgradients -- and an unordered pair is (a-b)*0, NaN in every
  // component from either side.  Both choices are symmetric, the shared
  // convention: the only one stable under the commutative reordering the
  // graph builder applies.
  if (val(a) == val(b)) {
    return DT{a + (b - a) / 2};
  }
  if (val(a) < val(b)) {
    return DT{b};
  }
  if (val(b) < val(a)) {
    return DT{a};
  }
  return DT{(a - b) * DT{}};
}

template <DualLike A, DualCompatible<A> B>
constexpr auto min(A &&a, B &&b) noexcept {
  using DT = std::remove_cvref_t<A>;
  if (val(a) == val(b)) {
    return DT{a + (b - a) / 2};
  }
  if (val(b) < val(a)) {
    return DT{b};
  }
  if (val(a) < val(b)) {
    return DT{a};
  }
  return DT{(a - b) * DT{}};
}

// max/min otherwise select an operand whole, so the bound stays a scalar; at
// a tie against the constant the derivative halves.
template <DualLike A, CArithmetic U> constexpr auto max(A &&a, U s) noexcept {
  using DT = std::remove_cvref_t<A>;
  if (val(a) == s) {
    return DT{a + (s - a) / 2};
  }
  if (val(a) < s) {
    return DT{s};
  }
  if (s < val(a)) {
    return DT{a};
  }
  return DT{(a - s) * DT{}};
}
template <DualLike A, CArithmetic U> constexpr auto max(U s, A &&a) noexcept {
  using DT = std::remove_cvref_t<A>;
  if (val(a) == s) {
    return DT{a + (s - a) / 2};
  }
  if (s < val(a)) {
    return DT{a};
  }
  if (val(a) < s) {
    return DT{s};
  }
  return DT{(a - s) * DT{}};
}
template <DualLike A, CArithmetic U> constexpr auto min(A &&a, U s) noexcept {
  using DT = std::remove_cvref_t<A>;
  if (val(a) == s) {
    return DT{a + (s - a) / 2};
  }
  if (s < val(a)) {
    return DT{s};
  }
  if (val(a) < s) {
    return DT{a};
  }
  return DT{(a - s) * DT{}};
}
template <DualLike A, CArithmetic U> constexpr auto min(U s, A &&a) noexcept {
  using DT = std::remove_cvref_t<A>;
  if (val(a) == s) {
    return DT{a + (s - a) / 2};
  }
  if (val(a) < s) {
    return DT{a};
  }
  if (s < val(a)) {
    return DT{s};
  }
  return DT{(a - s) * DT{}};
}

// atan2(y, x): the first operand is the numerator y, the second is x.
//   d atan2 = ((x/h)*dy - (y/h)*dx) / h  with h = hypot(x, y).
// Scaled by hypot rather than divided by x² + y², which overflows past 1e154
// and underflows the derivative to zero.
template <DualLike A, DualCompatible<A> B>
constexpr auto atan2(A &&y, B &&x) noexcept {
  using std::atan2, std::hypot;
  const auto &[yv, yd] = y;
  const auto &[xv, xd] = x;
  using DT = std::remove_cvref_t<A>;
  const auto h = hypot(xv, yv);
  return DT{atan2(yv, xv), ((xv / h) * yd - (yv / h) * xd) / h};
}

// hypot(x, y) = sqrt(x² + y²).  d hypot = (x/h)*dx + (y/h)*dy: the quotients
// live in [-1, 1], so the derivative cannot overflow where h does not.
template <DualLike A, DualCompatible<A> B>
constexpr auto hypot(A &&a, B &&b) noexcept {
  using std::hypot;
  const auto &[xv, xd] = a;
  const auto &[yv, yd] = b;
  using DT = std::remove_cvref_t<A>;
  const auto h = hypot(xv, yv);
  return DT{h, (xv / h) * xd + (yv / h) * yd};
}

// 3-argument hypot, all-dual only.
template <DualLike A, DualCompatible<A> B, DualCompatible<A> C>
constexpr auto hypot(A &&a, B &&b, C &&c) noexcept {
  using std::hypot;
  using T = dual_value_t<A>;
  const Dual<T> x = a, y = b, z = c;
  const auto &[xv, xd] = x;
  const auto &[yv, yd] = y;
  const auto &[zv, zd] = z;
  const T h = hypot(xv, yv, zv);
  return Dual<T>{h, (xv / h) * xd + (yv / h) * yd + (zv / h) * zd};
}

// Against a constant: the zero term is absent rather than added.
template <DualLike A, CArithmetic U> constexpr auto atan2(A &&y, U s) noexcept {
  using std::atan2, std::hypot;
  const auto &[yv, yd] = y;
  using DT = std::remove_cvref_t<A>;
  const auto h = hypot(yv, s);
  return DT{atan2(yv, s), ((s / h) * yd) / h};
}
template <DualLike A, CArithmetic U> constexpr auto atan2(U s, A &&x) noexcept {
  using std::atan2, std::hypot;
  const auto &[xv, xd] = x;
  using DT = std::remove_cvref_t<A>;
  const auto h = hypot(xv, s);
  return DT{atan2(s, xv), -((s / h) * xd) / h};
}
template <DualLike A, CArithmetic U> constexpr auto hypot(A &&a, U s) noexcept {
  using std::hypot;
  const auto &[av, ad] = a;
  using DT = std::remove_cvref_t<A>;
  const auto h = hypot(av, s);
  return DT{h, (av / h) * ad};
}
template <DualLike A, CArithmetic U> constexpr auto hypot(U s, A &&a) noexcept {
  using std::hypot;
  const auto &[av, ad] = a;
  using DT = std::remove_cvref_t<A>;
  const auto h = hypot(s, av);
  return DT{h, (av / h) * ad};
}

static_assert(Numeric<Dual<double>>);
static_assert(Numeric<Dual<float>>);

using dual = nth_dual_t<double, 1>;    // first-order forward dual
using dual2nd = nth_dual_t<double, 2>; // second-order (Hessian-capable) dual

} // namespace ddx::impl

// `v+de` -- the two-term case of the shared series renderer.
template <ddx::impl::Numeric T>
struct std::formatter<ddx::impl::Dual<T>, char>
    : ddx::impl::detail::dual_formatter_base<T> {
  auto format(const ddx::impl::Dual<T> &d, std::format_context &ctx) const {
    this->series(ctx, std::array{d.value(), d.deriv()});
    return ctx.out();
  }
};
