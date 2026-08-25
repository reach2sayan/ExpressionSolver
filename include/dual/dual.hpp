#pragma once

#include "ops/scalar.hpp"
#include "ops/unary_math.hpp"
#include "symbolic/expressions.hpp" // Variable, for dual_var_of
#include "util/config.hpp"          // DDX_ALWAYS_INLINE, DDX_SELF
#include "util/fmt.hpp"
#include <array>
#include <cmath>
#include <format>
#include <numeric> // std::midpoint
#include <tuple>
#include <type_traits>
#include <utility> // std::forward, std::move

namespace ddx::impl {

// (a+be)(c+de) = ac + (ad+bc)e, so a dual commutes exactly when the scalar
// underneath it does.
template <Numeric T> class Dual;
template <Numeric T>
inline constexpr bool is_commutative_multiply_v<Dual<T>> =
    is_commutative_multiply_v<T>;

// A zero-derivative operand for the dual A.
template <typename C, typename A>
concept ConstOperand =
    CArithmetic<C> || std::same_as<std::remove_cvref_t<C>, dual_value_t<A>>;

// Anything other than Dual<T>: selects the (Dual, scalar) formulas below.
template <typename C, typename T>
concept ScalarOperand = !std::same_as<std::remove_cvref_t<C>, Dual<T>>;

// Ref: Clifford, Proc. LMS s1-4 (1873) 381 -- adjoin ε with ε² = 0.
template <Numeric T> class Dual {
private:
  T val_{};
  T deriv_{};

public:
  constexpr Dual() noexcept = default;
  constexpr explicit Dual(T v, T d = T{}) noexcept : val_(v), deriv_(d) {}
  constexpr Dual(CArithmetic auto s) noexcept : val_(T(s)), deriv_(T{}) {}

  // Gated on what the binary operators below take, rather than on Numeric:
  // the body is `*this + o`, so anything looser only moves the refusal from
  // the call site into here.  TaylorDual's four match its own operators the
  // same way, which is why they take the class type and no scalar.
  template <typename B>
    requires DualCompatible<B, Dual> || ConstOperand<B, Dual>
  constexpr Dual &operator+=(const B &o) noexcept {
    return *this = *this + o;
  }
  template <typename B>
    requires DualCompatible<B, Dual> || ConstOperand<B, Dual>
  constexpr Dual &operator-=(const B &o) noexcept {
    return *this = *this - o;
  }
  template <typename B>
    requires DualCompatible<B, Dual> || ConstOperand<B, Dual>
  constexpr Dual &operator*=(const B &o) noexcept {
    return *this = *this * o;
  }
  template <typename B>
    requires DualCompatible<B, Dual> || ConstOperand<B, Dual>
  constexpr Dual &operator/=(const B &o) noexcept {
    return *this = *this / o;
  }

  constexpr Dual &operator++() noexcept {
    ++val_;
    return *this;
  }

private:
  // The one place the value category has to be right: a member of an rvalue
  // Dual is an rvalue, and the parentheses are what make decltype(auto) a
  // reference rather than a copy.  std::forward_like is libstdc++ 14.
  template <std::size_t Index>
  [[nodiscard]] static constexpr decltype(auto) slot(auto &&self) noexcept {
    static_assert(Index < 2, "Dual index out of bounds");
    if constexpr (Index == 0) {
      return (std::forward<decltype(self)>(self).val_);
    } else {
      return (std::forward<decltype(self)>(self).deriv_);
    }
  }

public:
  // P0847 writes each accessor once; DDX_DEDUCING_THIS=off is supported,
  // hence the quartets below.
#if DDX_DEDUCING_THIS
  [[nodiscard]] constexpr decltype(auto) value(DDX_SELF) noexcept {
    return slot<0>(DDX_FWD_SELF);
  }
  [[nodiscard]] constexpr decltype(auto) deriv(DDX_SELF) noexcept {
    return slot<1>(DDX_FWD_SELF);
  }
#else
  [[nodiscard]] constexpr const T &value() const & noexcept { return val_; }
  [[nodiscard]] constexpr T &value() & noexcept { return val_; }
  [[nodiscard]] constexpr const T &&value() const && noexcept {
    return std::move(val_);
  }
  [[nodiscard]] constexpr T &&value() && noexcept { return std::move(val_); }
  [[nodiscard]] constexpr const T &deriv() const & noexcept { return deriv_; }
  [[nodiscard]] constexpr T &deriv() & noexcept { return deriv_; }
  [[nodiscard]] constexpr const T &&deriv() const && noexcept {
    return std::move(deriv_);
  }
  [[nodiscard]] constexpr T &&deriv() && noexcept { return std::move(deriv_); }
#endif

  // get() on an rvalue yields an rvalue, as std::get does.
#if DDX_DEDUCING_THIS
  template <std::size_t Index>
  [[nodiscard]] constexpr decltype(auto) get(DDX_SELF) noexcept {
    return slot<Index>(DDX_FWD_SELF);
  }
#else
  template <std::size_t Index>
  [[nodiscard]] constexpr decltype(auto) get() const & noexcept {
    return slot<Index>(*this);
  }
  template <std::size_t Index>
  [[nodiscard]] constexpr decltype(auto) get() & noexcept {
    return slot<Index>(*this);
  }
  template <std::size_t Index>
  [[nodiscard]] constexpr decltype(auto) get() const && noexcept {
    return slot<Index>(std::move(*this));
  }
  template <std::size_t Index>
  [[nodiscard]] constexpr decltype(auto) get() && noexcept {
    return slot<Index>(std::move(*this));
  }
#endif
};

// The primary lives in util/fmt.hpp, next to the formatter that reads it.
namespace detail {
template <Numeric T> inline constexpr bool is_dual_family_v<Dual<T>> = true;
} // namespace detail

// The Dual ends of the recursions ops/scalar.hpp starts for a plain scalar.
template <Numeric T> constexpr auto val(const Dual<T> &d) noexcept {
  return val(d.template get<0>());
}
template <Numeric T> constexpr bool all_zero(const Dual<T> &d) noexcept {
  const auto &[real, dual] = d;
  return all_zero(real) && all_zero(dual);
}

// dual_var_of<"x">(v) — here rather than beside var_of() because it needs Dual
// complete, not declared.
template <FixedString S, Numeric T>
[[nodiscard]] constexpr auto dual_var_of(const T &) noexcept {
  return Variable<Dual<T>, S>{};
}

template <FixedString S, Numeric T>
[[nodiscard]] constexpr auto dual_var_of(symbol_type<S>, const T &) noexcept {
  return Variable<Dual<T>, S>{};
}

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

// All three shapes of each operator; LEFT spells (scalar, Dual), the only one
// that differs between operators.  Separate kernels rather than a promotion,
// which would leave an `ad + 0` IEEE will not let the compiler fold.
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

// The primal is reused when the descriptor can express its derivative in terms
// of f(u).
template <template <Numeric> class Fn> struct unary_dual_combine {
  DDX_ALWAYS_INLINE constexpr auto
  operator()(const DualLike auto &x) const noexcept {
    const auto &[v, d] = x;
    // Comp must stay a Dual at depth >= 2 or fv truncates.  The descriptor is
    // instantiated at the base scalar S: at Comp its constants would become
    // duals and leave a `0.0 - x` behind.
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
// The derivative is a sign, 0 at 0.  v - v reaches only ±0 and NaN, so a NaN
// value carries a NaN derivative.
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

// Each all-dual form is followed by its scalar-mixed forms.  The inner call is
// unqualified, so ADL makes these work at any nesting depth.

// pow(a, b) = a^b.  d(a^b) = a^b (b' ln a + b a'/a).
template <DualLike A, DualCompatible<A> B>
constexpr auto pow(A &&a, B &&b) noexcept {
  using std::log, std::pow;
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  using DT = std::remove_cvref_t<A>;
  using T = std::remove_cvref_t<decltype(av)>;
  const T p = pow(av, bv);
  // A constant exponent arrives here too, embedded with a zero derivative, and
  // its b' ln a term would still be 0 * log(0) = NaN at av <= 0.
  return (all_zero(bd)) ? DT{p, pow(av, bv - T{1}) * (bv * ad)}
                        : DT{p, p * (bd * log(av) + bv * ad / av)};
}

// pow(a, s), s constant.  d(a^s) = s a^(s-1) a', a second pow rather than
// a^s (s a'/a): the quotient form is 0/0 at av == 0 where this one is exact,
// and with no log a negative av with an integral exponent stays finite.
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
  // A tie averages and an unordered pair is (a-b)*0, NaN from either side:
  // both symmetric, hence stable under the builder's commutative reordering.
  if (val(a) == val(b)) {
    return DT{a + (b - a) / 2};
  } else if (val(a) < val(b)) {
    return DT{b};
  } else if (val(b) < val(a)) {
    return DT{a};
  }
  return DT{(a - b) * DT{}};
}

template <DualLike A, DualCompatible<A> B>
constexpr auto min(A &&a, B &&b) noexcept {
  using DT = std::remove_cvref_t<A>;
  if (val(a) == val(b)) {
    return DT{a + (b - a) / 2};
  } else if (val(b) < val(a)) {
    return DT{b};
  } else if (val(a) < val(b)) {
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
  } else if (val(a) < s) {
    return DT{s};
  } else if (s < val(a)) {
    return DT{a};
  }
  return DT{(a - s) * DT{}};
}
template <DualLike A, CArithmetic U> constexpr auto max(U s, A &&a) noexcept {
  using DT = std::remove_cvref_t<A>;
  if (val(a) == s) {
    return DT{a + (s - a) / 2};
  } else if (s < val(a)) {
    return DT{a};
  } else if (val(a) < s) {
    return DT{s};
  }
  return DT{(a - s) * DT{}};
}
template <DualLike A, CArithmetic U> constexpr auto min(A &&a, U s) noexcept {
  using DT = std::remove_cvref_t<A>;
  if (val(a) == s) {
    return DT{a + (s - a) / 2};
  } else if (s < val(a)) {
    return DT{s};
  } else if (val(a) < s) {
    return DT{a};
  }
  return DT{(a - s) * DT{}};
}
template <DualLike A, CArithmetic U> constexpr auto min(U s, A &&a) noexcept {
  using DT = std::remove_cvref_t<A>;
  if (val(a) == s) {
    return DT{a + (s - a) / 2};
  } else if (val(a) < s) {
    return DT{a};
  } else if (s < val(a)) {
    return DT{s};
  }
  return DT{(a - s) * DT{}};
}

// atan2(y, x), numerator first:
//   d atan2 = ((x/h)*dy - (y/h)*dx) / h  with h = hypot(x, y).
// Scaled by hypot rather than divided by x² + y², which overflows past 1e154.
template <DualLike A, DualCompatible<A> B>
constexpr auto atan2(A &&y, B &&x) noexcept {
  using std::atan2, std::hypot;
  const auto &[yv, yd] = y;
  const auto &[xv, xd] = x;
  using DT = std::remove_cvref_t<A>;
  const auto h = hypot(xv, yv);
  return DT{atan2(yv, xv), ((xv / h) * yd - (yv / h) * xd) / h};
}

// Linear, so it is its own derivative rule.  std::midpoint takes arithmetic
// types only, which is why max/min go through midpoint_impl instead.
template <DualLike A, DualCompatible<A> B>
constexpr auto midpoint(A &&a, B &&b) noexcept {
  using std::midpoint;
  const auto &[xv, xd] = a;
  const auto &[yv, yd] = b;
  using DT = std::remove_cvref_t<A>;
  return DT{midpoint(xv, yv), midpoint(xd, yd)};
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
