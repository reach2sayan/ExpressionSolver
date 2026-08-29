#pragma once

#include "ops/adjoints.hpp" // detail::sign
#include "ops/scalar.hpp"
#include "ops/unary_math.hpp"
#include "symbolic/expressions.hpp" // Variable, for dual_var_of
#include "util/config.hpp"          // DDX_ALWAYS_INLINE, DDX_SLOT_ACCESSOR
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

// Ref: Clifford, Proc. LMS s1-4 (1873) 381 -- adjoin ε with ε² = 0.
template <Numeric T> class Dual : public compound_from_binary {
private:
  T val_{};
  T deriv_{};

public:
  constexpr Dual() noexcept = default;
  constexpr explicit Dual(T v, T d = T{}) noexcept : val_(v), deriv_(d) {}
  constexpr Dual(CArithmetic auto s) noexcept : val_(T(s)), deriv_(T{}) {}

  constexpr Dual &operator++() noexcept {
    ++val_;
    return *this;
  }

private:
  // The one place the value category has to be right: the parentheses are what
  // make decltype(auto) a reference rather than a copy.  forward_like is
  // libstdc++ 14.
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
  DDX_SLOT_ACCESSOR(value, 0)
  DDX_SLOT_ACCESSOR(deriv, 1)

  // get() on an rvalue yields an rvalue, as std::get does.  No subscript
  // spelling: a dual is a pair with names, not a map.
  DDX_KEYED_GET(std::size_t Index, Index)
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

template <Numeric T, ConstOperand<Dual<T>> C>
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
template <Numeric T, ConstOperand<Dual<T>> C>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_sub(const Dual<T> &a,
                                             const C &s) noexcept {
  const auto &[av, ad] = a;
  return Dual<T>{av - s, ad};
}
template <Numeric T, ConstOperand<Dual<T>> C>
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
template <Numeric T, ConstOperand<Dual<T>> C>
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
template <Numeric T, ConstOperand<Dual<T>> C>
DDX_ALWAYS_INLINE constexpr Dual<T> dual_div(const Dual<T> &a,
                                             const C &s) noexcept {
  const auto &[av, ad] = a; // s is a zero-derivative constant
  const T inv = T{1} / T(s);
  return Dual<T>{av * inv, ad * inv};
}
template <Numeric T, ConstOperand<Dual<T>> C>
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
struct abs_combine {
  constexpr auto operator()(const DualLike auto &x) const noexcept {
    using std::abs;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    return DT{abs(v), detail::sign(v) * d};
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
  // its b' ln a term would still be 0 * log(0) = NaN at av <= 0.  At bv == 0
  // the b a^(b-1) term is 0 * a^-1, 0 * inf at av == 0, for what is the
  // constant 1.
  if (all_zero(bd)) {
    return bv == T{} ? DT{p, T{}} : DT{p, pow(av, bv - T{1}) * (bv * ad)};
  }
  return DT{p, p * (bd * log(av) + bv * ad / av)};
}

// pow(a, s), s constant.  d(a^s) = s a^(s-1) a', a second pow rather than
// a^s (s a'/a): the quotient is 0/0 at av == 0 where this is exact, and with no
// log a negative av with an integral exponent stays finite.
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
    // s == 0: a^0 is the constant 1, and s * a^(s-1) is 0 * inf at av == 0.
    return s == U{} ? DT{p, T{}}
                    : DT{p, pow(av, s - U{1}) * (static_cast<T>(s) * ad)};
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

namespace detail {
// A tie averages and an unordered pair is (a-b)*0, NaN from either side: both
// symmetric, hence stable under the builder's commutative reordering.  The dual
// is always the left operand, so neither depends on which spelling was used.
template <bool IsMax, DualLike A, typename B>
constexpr auto extremum(A &&a, const B &other) noexcept {
  using DT = std::remove_cvref_t<A>;
  if (val(a) == val(other)) {
    return DT{a + (other - a) / 2};
  }
  if (IsMax ? val(a) < val(other) : val(other) < val(a)) {
    return DT{other};
  }
  if (IsMax ? val(other) < val(a) : val(a) < val(other)) {
    return DT{a};
  }
  return DT{(a - other) * DT{}};
}
} // namespace detail

template <DualLike A, DualCompatible<A> B>
constexpr auto max(A &&a, B &&b) noexcept {
  return detail::extremum<true>(std::forward<A>(a), b);
}

template <DualLike A, DualCompatible<A> B>
constexpr auto min(A &&a, B &&b) noexcept {
  return detail::extremum<false>(std::forward<A>(a), b);
}

// max/min otherwise select an operand whole, so the bound stays a scalar; at
// a tie against the constant the derivative halves.
template <DualLike A, CArithmetic U> constexpr auto max(A &&a, U s) noexcept {
  return detail::extremum<true>(std::forward<A>(a), s);
}
template <DualLike A, CArithmetic U> constexpr auto max(U s, A &&a) noexcept {
  return detail::extremum<true>(std::forward<A>(a), s);
}
template <DualLike A, CArithmetic U> constexpr auto min(A &&a, U s) noexcept {
  return detail::extremum<false>(std::forward<A>(a), s);
}
template <DualLike A, CArithmetic U> constexpr auto min(U s, A &&a) noexcept {
  return detail::extremum<false>(std::forward<A>(a), s);
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
