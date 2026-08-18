#pragma once

#include "expr/expressions.hpp"
#include "expr/unary_math.hpp"
#include <cmath>
#include <ostream>
#include <tuple>
#include <type_traits>

namespace diff {

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

  friend std::ostream &operator<<(std::ostream &out, const Dual &d) {
    return out << d.val_ << "+" << d.deriv_ << "e";
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

template <typename X>
concept DualLike = is_dual_v<std::remove_cvref_t<X>>;

// dual_scalar_t<X>: peel one Dual<> layer if there is one, else X unchanged.
template <Numeric T> struct dual_scalar_type {
  using type = T;
};
template <Numeric T> struct dual_scalar_type<Dual<T>> {
  using type = T;
};
template <Numeric T> using dual_scalar_t = typename dual_scalar_type<T>::type;

// dual_value_t<X>: the component type T of a Dual<T>.
template <DualLike X> struct dual_value_type;
template <Numeric T> struct dual_value_type<Dual<T>> {
  using type = T;
};
template <DualLike X>
using dual_value_t = typename dual_value_type<std::remove_cvref_t<X>>::type;

// Two duals over the same component type — the pairing every binary dual
// operator and math function accepts.
template <typename A, typename B>
concept DualCompatible = DualLike<A> && DualLike<B> &&
                         std::same_as<dual_value_t<A>, dual_value_t<B>>;

} // namespace diff

namespace std {
template <diff::Numeric T>
struct tuple_size<diff::Dual<T>> : integral_constant<std::size_t, 2> {};
template <diff::Numeric T, std::size_t N>
struct tuple_element<N, diff::Dual<T>> {
  using type = T;
};
} // namespace std

namespace diff {

// nth_dual_t<T, N> = Dual<Dual<...<T>...>> nested N times
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

// embed_constant: lift a base scalar into nth_dual_t<T,N> with zero dual parts.
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

// get_real_part: peel N Dual<> layers to recover the base scalar.
template <std::size_t N, Numeric T>
constexpr auto get_real_part(const T &x) noexcept {
  if constexpr (N == 0) {
    return x;
  } else {
    return get_real_part<N - 1>(x.template get<0>());
  }
}

// X is a dual or a plain arithmetic scalar — the set of operands these
// helpers accept.
template <typename X>
concept DualOrArithmetic = DualLike<X> || CArithmetic<X>;

// val(): recursively peel every Dual<> layer to the underlying base scalar.
template <CArithmetic T> constexpr T val(T x) noexcept { return x; }
template <Numeric T> constexpr auto val(const Dual<T> &d) noexcept {
  return val(d.template get<0>());
}

template <Numeric X> constexpr double to_double(const X &x) noexcept {
  return static_cast<double>(val(x));
}

namespace detail {
template <Numeric D, CArithmetic U> constexpr D as_constant(U s) noexcept {
  using Base = scalar_base_t<D>;
  return embed_constant<Base, dual_depth_v<D>>(static_cast<Base>(s));
}
} // namespace detail

// C is a zero-derivative operand for the dual A: either a bare arithmetic
// scalar or A's own component type.
template <typename C, typename A>
concept ConstOperand =
    CArithmetic<C> || std::same_as<std::remove_cvref_t<C>, dual_value_t<A>>;

// C is anything *other* than Dual<T> — selects the (Dual, scalar) formulas
// below without competing with their (Dual, Dual) siblings.
template <typename C, typename T>
concept ScalarOperand = !std::same_as<std::remove_cvref_t<C>, Dual<T>>;

template <Numeric T>
constexpr Dual<T> dual_add(const Dual<T> &a, const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  return Dual<T>{av + bv, ad + bd};
}

template <Numeric T, ScalarOperand<T> C>
constexpr Dual<T> dual_add(const Dual<T> &a, const C &s) noexcept {
  const auto &[av, ad] = a;
  return Dual<T>{av + s, ad};
}
template <Numeric T>
constexpr Dual<T> dual_sub(const Dual<T> &a, const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  return Dual<T>{av - bv, ad - bd};
}
template <Numeric T, ScalarOperand<T> C>
constexpr Dual<T> dual_sub(const Dual<T> &a, const C &s) noexcept {
  const auto &[av, ad] = a;
  return Dual<T>{av - s, ad};
}
template <Numeric T, ScalarOperand<T> C>
constexpr Dual<T> dual_sub(const C &s, const Dual<T> &a) noexcept {
  const auto &[av, ad] = a; // s - a == -(a - s);
  return Dual<T>{-(av - s), -ad};
}
template <Numeric T>
constexpr Dual<T> dual_mul(const Dual<T> &a, const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  return Dual<T>{av * bv, ad * bv + av * bd};
}
template <Numeric T, ScalarOperand<T> C>
constexpr Dual<T> dual_mul(const Dual<T> &a, const C &s) noexcept {
  const auto &[av, ad] = a; // scalar distributes; no zero-derivative term
  return Dual<T>{av * s, ad * s};
}

// Division in reciprocal form: compute inv = 1/denominator once, then use
// multiplies (one hardware division per nesting level instead of two).
template <Numeric T>
constexpr Dual<T> dual_div(const Dual<T> &a, const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  const T inv = T{1} / bv;
  const T q = av * inv; // value = a / b
  return Dual<T>{q, (ad - q * bd) * inv};
}
template <Numeric T, ScalarOperand<T> C>
constexpr Dual<T> dual_div(const Dual<T> &a, const C &s) noexcept {
  const auto &[av, ad] = a; // s is a zero-derivative constant
  const T inv = T{1} / T(s);
  return Dual<T>{av * inv, ad * inv};
}
template <Numeric T, ScalarOperand<T> C>
constexpr Dual<T> dual_div(const C &s, const Dual<T> &a) noexcept {
  const auto &[av, ad] = a; // s / a; inner kept T-on-left (VectorDual-safe)
  const T inv = T{1} / av;
  const T q = T{s} * inv; // value = s / a
  return Dual<T>{q, -(q * ad) * inv};
}

// ---- binary operators (eager) ---------------------------------------------
// All three shapes of each operator in one place: (Dual, Dual), (Dual, scalar)
// and (scalar, Dual).  LEFT spells the last one, which is the only shape that
// differs between operators -- + and * commute, so they hand the dual over
// first and reuse the (Dual, scalar) kernel; - and / do not, so they take the
// reversed kernel instead.
//
// The scalar shapes are separate kernels rather than a promotion of the scalar
// to a zero-derivative Dual, and deliberately so: promotion leaves an `ad + 0`
// that IEEE will not let the compiler fold (-0.0 + 0.0 is +0.0), which at
// Dual<VectorDual<32>> turns a one-instruction add into thirteen.
#define DIFF_DUAL_BINOP(OP, COMB, LEFT)                                        \
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
DIFF_DUAL_BINOP(+, dual_add, dual_add(a, s))
DIFF_DUAL_BINOP(-, dual_sub, dual_sub(s, a))
DIFF_DUAL_BINOP(*, dual_mul, dual_mul(a, s))
DIFF_DUAL_BINOP(/, dual_div, dual_div(s, a))
#undef DIFF_DUAL_BINOP

// ---- unary minus + math functions (eager) ---------------------------------
template <DualLike A> constexpr auto operator-(A &&a) noexcept {
  const auto &[v, d] = a;
  using DT = std::remove_cvref_t<A>;
  return DT{-v, -d};
}

// Chain rule for a unary math node.  When the descriptor can express its
// derivative in terms of f(u), the primal is computed once and reused;
template <template <typename> class Fn> struct unary_dual_combine {
  constexpr auto operator()(const DualLike auto &x) const noexcept {
    const auto &[v, d] = x;
    using T = std::remove_cvref_t<decltype(v)>;
    using DT = std::remove_cvref_t<decltype(x)>;
    if constexpr (detail::has_deriv_from_value_v<Fn<T>, T>) {
      const T fv = Fn<T>{}(v);
      return DT{fv, Fn<T>::deriv_from_value(v, fv) * d};
    } else {
      return DT{Fn<T>{}(v), Fn<T>::deriv(v) * d};
    }
  }
};
// abs is the one unary that is not a unary_dual_combine: its derivative is a
// sign, not a function of the primal, and it is only piecewise differentiable —
// the derivative at 0 is taken as 0 rather than left undefined.
struct abs_combine {
  constexpr auto operator()(const DualLike auto &x) const noexcept {
    using std::abs;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    const T sign = v > T{} ? T{1} : v < T{} ? T{-1} : T{};
    return DT{abs(v), sign * d};
  }
};
// One chain-rule overload per unary math function, generated from the registry
// in expr/unary_math.hpp: the descriptor is mechanically detail::<Op>Fn, so the
// name list is not repeated here.
#define DIFF_DUAL_UNARY(FN, OP, LABEL)                                         \
  template <DualLike A> constexpr auto FN(A &&a) noexcept {                    \
    return unary_dual_combine<detail::OP##Fn>{}(a);                            \
  }
DIFF_UNARY_MATH_TABLE(DIFF_DUAL_UNARY)
#undef DIFF_DUAL_UNARY

template <DualLike A> constexpr auto abs(A &&a) noexcept {
  return abs_combine{}(a);
}

// ---- comparisons (operate on materialized values) -------------------------
template <typename A, typename B>
concept DualComparable =
    DualOrArithmetic<A> && DualOrArithmetic<B> && (DualLike<A> || DualLike<B>);

template <typename A, DualComparable<A> B>
constexpr auto operator<=>(const A &a, const B &b) noexcept {
  return val(a) <=> val(b);
}
template <typename A, DualComparable<A> B>
constexpr bool operator==(const A &a, const B &b) noexcept {
  return val(a) == val(b);
}

// ---- pow / max / min ------------------------------------------------------
// Each takes both operands dual; DIFF_PROMOTE_BINARY below supplies the
// scalar-mixed forms by lifting the scalar to a zero-derivative Dual, so the
// formula is written exactly once.  The inner call (std::pow on scalars,
// diff::pow by ADL on nested duals) is what makes these work at any depth.

// pow(a, b) = a^b.  d(a^b) = a^b (b' ln a + b a'/a).
template <DualLike A, DualCompatible<A> B>
constexpr auto pow(A &&a, B &&b) noexcept {
  using std::log, std::pow;
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  using DT = std::remove_cvref_t<A>;
  using T = std::remove_cvref_t<decltype(av)>;
  const T p = pow(av, bv);
  return DT{p, p * (bd * log(av) + bv * ad / av)};
}

template <DualLike A, DualCompatible<A> B>
constexpr auto max(A &&a, B &&b) noexcept {
  using DT = std::remove_cvref_t<A>;
  return val(a) < val(b) ? DT{b} : DT{a};
}

template <DualLike A, DualCompatible<A> B>
constexpr auto min(A &&a, B &&b) noexcept {
  using DT = std::remove_cvref_t<A>;
  return val(b) < val(a) ? DT{b} : DT{a};
}

// atan2(y, x): the first operand is the numerator y, the second is x.
//   d atan2 = (x*dy - y*dx) / (x² + y²).
template <DualLike A, DualCompatible<A> B>
constexpr auto atan2(A &&y, B &&x) noexcept {
  using std::atan2;
  const auto &[yv, yd] = y;
  const auto &[xv, xd] = x;
  using DT = std::remove_cvref_t<A>;
  const auto q = xv * xv + yv * yv;
  return DT{atan2(yv, xv), (xv * yd - yv * xd) / q};
}

// hypot(x, y) = sqrt(x² + y²).  d hypot = (x*dx + y*dy) / hypot.
template <DualLike A, DualCompatible<A> B>
constexpr auto hypot(A &&a, B &&b) noexcept {
  using std::hypot;
  const auto &[xv, xd] = a;
  const auto &[yv, yd] = b;
  using DT = std::remove_cvref_t<A>;
  const auto h = hypot(xv, yv);
  return DT{h, (xv * xd + yv * yd) / h};
}

// 3-argument hypot(x, y, z) = sqrt(x² + y² + z²) (all-dual; scalar mixing for
// the ternary form is not provided).  d hypot = (x*dx + y*dy + z*dz) / hypot.
template <DualLike A, DualCompatible<A> B, DualCompatible<A> C>
constexpr auto hypot(A &&a, B &&b, C &&c) noexcept {
  using std::hypot;
  using T = dual_value_t<A>;
  const Dual<T> x = a, y = b, z = c;
  const auto &[xv, xd] = x;
  const auto &[yv, yd] = y;
  const auto &[zv, zd] = z;
  const T h = hypot(xv, yv, zv);
  return Dual<T>{h, (xv * xd + yv * yd + zv * zd) / h};
}

#define DIFF_PROMOTE_BINARY(NAME)                                              \
  template <DualLike A, CArithmetic U>                                         \
  constexpr auto NAME(A &&a, U s) noexcept {                                   \
    using T = dual_value_t<A>;                                                 \
    return NAME(static_cast<A &&>(a), detail::as_constant<Dual<T>>(s));        \
  }                                                                            \
  template <DualLike A, CArithmetic U>                                         \
  constexpr auto NAME(U s, A &&a) noexcept {                                   \
    using T = dual_value_t<A>;                                                 \
    return NAME(detail::as_constant<Dual<T>>(s), static_cast<A &&>(a));        \
  }
DIFF_PROMOTE_BINARY(pow)
DIFF_PROMOTE_BINARY(max)
DIFF_PROMOTE_BINARY(min)
DIFF_PROMOTE_BINARY(atan2)
DIFF_PROMOTE_BINARY(hypot)
#undef DIFF_PROMOTE_BINARY

static_assert(Numeric<Dual<double>>);
static_assert(Numeric<Dual<float>>);

using dual = nth_dual_t<double, 1>;    // first-order forward dual
using dual2nd = nth_dual_t<double, 2>; // second-order (Hessian-capable) dual

} // namespace diff
