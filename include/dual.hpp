#pragma once

#include "expressions.hpp"
#include "unary_math.hpp"
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

  template <typename U>
    requires std::is_arithmetic_v<U>
  constexpr Dual(U s) noexcept : val(T(s)), deriv(T{}) {}
  template <typename O> constexpr Dual &operator+=(const O &o) noexcept {
    return *this = *this + o;
  }
  template <typename O> constexpr Dual &operator-=(const O &o) noexcept {
    return *this = *this - o;
  }
  template <typename O> constexpr Dual &operator*=(const O &o) noexcept {
    return *this = *this * o;
  }
  template <typename O> constexpr Dual &operator/=(const O &o) noexcept {
    return *this = *this / o;
  }

  constexpr Dual &operator++() noexcept {
    ++val;
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &out, const Dual &d) {
    return out << d.val << "+" << d.deriv << "e";
  }

  template <std::size_t Index>
  [[nodiscard]] constexpr const T &get() const noexcept {
    static_assert(Index < 2, "Dual index out of bounds");
    if constexpr (Index == 0) {
      return val;
    } else {
      return deriv;
    }
  }
  template <std::size_t Index> [[nodiscard]] constexpr T &get() noexcept {
    static_assert(Index < 2, "Dual index out of bounds");
    if constexpr (Index == 0) {
      return val;
    } else {
      return deriv;
    }
  }
};

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

// dual_value_t<X>: the component type T of a Dual<T>.
template <typename X> struct dual_value_type;
template <typename T> struct dual_value_type<Dual<T>> {
  using type = T;
};
template <typename X>
using dual_value_t = typename dual_value_type<std::remove_cvref_t<X>>::type;

template <typename X>
concept DualLike = is_dual_v<std::remove_cvref_t<X>>;

} // namespace diff

namespace std {
template <typename T>
struct tuple_size<diff::Dual<T>> : integral_constant<std::size_t, 2> {};
template <typename T, std::size_t N> struct tuple_element<N, diff::Dual<T>> {
  using type = T;
};
} // namespace std

namespace diff {

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

// Public aliases matching autodiff's spelling.
using dual = nth_dual_t<double, 1>;    // first-order forward dual
using dual2nd = nth_dual_t<double, 2>; // second-order (Hessian-capable) dual

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

template <typename X> constexpr double to_double(const X &x) noexcept {
  return static_cast<double>(val(x));
}

namespace detail {
template <typename D, typename U> constexpr D as_constant(U s) noexcept {
  using Base = scalar_base_t<D>;
  return embed_constant<Base, dual_depth_v<D>>(static_cast<Base>(s));
}
} // namespace detail

template <typename C, typename A>
concept ConstOperand = std::is_arithmetic_v<std::remove_cvref_t<C>> ||
                       std::is_same_v<std::remove_cvref_t<C>, dual_value_t<A>>;

template <typename T>
constexpr Dual<T> dual_add(const Dual<T> &a, const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  return Dual<T>{av + bv, ad + bd};
}

template <typename T, typename C>
  requires(!std::is_same_v<std::remove_cvref_t<C>, Dual<T>>)
constexpr Dual<T> dual_add(const Dual<T> &a, const C &s) noexcept {
  const auto &[av, ad] = a;
  return Dual<T>{av + s, ad};
}
template <typename T>
constexpr Dual<T> dual_sub(const Dual<T> &a, const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  return Dual<T>{av - bv, ad - bd};
}
template <typename T, typename C>
  requires(!std::is_same_v<std::remove_cvref_t<C>, Dual<T>>)
constexpr Dual<T> dual_sub(const Dual<T> &a, const C &s) noexcept {
  const auto &[av, ad] = a;
  return Dual<T>{av - s, ad};
}
template <typename T, typename C>
  requires(!std::is_same_v<std::remove_cvref_t<C>, Dual<T>>)
constexpr Dual<T> dual_sub(const C &s, const Dual<T> &a) noexcept {
  const auto &[av, ad] = a; // s - a == -(a - s);
  return Dual<T>{-(av - s), -ad};
}
template <typename T>
constexpr Dual<T> dual_mul(const Dual<T> &a, const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  return Dual<T>{av * bv, ad * bv + av * bd};
}
template <typename T, typename C>
  requires(!std::is_same_v<std::remove_cvref_t<C>, Dual<T>>)
constexpr Dual<T> dual_mul(const Dual<T> &a, const C &s) noexcept {
  const auto &[av, ad] = a; // scalar distributes; no zero-derivative term
  return Dual<T>{av * s, ad * s};
}
// Division in reciprocal form: compute inv = 1/denominator once, then use
// multiplies (one hardware division per nesting level instead of two).  This is
// what VectorDual and autodiff do; it's not bit-identical to the textbook
// quotient form (it reassociates) but agrees to rounding.  Inner ops stay
// T-on-T so VectorDual is safe.
template <typename T>
constexpr Dual<T> dual_div(const Dual<T> &a, const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  const T inv = T{1} / bv;
  const T q = av * inv; // value = a / b
  return Dual<T>{q, (ad - q * bd) * inv};
}
template <typename T, typename C>
  requires(!std::is_same_v<std::remove_cvref_t<C>, Dual<T>>)
constexpr Dual<T> dual_div(const Dual<T> &a, const C &s) noexcept {
  const auto &[av, ad] = a; // s is a zero-derivative constant
  const T inv = T{1} / T(s);
  return Dual<T>{av * inv, ad * inv};
}
template <typename T, typename C>
  requires(!std::is_same_v<std::remove_cvref_t<C>, Dual<T>>)
constexpr Dual<T> dual_div(const C &s, const Dual<T> &a) noexcept {
  const auto &[av, ad] = a; // s / a; inner kept T-on-left (VectorDual-safe)
  const T inv = T{1} / av;
  const T q = T(s) * inv; // value = s / a
  return Dual<T>{q, -(q * ad) * inv};
}

// ---- Op tags: stateless combiners carrying the exact eager formulas --------
// Each picks the (Dual,Dual) or (Dual,C) formula by overload resolution on the
// materialized operands.
struct add_combine {
  constexpr auto operator()(const auto &x, const auto &y) const noexcept {
    return dual_add(x, y);
  }
};
struct sub_combine {
  constexpr auto operator()(const auto &x, const auto &y) const noexcept {
    return dual_sub(x, y);
  }
};
struct mul_combine {
  constexpr auto operator()(const auto &x, const auto &y) const noexcept {
    return dual_mul(x, y);
  }
};
struct div_combine {
  constexpr auto operator()(const auto &x, const auto &y) const noexcept {
    return dual_div(x, y);
  }
};

// ---- binary operators (eager) ---------------------------------------------
// Each operator computes value+derivative immediately via the *_combine formula
// and returns a concrete Dual<T>.
#define DIFF_DUAL_BINOP(OP, COMB)                                              \
  template <DualLike A, DualLike B>                                            \
    requires(std::is_same_v<dual_value_t<A>, dual_value_t<B>>)                 \
  constexpr auto operator OP(A &&a, B &&b) noexcept {                          \
    return COMB{}(a, b);                                                       \
  }                                                                            \
  template <typename A, typename C>                                            \
    requires(DualLike<A> && ConstOperand<C, A>)                                \
  constexpr auto operator OP(A &&a, C &&s) noexcept {                          \
    return COMB{}(a, s);                                                       \
  }
DIFF_DUAL_BINOP(+, add_combine)
DIFF_DUAL_BINOP(-, sub_combine)
DIFF_DUAL_BINOP(*, mul_combine)
DIFF_DUAL_BINOP(/, div_combine)
#undef DIFF_DUAL_BINOP

// Scalar-on-the-left: + and * commute (pass the dual first); - and / use the
// reversed (C, Dual) combine (pass the scalar first).
template <typename C, DualLike A>
  requires(ConstOperand<C, A>)
constexpr auto operator+(C &&s, A &&a) noexcept {
  return add_combine{}(a, s);
}
template <typename C, DualLike A>
  requires(ConstOperand<C, A>)
constexpr auto operator*(C &&s, A &&a) noexcept {
  return mul_combine{}(a, s);
}
template <typename C, DualLike A>
  requires(ConstOperand<C, A>)
constexpr auto operator-(C &&s, A &&a) noexcept {
  return sub_combine{}(s, a);
}
template <typename C, DualLike A>
  requires(ConstOperand<C, A>)
constexpr auto operator/(C &&s, A &&a) noexcept {
  return div_combine{}(s, a);
}

// ---- unary minus + math functions (eager) ---------------------------------
struct neg_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    return DT{-v, -d};
  }
};

constexpr auto operator-(DualLike auto &&a) noexcept {
  return neg_combine{}(a);
}

#define DIFF_DUAL_UNARY(NAME)                                                  \
  template <typename A>                                                        \
    requires DualLike<A>                                                       \
  constexpr auto NAME(A &&a) noexcept {                                        \
    return NAME##_combine{}(a);                                                \
  }

template <template <typename> class Fn> struct unary_dual_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    const auto &[v, d] = x;
    using T = std::remove_cvref_t<decltype(v)>;
    using DT = std::remove_cvref_t<decltype(x)>;
    return DT{Fn<T>{}(v), Fn<T>::deriv(v) * d};
  }
};
using sin_combine = unary_dual_combine<detail::SineOpFn>;
using cos_combine = unary_dual_combine<detail::CosineOpFn>;
using exp_combine = unary_dual_combine<detail::ExpOpFn>;
using tan_combine = unary_dual_combine<detail::TanOpFn>;
using log_combine = unary_dual_combine<detail::LogOpFn>;
using log10_combine = unary_dual_combine<detail::Log10OpFn>;
using sqrt_combine = unary_dual_combine<detail::SqrtOpFn>;
using cbrt_combine = unary_dual_combine<detail::CbrtOpFn>;
using asin_combine = unary_dual_combine<detail::AsinOpFn>;
using acos_combine = unary_dual_combine<detail::AcosOpFn>;
using atan_combine = unary_dual_combine<detail::AtanOpFn>;
using sinh_combine = unary_dual_combine<detail::SinhOpFn>;
using cosh_combine = unary_dual_combine<detail::CoshOpFn>;
using tanh_combine = unary_dual_combine<detail::TanhOpFn>;
using asinh_combine = unary_dual_combine<detail::AsinhOpFn>;
using acosh_combine = unary_dual_combine<detail::AcoshOpFn>;
using atanh_combine = unary_dual_combine<detail::AtanhOpFn>;
using erf_combine = unary_dual_combine<detail::ErfOpFn>;
struct abs_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::abs;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    const T sign = v > T{} ? T{1} : v < T{} ? T{-1} : T{};
    return DT{abs(v), sign * d};
  }
};
DIFF_DUAL_UNARY(sin)
DIFF_DUAL_UNARY(cos)
DIFF_DUAL_UNARY(exp)
DIFF_DUAL_UNARY(tan)
DIFF_DUAL_UNARY(log)
DIFF_DUAL_UNARY(log10)
DIFF_DUAL_UNARY(sqrt)
DIFF_DUAL_UNARY(cbrt)
DIFF_DUAL_UNARY(abs)
DIFF_DUAL_UNARY(asin)
DIFF_DUAL_UNARY(acos)
DIFF_DUAL_UNARY(atan)
DIFF_DUAL_UNARY(sinh)
DIFF_DUAL_UNARY(cosh)
DIFF_DUAL_UNARY(tanh)
DIFF_DUAL_UNARY(asinh)
DIFF_DUAL_UNARY(acosh)
DIFF_DUAL_UNARY(atanh)
DIFF_DUAL_UNARY(erf)
#undef DIFF_DUAL_UNARY

// ---- comparisons (operate on materialized values) -------------------------
template <typename A, typename B>
concept DualComparable =
    (DualLike<A> || std::is_arithmetic_v<std::remove_cvref_t<A>>) &&
    (DualLike<B> || std::is_arithmetic_v<std::remove_cvref_t<B>>) &&
    (DualLike<A> || DualLike<B>);

template <typename A, typename B>
  requires DualComparable<A, B>
constexpr auto operator<=>(const A &a, const B &b) noexcept {
  return val(a) <=> val(b);
}
template <typename A, typename B>
  requires DualComparable<A, B>
constexpr bool operator==(const A &a, const B &b) noexcept {
  return val(a) == val(b);
}

// ---- pow / max / min ------------------------------------------------------
// pow(a, b) = a^b.  d(a^b) = a^b (b' ln a + b a'/a).
struct pow_combine {
  constexpr auto operator()(const auto &x, const auto &y) const noexcept {
    using std::log, std::pow;
    const auto &[av, ad] = x;
    const auto &[bv, bd] = y;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(av)>;
    const T p = pow(av, bv);
    return DT{p, p * (bd * log(av) + bv * ad / av)};
  }
};
struct max_combine {
  constexpr auto operator()(const auto &x, const auto &y) const noexcept {
    using DT = std::remove_cvref_t<decltype(x)>;
    return val(x) < val(y) ? DT{y} : DT{x};
  }
};
struct min_combine {
  constexpr auto operator()(const auto &x, const auto &y) const noexcept {
    using DT = std::remove_cvref_t<decltype(x)>;
    return val(y) < val(x) ? DT{y} : DT{x};
  }
};
// atan2(y, x): the first operand is the numerator y, the second is x.
//   d atan2 = (x*dy - y*dx) / (x² + y²).
struct atan2_combine {
  constexpr auto operator()(const auto &y, const auto &x) const noexcept {
    using std::atan2;
    const auto &[yv, yd] = y;
    const auto &[xv, xd] = x;
    using DT = std::remove_cvref_t<decltype(y)>;
    const auto q = xv * xv + yv * yv;
    return DT{atan2(yv, xv), (xv * yd - yv * xd) / q};
  }
};
// hypot(x, y) = sqrt(x² + y²).  d hypot = (x*dx + y*dy) / hypot.
struct hypot_combine {
  constexpr auto operator()(const auto &x, const auto &y) const noexcept {
    using std::hypot;
    const auto &[xv, xd] = x;
    const auto &[yv, yd] = y;
    using DT = std::remove_cvref_t<decltype(x)>;
    const auto h = hypot(xv, yv);
    return DT{h, (xv * xd + yv * yd) / h};
  }
};
#define DIFF_DUAL_BINFN(NAME, COMB)                                            \
  template <typename A, typename B>                                            \
    requires(DualLike<A> && DualLike<B> &&                                     \
             std::is_same_v<dual_value_t<A>, dual_value_t<B>>)                 \
  constexpr auto NAME(A &&a, B &&b) noexcept {                                 \
    return COMB{}(a, b);                                                       \
  }
DIFF_DUAL_BINFN(pow, pow_combine)
DIFF_DUAL_BINFN(max, max_combine)
DIFF_DUAL_BINFN(min, min_combine)
DIFF_DUAL_BINFN(atan2, atan2_combine)
DIFF_DUAL_BINFN(hypot, hypot_combine)
#undef DIFF_DUAL_BINFN

// 3-argument hypot(x, y, z) = sqrt(x² + y² + z²) (all-dual; scalar mixing for
// the ternary form is not provided).  d hypot = (x*dx + y*dy + z*dz) / hypot.
template <typename A, typename B, typename C>
  requires(DualLike<A> && DualLike<B> && DualLike<C> &&
           std::is_same_v<dual_value_t<A>, dual_value_t<B>> &&
           std::is_same_v<dual_value_t<A>, dual_value_t<C>>)
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
  template <typename A, typename U>                                            \
    requires(DualLike<A> && std::is_arithmetic_v<U>)                           \
  constexpr auto NAME(A &&a, U s) noexcept {                                   \
    using T = dual_value_t<A>;                                                 \
    return NAME(static_cast<A &&>(a), detail::as_constant<Dual<T>>(s));        \
  }                                                                            \
  template <typename A, typename U>                                            \
    requires(DualLike<A> && std::is_arithmetic_v<U>)                           \
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

template <typename T> constexpr bool isfinite(const Dual<T> &d) noexcept {
  using std::isfinite;
  return isfinite(val(d));
}

static_assert(Numeric<Dual<double>>);
static_assert(Numeric<Dual<float>>);

} // namespace diff
