#pragma once

#include "expressions.hpp"
#include <cmath>
#include <numbers>
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

// ===========================================================================
// Lazy expression-template layer for Dual — struct-based nodes (same shape as
// autodiff's forward dual, so the optimizer collapses an expression to the
// eager computation with no closure/indirection overhead).  Bin/Mono hold their
// operands and materialize to Dual<T> on demand (conversion / get<> /
// structured binding / assignment).  An Op tag is a stateless functor carrying
// the exact eager formula, so results are bit-identical.
// ===========================================================================
template <typename T, typename Op, typename L, typename R> struct Bin;
template <typename T, typename Op, typename E> struct Mono;

template <typename X> struct is_dual_expr_impl : std::false_type {};
template <typename T, typename Op, typename L, typename R>
struct is_dual_expr_impl<Bin<T, Op, L, R>> : std::true_type {};
template <typename T, typename Op, typename E>
struct is_dual_expr_impl<Mono<T, Op, E>> : std::true_type {};
template <typename X>
inline constexpr bool is_dual_expr_v =
    is_dual_expr_impl<std::remove_cvref_t<X>>::value;

// dual_value_t<X>: the component type T of a Dual<T> or a lazy node over it.
template <typename X> struct dual_value_type;
template <typename T> struct dual_value_type<Dual<T>> {
  using type = T;
};
template <typename T, typename Op, typename L, typename R>
struct dual_value_type<Bin<T, Op, L, R>> {
  using type = T;
};
template <typename T, typename Op, typename E>
struct dual_value_type<Mono<T, Op, E>> {
  using type = T;
};
template <typename X>
using dual_value_t = typename dual_value_type<std::remove_cvref_t<X>>::type;

// A Dual or a lazy Dual node.
template <typename X>
concept DualLike = is_dual_v<std::remove_cvref_t<X>> || is_dual_expr_v<X>;

// mat(): collapse an operand to its concrete form — a node to its Dual<T>, a
// Dual/scalar stays as-is (by reference).
template <typename X> constexpr decltype(auto) mat(const X &x) noexcept {
  if constexpr (is_dual_expr_v<X>) {
    return x.eval();
  } else {
    return (x);
  }
}

// Operand storage in a node: lvalues by const reference (no copy — matters for
// wide scalars like VectorDual<N>); rvalues owned by value (moved in), so a
// node stays valid even when returned from an `auto` energy lambda.  Standard
// expression-template lifetime rule applies: consume a node within the
// full-expression that built it.
template <typename U>
using dual_hold_t = std::conditional_t<std::is_lvalue_reference_v<U>,
                                       const std::remove_reference_t<U> &,
                                       std::remove_cvref_t<U>>;

template <typename T, typename Op, typename L, typename R> struct Bin {
  dual_hold_t<L> lhs;
  dual_hold_t<R> rhs;
  using value_type = Dual<T>;
  [[nodiscard]] constexpr Dual<T> eval() const noexcept {
    return Op{}(mat(lhs), mat(rhs));
  }
  constexpr operator Dual<T>() const noexcept { return eval(); }
  template <std::size_t I> [[nodiscard]] constexpr T get() const noexcept {
    return eval().template get<I>();
  }
};
template <typename T, typename Op, typename E> struct Mono {
  dual_hold_t<E> e;
  using value_type = Dual<T>;
  [[nodiscard]] constexpr Dual<T> eval() const noexcept { return Op{}(mat(e)); }
  constexpr operator Dual<T>() const noexcept { return eval(); }
  template <std::size_t I> [[nodiscard]] constexpr T get() const noexcept {
    return eval().template get<I>();
  }
};

// Laziness pays off only for narrow arithmetic-leaf duals (e.g. dual2nd).  For
// a wide custom scalar like VectorDual<N> (an N-lane SIMD array) the operators
// evaluate eagerly so the lane loops stay vectorizable (see operators below).
template <typename T> struct dual_leaf_arith : std::is_arithmetic<T> {};
template <typename T> struct dual_leaf_arith<Dual<T>> : dual_leaf_arith<T> {};

} // namespace diff

// Tuple protocol for Dual and lazy nodes, declared here (before the operators
// below) so structured bindings `auto [v, d] = ...` resolve via get<> rather
// than falling back to private-member decomposition during instantiation.
namespace std {
template <typename T>
struct tuple_size<diff::Dual<T>> : integral_constant<std::size_t, 2> {};
template <typename T, std::size_t N> struct tuple_element<N, diff::Dual<T>> {
  using type = T;
};
template <typename T, typename Op, typename L, typename R>
struct tuple_size<diff::Bin<T, Op, L, R>> : integral_constant<std::size_t, 2> {};
template <typename T, typename Op, typename L, typename R, std::size_t N>
struct tuple_element<N, diff::Bin<T, Op, L, R>> {
  using type = T;
};
template <typename T, typename Op, typename E>
struct tuple_size<diff::Mono<T, Op, E>> : integral_constant<std::size_t, 2> {};
template <typename T, typename Op, typename E, std::size_t N>
struct tuple_element<N, diff::Mono<T, Op, E>> {
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

// val() on a lazy node: materialize, then peel.
template <typename X>
  requires is_dual_expr_v<X>
constexpr auto val(const X &e) noexcept {
  return val(e.eval());
}

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

// ---- lazy binary operators ------------------------------------------------
// Arithmetic-leaf duals build a Bin node (lazy, struct-based — no closure);
// wide scalars (VectorDual) evaluate eagerly so the lane loops stay vectorizable.
#define DIFF_LAZY_BINOP(OP, COMB)                                              \
  template <DualLike A, DualLike B>                                            \
    requires(std::is_same_v<dual_value_t<A>, dual_value_t<B>>)                 \
  constexpr auto operator OP(A &&a, B &&b) noexcept {                          \
    using T = dual_value_t<A>;                                                 \
    if constexpr (dual_leaf_arith<T>::value)                                   \
      return Bin<T, COMB, A, B>{static_cast<A &&>(a), static_cast<B &&>(b)};   \
    else                                                                       \
      return COMB{}(mat(a), mat(b));                                           \
  }                                                                            \
  template <typename A, typename C>                                           \
    requires(DualLike<A> && ConstOperand<C, A>)                               \
  constexpr auto operator OP(A &&a, C &&s) noexcept {                          \
    using T = dual_value_t<A>;                                                 \
    if constexpr (dual_leaf_arith<T>::value)                                   \
      return Bin<T, COMB, A, C>{static_cast<A &&>(a), static_cast<C &&>(s)};   \
    else                                                                       \
      return COMB{}(mat(a), mat(s));                                           \
  }
DIFF_LAZY_BINOP(+, add_combine)
DIFF_LAZY_BINOP(-, sub_combine)
DIFF_LAZY_BINOP(*, mul_combine)
DIFF_LAZY_BINOP(/, div_combine)
#undef DIFF_LAZY_BINOP

// Scalar-on-the-left: + and * commute (store the dual first); - and / use the
// reversed (C, Dual) combine (store the scalar first).
template <typename C, typename A>
  requires(DualLike<A> && ConstOperand<C, A>)
constexpr auto operator+(C &&s, A &&a) noexcept {
  using T = dual_value_t<A>;
  if constexpr (dual_leaf_arith<T>::value)
    return Bin<T, add_combine, A, C>{static_cast<A &&>(a), static_cast<C &&>(s)};
  else
    return add_combine{}(mat(a), mat(s));
}
template <typename C, typename A>
  requires(DualLike<A> && ConstOperand<C, A>)
constexpr auto operator*(C &&s, A &&a) noexcept {
  using T = dual_value_t<A>;
  if constexpr (dual_leaf_arith<T>::value)
    return Bin<T, mul_combine, A, C>{static_cast<A &&>(a), static_cast<C &&>(s)};
  else
    return mul_combine{}(mat(a), mat(s));
}
template <typename C, typename A>
  requires(DualLike<A> && ConstOperand<C, A>)
constexpr auto operator-(C &&s, A &&a) noexcept {
  using T = dual_value_t<A>;
  if constexpr (dual_leaf_arith<T>::value)
    return Bin<T, sub_combine, C, A>{static_cast<C &&>(s), static_cast<A &&>(a)};
  else
    return sub_combine{}(mat(s), mat(a));
}
template <typename C, typename A>
  requires(DualLike<A> && ConstOperand<C, A>)
constexpr auto operator/(C &&s, A &&a) noexcept {
  using T = dual_value_t<A>;
  if constexpr (dual_leaf_arith<T>::value)
    return Bin<T, div_combine, C, A>{static_cast<C &&>(s), static_cast<A &&>(a)};
  else
    return div_combine{}(mat(s), mat(a));
}

// ---- unary minus + math functions (return Mono nodes) ---------------------
struct neg_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    return DT{-v, -d};
  }
};
template <typename A>
  requires DualLike<A>
constexpr auto operator-(A &&a) noexcept {
  using T = dual_value_t<A>;
  if constexpr (dual_leaf_arith<T>::value)
    return Mono<T, neg_combine, A>{static_cast<A &&>(a)};
  else
    return neg_combine{}(mat(a));
}

#define DIFF_LAZY_UNARY(NAME)                                                  \
  template <typename A>                                                        \
    requires DualLike<A>                                                       \
  constexpr auto NAME(A &&a) noexcept {                                        \
    using T = dual_value_t<A>;                                                 \
    if constexpr (dual_leaf_arith<T>::value)                                   \
      return Mono<T, NAME##_combine, A>{static_cast<A &&>(a)};                 \
    else                                                                       \
      return NAME##_combine{}(mat(a));                                         \
  }
struct sin_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::sin, std::cos;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    return DT{sin(v), cos(v) * d};
  }
};
struct cos_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::sin, std::cos;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    return DT{cos(v), -sin(v) * d};
  }
};
struct exp_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::exp;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    const T e = exp(v);
    return DT{e, e * d};
  }
};
struct tan_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::tan, std::cos;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    const T c = cos(v);
    return DT{tan(v), d / (c * c)};
  }
};
struct log_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::log;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    return DT{log(v), d / v};
  }
};
struct sqrt_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::sqrt;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    const T s = sqrt(v);
    return DT{s, d / (T{2} * s)};
  }
};
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
struct asin_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::asin, std::sqrt;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    return DT{asin(v), d / sqrt(T{1} - v * v)};
  }
};
struct acos_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::acos, std::sqrt;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    return DT{acos(v), -d / sqrt(T{1} - v * v)};
  }
};
struct atan_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::atan;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    return DT{atan(v), d / (T{1} + v * v)};
  }
};
struct sinh_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::sinh, std::cosh;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    return DT{sinh(v), cosh(v) * d};
  }
};
struct cosh_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::sinh, std::cosh;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    return DT{cosh(v), sinh(v) * d};
  }
};
struct tanh_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::tanh, std::cosh;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    const T c = cosh(v);
    return DT{tanh(v), d / (c * c)};
  }
};
struct log10_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::log10;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    const T ln10 = static_cast<T>(std::numbers::ln10);
    return DT{log10(v), d / (v * ln10)};
  }
};
struct cbrt_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::cbrt;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    const T c = cbrt(v);
    return DT{c, d / (T{3} * c * c)};
  }
};
struct asinh_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::asinh, std::sqrt;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    return DT{asinh(v), d / sqrt(v * v + T{1})};
  }
};
struct acosh_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::acosh, std::sqrt;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    return DT{acosh(v), d / sqrt(v * v - T{1})};
  }
};
struct atanh_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::atanh;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    return DT{atanh(v), d / (T{1} - v * v)};
  }
};
struct erf_combine {
  constexpr auto operator()(const auto &x) const noexcept {
    using std::erf, std::exp;
    const auto &[v, d] = x;
    using DT = std::remove_cvref_t<decltype(x)>;
    using T = std::remove_cvref_t<decltype(v)>;
    const T two_over_sqrt_pi = static_cast<T>(2.0 * std::numbers::inv_sqrtpi);
    return DT{erf(v), d * two_over_sqrt_pi * exp(-(v * v))};
  }
};
DIFF_LAZY_UNARY(sin)
DIFF_LAZY_UNARY(cos)
DIFF_LAZY_UNARY(exp)
DIFF_LAZY_UNARY(tan)
DIFF_LAZY_UNARY(log)
DIFF_LAZY_UNARY(log10)
DIFF_LAZY_UNARY(sqrt)
DIFF_LAZY_UNARY(cbrt)
DIFF_LAZY_UNARY(abs)
DIFF_LAZY_UNARY(asin)
DIFF_LAZY_UNARY(acos)
DIFF_LAZY_UNARY(atan)
DIFF_LAZY_UNARY(sinh)
DIFF_LAZY_UNARY(cosh)
DIFF_LAZY_UNARY(tanh)
DIFF_LAZY_UNARY(asinh)
DIFF_LAZY_UNARY(acosh)
DIFF_LAZY_UNARY(atanh)
DIFF_LAZY_UNARY(erf)
#undef DIFF_LAZY_UNARY

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
#define DIFF_LAZY_BINFN(NAME, COMB)                                            \
  template <typename A, typename B>                                            \
    requires(DualLike<A> && DualLike<B> &&                                     \
             std::is_same_v<dual_value_t<A>, dual_value_t<B>>)                 \
  constexpr auto NAME(A &&a, B &&b) noexcept {                                 \
    using T = dual_value_t<A>;                                                 \
    if constexpr (dual_leaf_arith<T>::value)                                   \
      return Bin<T, COMB, A, B>{static_cast<A &&>(a), static_cast<B &&>(b)};   \
    else                                                                       \
      return COMB{}(mat(a), mat(b));                                           \
  }
DIFF_LAZY_BINFN(pow, pow_combine)
DIFF_LAZY_BINFN(max, max_combine)
DIFF_LAZY_BINFN(min, min_combine)
DIFF_LAZY_BINFN(atan2, atan2_combine)
DIFF_LAZY_BINFN(hypot, hypot_combine)
#undef DIFF_LAZY_BINFN

// 3-argument hypot(x, y, z) = sqrt(x² + y² + z²) (all-dual; scalar mixing for
// the ternary form is not provided).  d hypot = (x*dx + y*dy + z*dz) / hypot.
template <typename A, typename B, typename C>
  requires(DualLike<A> && DualLike<B> && DualLike<C> &&
           std::is_same_v<dual_value_t<A>, dual_value_t<B>> &&
           std::is_same_v<dual_value_t<A>, dual_value_t<C>>)
constexpr auto hypot(A &&a, B &&b, C &&c) noexcept {
  using std::hypot;
  using T = dual_value_t<A>;
  const Dual<T> x = mat(a), y = mat(b), z = mat(c);
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
