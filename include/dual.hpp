#pragma once

#include "expressions.hpp"
#include <cmath>
#include <functional>
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

template <typename T, typename F> struct DualExpr {
  F fn;
  using value_type = Dual<T>;
  [[nodiscard]] constexpr Dual<T> eval() const noexcept { return fn(); }
  constexpr operator Dual<T>() const noexcept { return fn(); }
  template <std::size_t I> [[nodiscard]] constexpr T get() const noexcept {
    return fn().template get<I>();
  }
};

template <typename X> struct is_dual_expr_impl : std::false_type {};
template <typename T, typename F>
struct is_dual_expr_impl<DualExpr<T, F>> : std::true_type {};
template <typename X>
inline constexpr bool is_dual_expr_v =
    is_dual_expr_impl<std::remove_cvref_t<X>>::value;

// dual_value_t<X>: the stored component type T of a Dual<T> or DualExpr<T,F>.
template <typename X> struct dual_value_type;
template <typename T> struct dual_value_type<Dual<T>> {
  using type = T;
};
template <typename T, typename F> struct dual_value_type<DualExpr<T, F>> {
  using type = T;
};
template <typename X>
using dual_value_t = typename dual_value_type<std::remove_cvref_t<X>>::type;

// A Dual or a lazy Dual node.
template <typename X>
concept DualLike = is_dual_v<std::remove_cvref_t<X>> || is_dual_expr_v<X>;

// Laziness is the default, but only pays off for narrow arithmetic-leaf duals
// (e.g. dual2nd: a few doubles the optimizer keeps in registers).  For a wide
// custom scalar like VectorDual<N> (an N-lane SIMD array) the whole performance
// story is auto-vectorizing the lane loops — and any closure/eval indirection
// defeats that.  So for those the operators evaluate eagerly with no node at
// all (see lazy_node): minimum eagerness, applied only where it's needed.
template <typename T> struct dual_leaf_arith : std::is_arithmetic<T> {};
template <typename T> struct dual_leaf_arith<Dual<T>> : dual_leaf_arith<T> {};

// Wrap a closure as a lazy node (always — the eager decision is made by the
// caller, lazy_node, which never reaches here for wide scalars).
template <typename T, typename F>
[[nodiscard]] constexpr DualExpr<T, std::decay_t<F>>
make_dual_expr(F &&f) noexcept {
  return DualExpr<T, std::decay_t<F>>{static_cast<F &&>(f)};
}

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
template <typename T, typename F>
struct tuple_size<diff::DualExpr<T, F>> : integral_constant<std::size_t, 2> {};
template <typename T, typename F, std::size_t N>
struct tuple_element<N, diff::DualExpr<T, F>> {
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

template <typename T, typename F>
constexpr auto val(const DualExpr<T, F> &e) noexcept {
  return val(e.eval());
}

template <typename T> constexpr const Dual<T> &mat(const Dual<T> &d) noexcept {
  return d;
}
template <typename T, typename F>
constexpr Dual<T> mat(const DualExpr<T, F> &e) noexcept {
  return e.eval();
}
template <typename X>
  requires(!is_dual_v<std::remove_cvref_t<X>> && !is_dual_expr_v<X>)
constexpr const X &mat(const X &x) noexcept {
  return x;
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
template <typename T>
constexpr Dual<T> dual_div(const Dual<T> &a, const Dual<T> &b) noexcept {
  const auto &[av, ad] = a;
  const auto &[bv, bd] = b;
  return Dual<T>{av / bv, (ad * bv - av * bd) / (bv * bv)};
}
template <typename T, typename C>
  requires(!std::is_same_v<std::remove_cvref_t<C>, Dual<T>>)
constexpr Dual<T> dual_div(const Dual<T> &a, const C &s) noexcept {
  const auto &[av, ad] = a;
  return Dual<T>{av / s, ad / s};
}
template <typename T, typename C>
  requires(!std::is_same_v<std::remove_cvref_t<C>, Dual<T>>)
constexpr Dual<T> dual_div(const C &s, const Dual<T> &a) noexcept {
  const auto &[av, ad] = a; // s / a; inner kept T-on-left (VectorDual-safe)
  return Dual<T>{T(s) / av, -(ad * s) / (av * av)};
}

// ---- forwarding-capture lazy-node builder ---------------------------------
// Operands are captured per value category: lvalues by reference (no copy —
// important for wide scalars like VectorDual), rvalues moved into the node so
// it owns them and stays valid even if it escapes the full-expression (e.g. an
// energy lambda that `return`s an `auto` expression).
namespace detail {
template <typename X> struct is_ref_wrap : std::false_type {};
template <typename X>
struct is_ref_wrap<std::reference_wrapper<X>> : std::true_type {};

template <typename U> constexpr auto fwd_cap(U &&u) noexcept {
  if constexpr (std::is_lvalue_reference_v<U>) {
    return std::reference_wrapper<std::remove_reference_t<U>>(u);
  } else {
    return std::remove_cvref_t<U>(std::move(u));
  }
}
template <typename X> constexpr const auto &unwrap(const X &x) noexcept {
  if constexpr (is_ref_wrap<X>::value) {
    return x.get();
  } else {
    return x;
  }
}
template <typename T, typename Comb, typename... Ops>
constexpr auto lazy_node(Comb comb, Ops &&...ops) noexcept {
  if constexpr (dual_leaf_arith<T>::value) {
    return make_dual_expr<T>([comb, ... caps = fwd_cap(static_cast<Ops &&>(
                                        ops))]() constexpr noexcept {
      return std::invoke(comb, mat(unwrap(caps))...);
    });
  } else {
    return std::invoke(comb,mat(static_cast<Ops &&>(ops))...);
  }
}
} // namespace detail

// ---- lazy binary operators (return DualExpr nodes) ------------------------
// Both overloads share one combiner: dual_add/sub/mul/div pick the (Dual,Dual)
// or (Dual,C) formula by overload resolution on the materialized operands.
#define DIFF_LAZY_BINOP(OP, COMBINE)                                           \
  template <DualLike A, DualLike B>                                            \
    requires(std::is_same_v<dual_value_t<A>, dual_value_t<B>>)                 \
  constexpr auto operator OP(A &&a, B &&b) noexcept {                          \
    return detail::lazy_node<dual_value_t<A>>(                                 \
        [](const auto &x, const auto &y) constexpr noexcept {                  \
          return COMBINE(x, y);                                                \
        },                                                                     \
        static_cast<A &&>(a), static_cast<B &&>(b));                           \
  }                                                                            \
  template <typename A, typename C>                                            \
    requires(DualLike<A> && ConstOperand<C, A>)                                \
  constexpr auto operator OP(A &&a, C &&s) noexcept {                          \
    return detail::lazy_node<dual_value_t<A>>(                                 \
        [](const auto &x, const auto &y) constexpr noexcept {                  \
          return COMBINE(x, y);                                                \
        },                                                                     \
        static_cast<A &&>(a), static_cast<C &&>(s));                           \
  }
DIFF_LAZY_BINOP(+, dual_add)
DIFF_LAZY_BINOP(-, dual_sub)
DIFF_LAZY_BINOP(*, dual_mul)
DIFF_LAZY_BINOP(/, dual_div)
#undef DIFF_LAZY_BINOP

// Scalar-on-the-left: + and * commute (pass the dual first); - and / use the
// reversed (C, Dual) combine.
template <typename C, typename A>
  requires(DualLike<A> && ConstOperand<C, A>)
constexpr auto operator+(C &&s, A &&a) noexcept {
  return detail::lazy_node<dual_value_t<A>>(
      [](const auto &x, const auto &y) constexpr noexcept {
        return dual_add(x, y);
      },
      static_cast<A &&>(a), static_cast<C &&>(s));
}
template <typename C, typename A>
  requires(DualLike<A> && ConstOperand<C, A>)
constexpr auto operator*(C &&s, A &&a) noexcept {
  return detail::lazy_node<dual_value_t<A>>(
      [](const auto &x, const auto &y) constexpr noexcept {
        return dual_mul(x, y);
      },
      static_cast<A &&>(a), static_cast<C &&>(s));
}
template <typename C, typename A>
  requires(DualLike<A> && ConstOperand<C, A>)
constexpr auto operator-(C &&s, A &&a) noexcept {
  return detail::lazy_node<dual_value_t<A>>(
      [](const auto &x, const auto &y) constexpr noexcept {
        return dual_sub(x, y);
      },
      static_cast<C &&>(s), static_cast<A &&>(a));
}
template <typename C, typename A>
  requires(DualLike<A> && ConstOperand<C, A>)
constexpr auto operator/(C &&s, A &&a) noexcept {
  return detail::lazy_node<dual_value_t<A>>(
      [](const auto &x, const auto &y) constexpr noexcept {
        return dual_div(x, y);
      },
      static_cast<C &&>(s), static_cast<A &&>(a));
}

// ---- unary minus + math functions (return nodes) --------------------------
template <typename A>
  requires DualLike<A>
constexpr auto operator-(A &&a) noexcept {
  return detail::lazy_node<dual_value_t<A>>(
      [](const auto &x) constexpr noexcept {
        const auto &[v, d] = x;
        using DT = std::remove_cvref_t<decltype(x)>;
        return DT{-v, -d};
      },
      static_cast<A &&>(a));
}

#define DIFF_LAZY_UNARY(NAME)                                                  \
  template <typename A>                                                        \
    requires DualLike<A>                                                       \
  constexpr auto NAME(A &&a) noexcept {                                        \
    return detail::lazy_node<dual_value_t<A>>(NAME##_combine{},                \
                                              static_cast<A &&>(a));           \
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
DIFF_LAZY_UNARY(sin)
DIFF_LAZY_UNARY(cos)
DIFF_LAZY_UNARY(exp)
DIFF_LAZY_UNARY(tan)
DIFF_LAZY_UNARY(log)
DIFF_LAZY_UNARY(sqrt)
DIFF_LAZY_UNARY(abs)
DIFF_LAZY_UNARY(asin)
DIFF_LAZY_UNARY(acos)
DIFF_LAZY_UNARY(atan)
DIFF_LAZY_UNARY(sinh)
DIFF_LAZY_UNARY(cosh)
DIFF_LAZY_UNARY(tanh)
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
template <typename A, typename B>
  requires(DualLike<A> && DualLike<B> &&
           std::is_same_v<dual_value_t<A>, dual_value_t<B>>)
constexpr auto pow(A &&a, B &&b) noexcept {
  return detail::lazy_node<dual_value_t<A>>(
      [](const auto &x, const auto &y) constexpr noexcept {
        using std::log, std::pow;
        const auto &[av, ad] = x;
        const auto &[bv, bd] = y;
        using DT = std::remove_cvref_t<decltype(x)>;
        using T = std::remove_cvref_t<decltype(av)>;
        const T p = pow(av, bv);
        return DT{p, p * (bd * log(av) + bv * ad / av)};
      },
      static_cast<A &&>(a), static_cast<B &&>(b));
}
template <typename A, typename B>
  requires(DualLike<A> && DualLike<B> &&
           std::is_same_v<dual_value_t<A>, dual_value_t<B>>)
constexpr auto max(A &&a, B &&b) noexcept {
  return detail::lazy_node<dual_value_t<A>>(
      [](const auto &x, const auto &y) constexpr noexcept {
        using DT = std::remove_cvref_t<decltype(x)>;
        return val(x) < val(y) ? DT{y} : DT{x};
      },
      static_cast<A &&>(a), static_cast<B &&>(b));
}
template <typename A, typename B>
  requires(DualLike<A> && DualLike<B> &&
           std::is_same_v<dual_value_t<A>, dual_value_t<B>>)
constexpr auto min(A &&a, B &&b) noexcept {
  return detail::lazy_node<dual_value_t<A>>(
      [](const auto &x, const auto &y) constexpr noexcept {
        using DT = std::remove_cvref_t<decltype(x)>;
        return val(y) < val(x) ? DT{y} : DT{x};
      },
      static_cast<A &&>(a), static_cast<B &&>(b));
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
#undef DIFF_PROMOTE_BINARY

template <typename T> constexpr bool isfinite(const Dual<T> &d) noexcept {
  using std::isfinite;
  return isfinite(val(d));
}

static_assert(Numeric<Dual<double>>);
static_assert(Numeric<Dual<float>>);

} // namespace diff
