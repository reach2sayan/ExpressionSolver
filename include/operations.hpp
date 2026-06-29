#pragma once

#include "expressions.hpp"
#include <array>
#include <cmath>
#include <functional>
#include <numbers>
#include <utility>

namespace diff {

enum class OpType : short {
  Unary = 0,
  Binary = 1,
};
using ExpressionType = OpType;

template <typename F, typename T>
concept cunary_op = std::regular_invocable<F, const T &> &&
                    std::convertible_to<std::invoke_result_t<F, const T &>, T>;

template <typename F, typename T>
concept cbinary_op =
    std::regular_invocable<F, const T &, const T &> &&
    std::convertible_to<std::invoke_result_t<F, const T &, const T &>, T>;

template <typename T, OpType type> struct Op {
  using value_type = T;
  static constexpr OpType op_type = type;
};

template <typename T, typename func, CFixedString auto symbol>
  requires cunary_op<func, T>
struct UnaryOp : Op<T, OpType::Unary> {
  using value_type = Op<T, OpType::Unary>::value_type;
  using func_type = func;
  static constexpr void print(std::ostream &out,
                              const CExpression auto &lhs) noexcept {
    out << symbol.view() << '(' << lhs << ')';
  }
  [[nodiscard]] static constexpr auto
  eval(const CExpression auto &lhs) noexcept {
    using VT = typename std::remove_cvref_t<decltype(lhs)>::value_type;
    return std::invoke(func{}, static_cast<VT>(lhs));
  }
};

template <typename T, typename func, CFixedString auto symbol,
          bool prefix = false>
  requires cbinary_op<func, T>
struct BinaryOp : Op<T, OpType::Binary> {
  using value_type = Op<T, OpType::Binary>::value_type;
  using func_type = func;
  // Infix style for operators ("a+b"); prefix=true gives function notation
  // ("pow(a, b)") for math functions like pow/atan2/hypot/min/max.
  static constexpr void print(std::ostream &out, const CExpression auto &lhs,
                              const CExpression auto &rhs) noexcept {
    if constexpr (prefix)
      out << symbol.view() << '(' << lhs << ", " << rhs << ')';
    else
      out << lhs << symbol.view() << rhs;
  }
  [[nodiscard]] static constexpr auto
  eval(const CExpression auto &lhs, const CExpression auto &rhs) noexcept {
    using LT = typename std::remove_cvref_t<decltype(lhs)>::value_type;
    using RT = typename std::remove_cvref_t<decltype(rhs)>::value_type;
    // Returns a lazy node; materializes at the consumption boundary.
    return std::invoke(func{}, static_cast<LT>(lhs), static_cast<RT>(rhs));
  }
};

template <typename T> struct SumOp : BinaryOp<T, std::plus<void>, FixedString{"+"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    return lhs.derivative() + rhs.derivative();
  }
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &) noexcept {
    return {adj, adj};
  }
};

template <Numeric T>
struct MultiplyOp : BinaryOp<T, std::multiplies<void>, FixedString{"*"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    auto lmul = lhs.derivative() * rhs;
    auto rmul = lhs * rhs.derivative();
    return std::move(lmul) + std::move(rmul);
  }
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    return {adj * cache[cb[1]], adj * cache[cb[0]]};
  }
};

template <Numeric T> struct NegateOp : UnaryOp<T, std::negate<void>, FixedString{"-"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept {
    auto d = lhs.derivative();
    return MonoExpression<NegateOp<T>, decltype(d)>{std::move(d)};
  }
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &) noexcept {
    return {-adj};
  }
};

template <Numeric T> struct DivideOp : BinaryOp<T, std::divides<void>, FixedString{"/"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    auto num_l = lhs.derivative() * rhs;
    auto num_r = lhs * rhs.derivative();
    auto numerator = std::move(num_l) - std::move(num_r);
    auto denominator = rhs * rhs;
    return std::move(numerator) / std::move(denominator);
  }
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    const T b = cache[cb[1]];
    return {adj / b, -adj * cache[cb[0]] / (b * b)};
  }
};

// --- binary math ops (function-style: pow / atan2 / hypot / min / max) ------
// derivative() builds a forward symbolic derivative tree; backward() pushes the
// partial adjoints for reverse mode.  Op-level impls live in detail:: below.

template <Numeric T> struct PowOp;
template <Numeric T> struct Atan2Op;
template <Numeric T> struct HypotOp;
template <Numeric T> struct MaxOp;
template <Numeric T> struct MinOp;

namespace detail {
struct abs_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::abs;
    return abs(a);
  }
};
// Binary impls.  ADL resolves pow/atan2/hypot for diff::Dual (defined in
// dual.hpp) when T is a Dual, and to std::* for plain arithmetic T.
struct pow_impl {
  template <Numeric T>
  constexpr T operator()(const T &a, const T &b) const noexcept {
    using std::pow;
    return pow(a, b);
  }
};
struct atan2_impl {
  template <Numeric T>
  constexpr T operator()(const T &a, const T &b) const noexcept {
    using std::atan2;
    return atan2(a, b);
  }
};
struct hypot_impl {
  template <Numeric T>
  constexpr T operator()(const T &a, const T &b) const noexcept {
    using std::hypot;
    return hypot(a, b);
  }
};
struct max_impl {
  template <Numeric T>
  constexpr T operator()(const T &a, const T &b) const noexcept {
    return a < b ? b : a;
  }
};
struct min_impl {
  template <Numeric T>
  constexpr T operator()(const T &a, const T &b) const noexcept {
    return b < a ? b : a;
  }
};
} // namespace detail

// Each unary math op is declared in one line, co-locating its value f(u) and its
// local derivative f'(u) in a single descriptor functor detail::<Name>Fn:
// operator() gives the value (so the functor doubles as the eval func), and
// deriv() gives f'(u).  Both are dependent expressions: on an Expression operand
// they build a tree (ADL selects the diff:: math builders); on a scalar they
// compute numerically (using std:: math).  The op pulls deriv() for both the
// symbolic derivative() (f'(lhs)·lhs') and the reverse-mode adjoints()
// (adj·f'(value)), so every rule lives in exactly one place.  abs is kept
// explicit (its slope sign(u) has a removable 0/0 at the origin).
#define DIFF_UNARY_MATH_FNS                                                    \
  using std::sin, std::cos, std::tan, std::exp, std::log, std::log10,         \
      std::sqrt, std::cbrt, std::asin, std::acos, std::atan, std::sinh,        \
      std::cosh, std::tanh, std::asinh, std::acosh, std::atanh, std::erf
#define DIFF_UNARY_MATH_OP(NAME, LABEL, VAL, ...)                             \
  namespace detail {                                                           \
  template <Numeric T> struct NAME##Fn {                                       \
    /* value: templated on its own type so a SineOp<T> still evaluates at a    \
       deeper nested-dual type during forward AD.  deriv keeps the op's T for  \
       its literals (it is only ever evaluated at the op's value_type). */     \
    template <Numeric U> constexpr U operator()(const U &u) const noexcept {   \
      DIFF_UNARY_MATH_FNS;                                                     \
      return (VAL);                                                            \
    }                                                                          \
    static constexpr auto deriv(const auto &u) noexcept {                      \
      DIFF_UNARY_MATH_FNS;                                                     \
      return (__VA_ARGS__);                                                    \
    }                                                                          \
  };                                                                           \
  }                                                                            \
  template <Numeric T>                                                         \
  struct NAME : UnaryOp<T, detail::NAME##Fn<T>, FixedString{LABEL}> {          \
    [[nodiscard]] static constexpr auto                                        \
    derivative(const CExpression auto &lhs) noexcept {                         \
      return detail::NAME##Fn<T>::deriv(lhs) * lhs.derivative();               \
    }                                                                          \
    template <std::size_t Base, std::size_t... CB>                             \
    static constexpr std::array<T, sizeof...(CB)>                             \
    adjoints(T adj, const auto &cache) noexcept {                              \
      constexpr std::size_t cb[]{CB...};                                       \
      return {adj * detail::NAME##Fn<T>::deriv(cache[cb[0]])};                 \
    }                                                                          \
  };

DIFF_UNARY_MATH_OP(SineOp, "sin", sin(u), cos(u))
DIFF_UNARY_MATH_OP(CosineOp, "cos", cos(u), -sin(u))

DIFF_UNARY_MATH_OP(ExpOp, "exp", exp(u), exp(u))
DIFF_UNARY_MATH_OP(TanOp, "tan", tan(u), T{1} / (cos(u) * cos(u)))
DIFF_UNARY_MATH_OP(LogOp, "log", log(u), T{1} / u)
DIFF_UNARY_MATH_OP(SqrtOp, "sqrt", sqrt(u), T{1} / (T{2} * sqrt(u)))

template <Numeric T> struct AbsOp : UnaryOp<T, detail::abs_impl, FixedString{"abs"}> {
  // Explicit (not macro-generated): the slope is sign(u), with a removable 0/0
  // at the origin that the generic factor form can't express.
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept {
    auto abs_lhs = MonoExpression<AbsOp<T>, std::decay_t<decltype(lhs)>>{lhs};
    return (lhs / abs_lhs) * lhs.derivative();
  }
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    const T v = cache[cb[0]];
    const T sign = v > T{} ? T{1} : v < T{} ? T{-1} : T{};
    return {adj * sign};
  }
};

DIFF_UNARY_MATH_OP(AsinOp, "asin", asin(u), T{1} / sqrt(T{1} - u * u))
DIFF_UNARY_MATH_OP(AcosOp, "acos", acos(u), T{-1} / sqrt(T{1} - u * u))
DIFF_UNARY_MATH_OP(AtanOp, "atan", atan(u), T{1} / (T{1} + u * u))

DIFF_UNARY_MATH_OP(SinhOp, "sinh", sinh(u), cosh(u))
DIFF_UNARY_MATH_OP(CoshOp, "cosh", cosh(u), sinh(u))
DIFF_UNARY_MATH_OP(TanhOp, "tanh", tanh(u), T{1} / (cosh(u) * cosh(u)))
DIFF_UNARY_MATH_OP(Log10Op, "log10", log10(u),
                   T{1} / (u * static_cast<T>(std::numbers::ln10)))
DIFF_UNARY_MATH_OP(CbrtOp, "cbrt", cbrt(u),
                   T{1} / (T{3} * cbrt(u) * cbrt(u)))
DIFF_UNARY_MATH_OP(AsinhOp, "asinh", asinh(u), T{1} / sqrt(u * u + T{1}))
DIFF_UNARY_MATH_OP(AcoshOp, "acosh", acosh(u), T{1} / sqrt(u * u - T{1}))
DIFF_UNARY_MATH_OP(AtanhOp, "atanh", atanh(u), T{1} / (T{1} - u * u))
DIFF_UNARY_MATH_OP(ErfOp, "erf", erf(u),
                   static_cast<T>(2.0 * std::numbers::inv_sqrtpi) *
                       exp(-(u * u)))
#undef DIFF_UNARY_MATH_OP
#undef DIFF_UNARY_MATH_FNS

// pow(a, b) = a^b.  d(a^b) = a^b * (b' ln a + b a'/a).
template <Numeric T>
struct PowOp : BinaryOp<T, detail::pow_impl, FixedString{"pow"}, true> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs, const CExpression auto &rhs) noexcept;
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    using std::pow, std::log;
    const T a = cache[cb[0]];
    const T b = cache[cb[1]];
    const T p = pow(a, b);
    return {adj * b * pow(a, b - T{1}), adj * p * log(a)};
  }
};

// atan2(y, x): lhs is y, rhs is x.  d = (x y' - y x') / (x² + y²).
template <Numeric T>
struct Atan2Op : BinaryOp<T, detail::atan2_impl, FixedString{"atan2"}, true> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs, const CExpression auto &rhs) noexcept;
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    const T y = cache[cb[0]];
    const T x = cache[cb[1]];
    const T q = x * x + y * y;
    return {adj * x / q, -adj * y / q};
  }
};

// hypot(x, y) = sqrt(x² + y²).  d = (x x' + y y') / hypot.
template <Numeric T>
struct HypotOp : BinaryOp<T, detail::hypot_impl, FixedString{"hypot"}, true> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs, const CExpression auto &rhs) noexcept;
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    using std::hypot;
    const T x = cache[cb[0]];
    const T y = cache[cb[1]];
    const T h = hypot(x, y);
    return {adj * x / h, adj * y / h};
  }
};

// max / min: a non-smooth selection — the (sub)gradient follows the selected
// operand.  derivative() is well-formed only when both operands' derivative
// trees share a type (e.g. variables/constants); reverse mode always works.
template <Numeric T>
struct MaxOp : BinaryOp<T, detail::max_impl, FixedString{"max"}, true> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs, const CExpression auto &rhs) noexcept {
    return static_cast<T>(lhs) < static_cast<T>(rhs) ? rhs.derivative()
                                                     : lhs.derivative();
  }
  // Subgradient: the full adjoint flows to the selected operand and zero to the
  // other, which the generic reverse sweep still visits with a zero adjoint.
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    if (cache[cb[0]] < cache[cb[1]])
      return {T{}, adj};
    return {adj, T{}};
  }
};

template <Numeric T>
struct MinOp : BinaryOp<T, detail::min_impl, FixedString{"min"}, true> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs, const CExpression auto &rhs) noexcept {
    return static_cast<T>(rhs) < static_cast<T>(lhs) ? rhs.derivative()
                                                     : lhs.derivative();
  }
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    if (cache[cb[1]] < cache[cb[0]])
      return {T{}, adj};
    return {adj, T{}};
  }
};

// --- out-of-line derivative definitions (binary math ops) ---

template <Numeric T>
constexpr auto PowOp<T>::derivative(const CExpression auto &lhs,
                                    const CExpression auto &rhs) noexcept {
  return pow(lhs, rhs) *
         (rhs.derivative() * log(lhs) + rhs * (lhs.derivative() / lhs));
}

template <Numeric T>
constexpr auto Atan2Op<T>::derivative(const CExpression auto &lhs,
                                      const CExpression auto &rhs) noexcept {
  return (rhs * lhs.derivative() - lhs * rhs.derivative()) /
         (rhs * rhs + lhs * lhs);
}

template <Numeric T>
constexpr auto HypotOp<T>::derivative(const CExpression auto &lhs,
                                      const CExpression auto &rhs) noexcept {
  return (lhs * lhs.derivative() + rhs * rhs.derivative()) / hypot(lhs, rhs);
}

} // namespace diff
