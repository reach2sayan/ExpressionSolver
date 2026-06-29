#pragma once

#include "expressions.hpp"
#include "unary_math.hpp"
#include <array>
#include <cmath>
#include <functional>
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
    return std::invoke(func{}, static_cast<LT>(lhs), static_cast<RT>(rhs));
  }
};

template <typename T>
struct SumOp : BinaryOp<T, std::plus<void>, FixedString{"+"}> {
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

template <Numeric T>
struct NegateOp : UnaryOp<T, std::negate<void>, FixedString{"-"}> {
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

template <Numeric T>
struct DivideOp : BinaryOp<T, std::divides<void>, FixedString{"/"}> {
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

// Each unary math op pulls its value + derivative from the shared descriptor
// detail::<Name>Fn (unary_math.hpp).  derivative() builds f'(lhs)·lhs'; adjoints
// pushes adj·f'(value).  The same descriptor drives dual.hpp's forward combine.
#define DIFF_UNARY_MATH_OP(NAME, LABEL)                                        \
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

DIFF_UNARY_MATH_OP(SineOp, "sin")
DIFF_UNARY_MATH_OP(CosineOp, "cos")

DIFF_UNARY_MATH_OP(ExpOp, "exp")
DIFF_UNARY_MATH_OP(TanOp, "tan")
DIFF_UNARY_MATH_OP(LogOp, "log")
DIFF_UNARY_MATH_OP(SqrtOp, "sqrt")

template <Numeric T>
struct AbsOp : UnaryOp<T, detail::abs_impl, FixedString{"abs"}> {
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

DIFF_UNARY_MATH_OP(AsinOp, "asin")
DIFF_UNARY_MATH_OP(AcosOp, "acos")
DIFF_UNARY_MATH_OP(AtanOp, "atan")

DIFF_UNARY_MATH_OP(SinhOp, "sinh")
DIFF_UNARY_MATH_OP(CoshOp, "cosh")
DIFF_UNARY_MATH_OP(TanhOp, "tanh")
DIFF_UNARY_MATH_OP(Log10Op, "log10")
DIFF_UNARY_MATH_OP(CbrtOp, "cbrt")
DIFF_UNARY_MATH_OP(AsinhOp, "asinh")
DIFF_UNARY_MATH_OP(AcoshOp, "acosh")
DIFF_UNARY_MATH_OP(AtanhOp, "atanh")
DIFF_UNARY_MATH_OP(ErfOp, "erf")
#undef DIFF_UNARY_MATH_OP

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

template <Numeric T>
struct MaxOp : BinaryOp<T, detail::max_impl, FixedString{"max"}, true> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    return static_cast<T>(lhs) < static_cast<T>(rhs) ? rhs.derivative()
                                                     : lhs.derivative();
  }

  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    using ret_t = std::array<T, sizeof...(CB)>;
    return (cache[cb[0]] < cache[cb[1]]) ? ret_t{T{}, adj} : ret_t{adj, T{}};
  }
};

template <Numeric T>
struct MinOp : BinaryOp<T, detail::min_impl, FixedString{"min"}, true> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
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
