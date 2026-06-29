#pragma once

#include "expressions.hpp"
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
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &lhs,
                                 const CExpression auto &rhs, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t LB = child_base_v<Base>;
    constexpr std::size_t RB =
        rhs_base_v<Base, std::remove_cvref_t<decltype(lhs)>>;
    lhs.template backward<LB>(syms, adj, grads, cache);
    rhs.template backward<RB>(syms, adj, grads, cache);
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
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &lhs,
                                 const CExpression auto &rhs, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t LB = child_base_v<Base>;
    constexpr std::size_t RB =
        rhs_base_v<Base, std::remove_cvref_t<decltype(lhs)>>;
    lhs.template backward<LB>(syms, adj * cache[RB], grads, cache);
    rhs.template backward<RB>(syms, adj * cache[LB], grads, cache);
  }
};

template <Numeric T> struct NegateOp : UnaryOp<T, std::negate<void>, FixedString{"-"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept {
    auto d = lhs.derivative();
    return MonoExpression<NegateOp<T>, decltype(d)>{std::move(d)};
  }
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    expr.template backward<child_base_v<Base>>(syms, -adj, grads, cache);
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
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &lhs,
                                 const CExpression auto &rhs, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t LB = child_base_v<Base>;
    constexpr std::size_t RB =
        rhs_base_v<Base, std::remove_cvref_t<decltype(lhs)>>;
    const T b = cache[RB];
    lhs.template backward<LB>(syms, adj / b, grads, cache);
    rhs.template backward<RB>(syms, -adj * cache[LB] / (b * b), grads, cache);
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
struct sine_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::sin;
    return sin(a);
  }
};
struct cosine_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::cos;
    return cos(a);
  }
};
struct tan_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::tan;
    return tan(a);
  }
};
struct log_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::log;
    return log(a);
  }
};
struct sqrt_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::sqrt;
    return sqrt(a);
  }
};
struct abs_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::abs;
    return abs(a);
  }
};
struct asin_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::asin;
    return asin(a);
  }
};
struct acos_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::acos;
    return acos(a);
  }
};
struct atan_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::atan;
    return atan(a);
  }
};
struct sinh_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::sinh;
    return sinh(a);
  }
};
struct cosh_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::cosh;
    return cosh(a);
  }
};
struct tanh_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::tanh;
    return tanh(a);
  }
};
struct exp_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::exp;
    return exp(a);
  }
};
struct log10_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::log10;
    return log10(a);
  }
};
struct cbrt_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::cbrt;
    return cbrt(a);
  }
};
struct asinh_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::asinh;
    return asinh(a);
  }
};
struct acosh_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::acosh;
    return acosh(a);
  }
};
struct atanh_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::atanh;
    return atanh(a);
  }
};
struct erf_impl {
  template <Numeric T> constexpr T operator()(const T &a) const noexcept {
    using std::erf;
    return erf(a);
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

template <Numeric T> struct SineOp : UnaryOp<T, detail::sine_impl, FixedString{"sin"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    using std::cos;
    expr.template backward<CB>(syms, adj * cos(cache[CB]), grads, cache);
  }
};

template <Numeric T> struct CosineOp : UnaryOp<T, detail::cosine_impl, FixedString{"cos"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    using std::sin;
    expr.template backward<CB>(syms, -adj * sin(cache[CB]), grads, cache);
  }
};

template <Numeric T>
constexpr auto CosineOp<T>::derivative(const CExpression auto &lhs) noexcept {
  return Negate<T>(sin(lhs)) * lhs.derivative();
}

template <Numeric T>
constexpr auto SineOp<T>::derivative(const CExpression auto &expr) noexcept {
  return cos(expr) * expr.derivative();
}

template <Numeric T> struct ExpOp : UnaryOp<T, detail::exp_impl, FixedString{"exp"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept {
    return MonoExpression<ExpOp<T>, std::decay_t<decltype(lhs)>>{lhs} *
           lhs.derivative();
  }
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    // exp'(u) == exp(u) == this node's own cached value (cache[Base]).
    expr.template backward<child_base_v<Base>>(syms, adj * cache[Base], grads,
                                               cache);
  }
};

template <Numeric T> struct TanOp : UnaryOp<T, detail::tan_impl, FixedString{"tan"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    using std::cos;
    const T c = cos(cache[CB]);
    expr.template backward<CB>(syms, adj / (c * c), grads, cache);
  }
};

template <Numeric T> struct LogOp : UnaryOp<T, detail::log_impl, FixedString{"log"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    expr.template backward<CB>(syms, adj / cache[CB], grads, cache);
  }
};

template <Numeric T> struct SqrtOp : UnaryOp<T, detail::sqrt_impl, FixedString{"sqrt"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    using std::sqrt;
    expr.template backward<CB>(syms, adj / (T{2} * sqrt(cache[CB])), grads,
                               cache);
  }
};

template <Numeric T> struct AbsOp : UnaryOp<T, detail::abs_impl, FixedString{"abs"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    const T v = cache[CB];
    const T sign = v > T{} ? T{1} : v < T{} ? T{-1} : T{};
    expr.template backward<CB>(syms, adj * sign, grads, cache);
  }
};

template <Numeric T> struct AsinOp : UnaryOp<T, detail::asin_impl, FixedString{"asin"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    using std::sqrt;
    const T v = cache[CB];
    expr.template backward<CB>(syms, adj / sqrt(T{1} - v * v), grads, cache);
  }
};

template <Numeric T> struct AcosOp : UnaryOp<T, detail::acos_impl, FixedString{"acos"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    using std::sqrt;
    const T v = cache[CB];
    expr.template backward<CB>(syms, -adj / sqrt(T{1} - v * v), grads, cache);
  }
};

template <Numeric T> struct AtanOp : UnaryOp<T, detail::atan_impl, FixedString{"atan"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    const T v = cache[CB];
    expr.template backward<CB>(syms, adj / (T{1} + v * v), grads, cache);
  }
};

template <Numeric T> struct SinhOp : UnaryOp<T, detail::sinh_impl, FixedString{"sinh"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    using std::cosh;
    expr.template backward<CB>(syms, adj * cosh(cache[CB]), grads, cache);
  }
};

template <Numeric T> struct CoshOp : UnaryOp<T, detail::cosh_impl, FixedString{"cosh"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    using std::sinh;
    expr.template backward<CB>(syms, adj * sinh(cache[CB]), grads, cache);
  }
};

template <Numeric T> struct TanhOp : UnaryOp<T, detail::tanh_impl, FixedString{"tanh"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    using std::cosh;
    const T c = cosh(cache[CB]);
    expr.template backward<CB>(syms, adj / (c * c), grads, cache);
  }
};

template <Numeric T> struct Log10Op : UnaryOp<T, detail::log10_impl, FixedString{"log10"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    const T ln10 = static_cast<T>(std::numbers::ln10);
    expr.template backward<CB>(syms, adj / (cache[CB] * ln10), grads, cache);
  }
};

template <Numeric T> struct CbrtOp : UnaryOp<T, detail::cbrt_impl, FixedString{"cbrt"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    using std::cbrt;
    const T c = cbrt(cache[CB]);
    expr.template backward<CB>(syms, adj / (T{3} * c * c), grads, cache);
  }
};

template <Numeric T> struct AsinhOp : UnaryOp<T, detail::asinh_impl, FixedString{"asinh"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    using std::sqrt;
    const T v = cache[CB];
    expr.template backward<CB>(syms, adj / sqrt(v * v + T{1}), grads, cache);
  }
};

template <Numeric T> struct AcoshOp : UnaryOp<T, detail::acosh_impl, FixedString{"acosh"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    using std::sqrt;
    const T v = cache[CB];
    expr.template backward<CB>(syms, adj / sqrt(v * v - T{1}), grads, cache);
  }
};

template <Numeric T> struct AtanhOp : UnaryOp<T, detail::atanh_impl, FixedString{"atanh"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    const T v = cache[CB];
    expr.template backward<CB>(syms, adj / (T{1} - v * v), grads, cache);
  }
};

template <Numeric T> struct ErfOp : UnaryOp<T, detail::erf_impl, FixedString{"erf"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t CB = child_base_v<Base>;
    using std::exp;
    const T v = cache[CB];
    const T two_over_sqrt_pi = static_cast<T>(2.0 * std::numbers::inv_sqrtpi);
    expr.template backward<CB>(syms, adj * two_over_sqrt_pi * exp(-(v * v)),
                               grads, cache);
  }
};

// pow(a, b) = a^b.  d(a^b) = a^b * (b' ln a + b a'/a).
template <Numeric T>
struct PowOp : BinaryOp<T, detail::pow_impl, FixedString{"pow"}, true> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs, const CExpression auto &rhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &lhs,
                                 const CExpression auto &rhs, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t LB = child_base_v<Base>;
    constexpr std::size_t RB =
        rhs_base_v<Base, std::remove_cvref_t<decltype(lhs)>>;
    using std::pow, std::log;
    const T a = cache[LB];
    const T b = cache[RB];
    const T p = pow(a, b);
    lhs.template backward<LB>(syms, adj * b * pow(a, b - T{1}), grads, cache);
    rhs.template backward<RB>(syms, adj * p * log(a), grads, cache);
  }
};

// atan2(y, x): lhs is y, rhs is x.  d = (x y' - y x') / (x² + y²).
template <Numeric T>
struct Atan2Op : BinaryOp<T, detail::atan2_impl, FixedString{"atan2"}, true> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs, const CExpression auto &rhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &lhs,
                                 const CExpression auto &rhs, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t LB = child_base_v<Base>;
    constexpr std::size_t RB =
        rhs_base_v<Base, std::remove_cvref_t<decltype(lhs)>>;
    const T y = cache[LB];
    const T x = cache[RB];
    const T q = x * x + y * y;
    lhs.template backward<LB>(syms, adj * x / q, grads, cache);
    rhs.template backward<RB>(syms, -adj * y / q, grads, cache);
  }
};

// hypot(x, y) = sqrt(x² + y²).  d = (x x' + y y') / hypot.
template <Numeric T>
struct HypotOp : BinaryOp<T, detail::hypot_impl, FixedString{"hypot"}, true> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs, const CExpression auto &rhs) noexcept;
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &lhs,
                                 const CExpression auto &rhs, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t LB = child_base_v<Base>;
    constexpr std::size_t RB =
        rhs_base_v<Base, std::remove_cvref_t<decltype(lhs)>>;
    using std::hypot;
    const T x = cache[LB];
    const T y = cache[RB];
    const T h = hypot(x, y);
    lhs.template backward<LB>(syms, adj * x / h, grads, cache);
    rhs.template backward<RB>(syms, adj * y / h, grads, cache);
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
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &lhs,
                                 const CExpression auto &rhs, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t LB = child_base_v<Base>;
    constexpr std::size_t RB =
        rhs_base_v<Base, std::remove_cvref_t<decltype(lhs)>>;
    if (cache[LB] < cache[RB])
      rhs.template backward<RB>(syms, adj, grads, cache);
    else
      lhs.template backward<LB>(syms, adj, grads, cache);
  }
};

template <Numeric T>
struct MinOp : BinaryOp<T, detail::min_impl, FixedString{"min"}, true> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs, const CExpression auto &rhs) noexcept {
    return static_cast<T>(rhs) < static_cast<T>(lhs) ? rhs.derivative()
                                                     : lhs.derivative();
  }
  template <std::size_t Base>
  static constexpr void backward(const CExpression auto &lhs,
                                 const CExpression auto &rhs, T adj,
                                 const auto &syms, auto &grads,
                                 const auto &cache) noexcept {
    constexpr std::size_t LB = child_base_v<Base>;
    constexpr std::size_t RB =
        rhs_base_v<Base, std::remove_cvref_t<decltype(lhs)>>;
    if (cache[RB] < cache[LB])
      rhs.template backward<RB>(syms, adj, grads, cache);
    else
      lhs.template backward<LB>(syms, adj, grads, cache);
  }
};

// --- out-of-line derivative definitions ---

template <Numeric T>
constexpr auto TanOp<T>::derivative(const CExpression auto &lhs) noexcept {
  auto c = cos(lhs);
  return lhs.derivative() / (c * c);
}

template <Numeric T>
constexpr auto LogOp<T>::derivative(const CExpression auto &lhs) noexcept {
  return lhs.derivative() / lhs;
}

template <Numeric T>
constexpr auto SqrtOp<T>::derivative(const CExpression auto &lhs) noexcept {
  return lhs.derivative() / (sqrt(lhs) * T{2});
}

template <Numeric T>
constexpr auto AbsOp<T>::derivative(const CExpression auto &lhs) noexcept {
  auto abs_lhs = MonoExpression<AbsOp<T>, std::decay_t<decltype(lhs)>>{lhs};
  return (lhs / abs_lhs) * lhs.derivative();
}

template <Numeric T>
constexpr auto AsinOp<T>::derivative(const CExpression auto &lhs) noexcept {
  return lhs.derivative() / sqrt(T{1} - lhs * lhs);
}

template <Numeric T>
constexpr auto AcosOp<T>::derivative(const CExpression auto &lhs) noexcept {
  auto d = lhs.derivative() / sqrt(T{1} - lhs * lhs);
  return MonoExpression<NegateOp<T>, decltype(d)>{std::move(d)};
}

template <Numeric T>
constexpr auto AtanOp<T>::derivative(const CExpression auto &lhs) noexcept {
  return lhs.derivative() / (T{1} + lhs * lhs);
}

template <Numeric T>
constexpr auto SinhOp<T>::derivative(const CExpression auto &lhs) noexcept {
  return cosh(lhs) * lhs.derivative();
}

template <Numeric T>
constexpr auto CoshOp<T>::derivative(const CExpression auto &lhs) noexcept {
  return sinh(lhs) * lhs.derivative();
}

template <Numeric T>
constexpr auto TanhOp<T>::derivative(const CExpression auto &lhs) noexcept {
  auto c = cosh(lhs);
  return lhs.derivative() / (c * c);
}

template <Numeric T>
constexpr auto Log10Op<T>::derivative(const CExpression auto &lhs) noexcept {
  return lhs.derivative() / (lhs * T{2.302585092994045901094});
}

template <Numeric T>
constexpr auto CbrtOp<T>::derivative(const CExpression auto &lhs) noexcept {
  auto c = cbrt(lhs);
  return lhs.derivative() / (c * c * T{3});
}

template <Numeric T>
constexpr auto AsinhOp<T>::derivative(const CExpression auto &lhs) noexcept {
  return lhs.derivative() / sqrt(lhs * lhs + T{1});
}

template <Numeric T>
constexpr auto AcoshOp<T>::derivative(const CExpression auto &lhs) noexcept {
  return lhs.derivative() / sqrt(lhs * lhs - T{1});
}

template <Numeric T>
constexpr auto AtanhOp<T>::derivative(const CExpression auto &lhs) noexcept {
  return lhs.derivative() / (T{1} - lhs * lhs);
}

template <Numeric T>
constexpr auto ErfOp<T>::derivative(const CExpression auto &lhs) noexcept {
  return exp(-(lhs * lhs)) * lhs.derivative() * T{1.1283791670955125738962};
}

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

template <Numeric T>
[[nodiscard]] constexpr inline auto Negate(CExpression auto expr) noexcept {
  return MonoExpression<NegateOp<T>, decltype(expr)>{std::move(expr)};
}

} // namespace diff
