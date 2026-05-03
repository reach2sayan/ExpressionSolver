#pragma once

#include "expressions.hpp"
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
                    std::same_as<std::invoke_result_t<F, const T &>, T>;

template <typename F, typename T>
concept cbinary_op =
    std::regular_invocable<F, const T &, const T &> &&
    std::same_as<std::invoke_result_t<F, const T &, const T &>, T>;

template <typename T, OpType type> struct Op {
  using value_type = T;
  constexpr static OpType op_type = type;
};

template <typename T, typename func, char symbol>
  requires cunary_op<func, T>
struct UnaryOp : Op<T, OpType::Unary> {
  using value_type = Op<T, OpType::Unary>::value_type;
  static void print(std::ostream &out, const CExpression auto &lhs) {
    out << symbol << lhs;
  }
  [[nodiscard]] constexpr static auto eval(const CExpression auto &lhs) {
    return std::invoke(func{}, lhs);
  }
};

template <typename T, typename func, char symbol>
  requires cbinary_op<func, T>
struct BinaryOp : Op<T, OpType::Binary> {
  using value_type = Op<T, OpType::Binary>::value_type;
  static void print(std::ostream &out, const CExpression auto &lhs,
                    const CExpression auto &rhs) {
    out << lhs << symbol << rhs;
  }
  [[nodiscard]] constexpr static auto eval(const CExpression auto &lhs,
                                           const CExpression auto &rhs) {
    return std::invoke(func{}, lhs, rhs);
  }
};

template <typename T> struct SumOp : BinaryOp<T, std::plus<T>, '+'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) {
    return lhs.derivative() + rhs.derivative();
  }
  constexpr static void backward(const CExpression auto &lhs,
                                 const CExpression auto &rhs, T adj,
                                 const auto &syms, auto &grads) {
    lhs.backward(syms, adj, grads);
    rhs.backward(syms, adj, grads);
  }
};

template <Numeric T> struct MultiplyOp : BinaryOp<T, std::multiplies<T>, '*'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) {
    auto lmul = lhs.derivative() * rhs;
    auto rmul = lhs * rhs.derivative();
    return std::move(lmul) + std::move(rmul);
  }
  constexpr static void backward(const CExpression auto &lhs,
                                 const CExpression auto &rhs, T adj,
                                 const auto &syms, auto &grads) {
    lhs.backward(syms, adj * static_cast<T>(rhs), grads);
    rhs.backward(syms, adj * static_cast<T>(lhs), grads);
  }
};

template <Numeric T> struct NegateOp : UnaryOp<T, std::negate<T>, '-'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs) {
    auto d = lhs.derivative();
    return MonoExpression<NegateOp<T>, decltype(d)>{std::move(d)};
  }
  constexpr static void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) {
    expr.backward(syms, -adj, grads);
  }
};

template <Numeric T> struct DivideOp : BinaryOp<T, std::divides<T>, '/'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) {
    auto num_l = lhs.derivative() * rhs;
    auto num_r = lhs * rhs.derivative();
    auto numerator = std::move(num_l) - std::move(num_r);
    auto denominator = rhs * rhs;
    return std::move(numerator) / std::move(denominator);
  }
  constexpr static void backward(const CExpression auto &lhs,
                                 const CExpression auto &rhs, T adj,
                                 const auto &syms, auto &grads) {
    const T b = static_cast<T>(rhs);
    lhs.backward(syms, adj / b, grads);
    rhs.backward(syms, -adj * static_cast<T>(lhs) / (b * b), grads);
  }
};

namespace detail {
template <Numeric T> struct sine_impl {
  T operator()(const T &a) const {
    using std::sin;
    return sin(a);
  }
};
template <Numeric T> struct cosine_impl {
  T operator()(const T &a) const {
    using std::cos;
    return cos(a);
  }
};
template <Numeric T> struct tan_impl {
  T operator()(const T &a) const {
    using std::tan;
    return tan(a);
  }
};
template <Numeric T> struct log_impl {
  T operator()(const T &a) const {
    using std::log;
    return log(a);
  }
};
template <Numeric T> struct sqrt_impl {
  T operator()(const T &a) const {
    using std::sqrt;
    return sqrt(a);
  }
};
template <Numeric T> struct abs_impl {
  T operator()(const T &a) const {
    using std::abs;
    return abs(a);
  }
};
template <Numeric T> struct asin_impl {
  T operator()(const T &a) const {
    using std::asin;
    return asin(a);
  }
};
template <Numeric T> struct acos_impl {
  T operator()(const T &a) const {
    using std::acos;
    return acos(a);
  }
};
template <Numeric T> struct atan_impl {
  T operator()(const T &a) const {
    using std::atan;
    return atan(a);
  }
};
template <Numeric T> struct sinh_impl {
  T operator()(const T &a) const {
    using std::sinh;
    return sinh(a);
  }
};
template <Numeric T> struct cosh_impl {
  T operator()(const T &a) const {
    using std::cosh;
    return cosh(a);
  }
};
template <Numeric T> struct tanh_impl {
  T operator()(const T &a) const {
    using std::tanh;
    return tanh(a);
  }
};
template <Numeric T> struct exp_impl {
  T operator()(const T &a) const {
    using std::exp;
    return exp(a);
  }
};
} // namespace detail

template <Numeric T> struct SineOp : UnaryOp<T, detail::sine_impl<T>, '$'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs);
  constexpr static void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) {
    using std::cos;
    expr.backward(syms, adj * cos(static_cast<T>(expr)), grads);
  }
};

template <Numeric T> struct CosineOp : UnaryOp<T, detail::cosine_impl<T>, '['> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs);
  constexpr static void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) {
    using std::sin;
    expr.backward(syms, -adj * sin(static_cast<T>(expr)), grads);
  }
};

template <Numeric T>
constexpr auto CosineOp<T>::derivative(const CExpression auto &lhs) {
  return Negate<T>(sin(lhs)) * lhs.derivative();
}

template <Numeric T>
constexpr auto SineOp<T>::derivative(const CExpression auto &expr) {
  return cos(expr) * expr.derivative();
}

template <Numeric T> struct ExpOp : UnaryOp<T, detail::exp_impl<T>, 'e'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs) {
    return MonoExpression<ExpOp<T>, std::decay_t<decltype(lhs)>>{lhs} *
           lhs.derivative();
  }
  constexpr static void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) {
    using std::exp;
    expr.backward(syms, adj * exp(static_cast<T>(expr)), grads);
  }
};

template <Numeric T> struct TanOp : UnaryOp<T, detail::tan_impl<T>, 't'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs);
  constexpr static void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) {
    using std::cos;
    const T c = cos(static_cast<T>(expr));
    expr.backward(syms, adj / (c * c), grads);
  }
};

template <Numeric T> struct LogOp : UnaryOp<T, detail::log_impl<T>, 'l'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs);
  constexpr static void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) {
    expr.backward(syms, adj / static_cast<T>(expr), grads);
  }
};

template <Numeric T> struct SqrtOp : UnaryOp<T, detail::sqrt_impl<T>, 'q'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs);
  constexpr static void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) {
    using std::sqrt;
    expr.backward(syms, adj / (T{2} * sqrt(static_cast<T>(expr))), grads);
  }
};

template <Numeric T> struct AbsOp : UnaryOp<T, detail::abs_impl<T>, '|'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs);
  constexpr static void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) {
    const T v = static_cast<T>(expr);
    const T sign = v > T{} ? T{1} : v < T{} ? T{-1} : T{};
    expr.backward(syms, adj * sign, grads);
  }
};

template <Numeric T> struct AsinOp : UnaryOp<T, detail::asin_impl<T>, 'S'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs);
  constexpr static void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) {
    using std::sqrt;
    const T v = static_cast<T>(expr);
    expr.backward(syms, adj / sqrt(T{1} - v * v), grads);
  }
};

template <Numeric T> struct AcosOp : UnaryOp<T, detail::acos_impl<T>, 'K'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs);
  constexpr static void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) {
    using std::sqrt;
    const T v = static_cast<T>(expr);
    expr.backward(syms, -adj / sqrt(T{1} - v * v), grads);
  }
};

template <Numeric T> struct AtanOp : UnaryOp<T, detail::atan_impl<T>, 'N'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs);
  constexpr static void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) {
    const T v = static_cast<T>(expr);
    expr.backward(syms, adj / (T{1} + v * v), grads);
  }
};

template <Numeric T> struct SinhOp : UnaryOp<T, detail::sinh_impl<T>, 'H'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs);
  constexpr static void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) {
    using std::cosh;
    expr.backward(syms, adj * cosh(static_cast<T>(expr)), grads);
  }
};

template <Numeric T> struct CoshOp : UnaryOp<T, detail::cosh_impl<T>, 'G'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs);
  constexpr static void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) {
    using std::sinh;
    expr.backward(syms, adj * sinh(static_cast<T>(expr)), grads);
  }
};

template <Numeric T> struct TanhOp : UnaryOp<T, detail::tanh_impl<T>, 'Y'> {
  [[nodiscard]] constexpr static auto
  derivative(const CExpression auto &lhs);
  constexpr static void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) {
    using std::cosh;
    const T c = cosh(static_cast<T>(expr));
    expr.backward(syms, adj / (c * c), grads);
  }
};

// --- out-of-line derivative definitions ---

template <Numeric T>
constexpr auto TanOp<T>::derivative(const CExpression auto &lhs) {
  auto c = cos(lhs);
  return lhs.derivative() / (c * c);
}

template <Numeric T>
constexpr auto LogOp<T>::derivative(const CExpression auto &lhs) {
  return lhs.derivative() / lhs;
}

template <Numeric T>
constexpr auto SqrtOp<T>::derivative(const CExpression auto &lhs) {
  return lhs.derivative() / (sqrt(lhs) * T{2});
}

template <Numeric T>
constexpr auto AbsOp<T>::derivative(const CExpression auto &lhs) {
  auto abs_lhs = MonoExpression<AbsOp<T>, std::decay_t<decltype(lhs)>>{lhs};
  return (lhs / abs_lhs) * lhs.derivative();
}

template <Numeric T>
constexpr auto AsinOp<T>::derivative(const CExpression auto &lhs) {
  return lhs.derivative() / sqrt(T{1} - lhs * lhs);
}

template <Numeric T>
constexpr auto AcosOp<T>::derivative(const CExpression auto &lhs) {
  auto d = lhs.derivative() / sqrt(T{1} - lhs * lhs);
  return MonoExpression<NegateOp<T>, decltype(d)>{std::move(d)};
}

template <Numeric T>
constexpr auto AtanOp<T>::derivative(const CExpression auto &lhs) {
  return lhs.derivative() / (T{1} + lhs * lhs);
}

template <Numeric T>
constexpr auto SinhOp<T>::derivative(const CExpression auto &lhs) {
  return cosh(lhs) * lhs.derivative();
}

template <Numeric T>
constexpr auto CoshOp<T>::derivative(const CExpression auto &lhs) {
  return sinh(lhs) * lhs.derivative();
}

template <Numeric T>
constexpr auto TanhOp<T>::derivative(const CExpression auto &lhs) {
  auto c = cosh(lhs);
  return lhs.derivative() / (c * c);
}

template <Numeric T>
[[nodiscard]] constexpr inline auto Negate(CExpression auto expr) {
  return MonoExpression<NegateOp<T>, decltype(expr)>{std::move(expr)};
}

} // namespace diff
