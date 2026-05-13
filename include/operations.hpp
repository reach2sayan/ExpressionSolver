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
  static constexpr OpType op_type = type;
};

template <typename T, typename func, char symbol>
  requires cunary_op<func, T>
struct UnaryOp : Op<T, OpType::Unary> {
  using value_type = Op<T, OpType::Unary>::value_type;
  using func_type = func;
  static constexpr void print(std::ostream &out, const CExpression auto &lhs) noexcept {
    out << symbol << lhs;
  }
  [[nodiscard]] static constexpr auto eval(const CExpression auto &lhs) noexcept {
    using VT = typename std::remove_cvref_t<decltype(lhs)>::value_type;
    return std::invoke(func{}, static_cast<VT>(lhs));
  }
};

template <typename T, typename func, char symbol>
  requires cbinary_op<func, T>
struct BinaryOp : Op<T, OpType::Binary> {
  using value_type = Op<T, OpType::Binary>::value_type;
  using func_type = func;
  static constexpr void print(std::ostream &out, const CExpression auto &lhs,
                    const CExpression auto &rhs) noexcept {
    out << lhs << symbol << rhs;
  }
  [[nodiscard]] static constexpr auto eval(const CExpression auto &lhs,
                                           const CExpression auto &rhs) noexcept {
    using LT = typename std::remove_cvref_t<decltype(lhs)>::value_type;
    using RT = typename std::remove_cvref_t<decltype(rhs)>::value_type;
    return std::invoke(func{}, static_cast<LT>(lhs), static_cast<RT>(rhs));
  }
};

template <typename T> struct SumOp : BinaryOp<T, std::plus<void>, '+'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    return lhs.derivative() + rhs.derivative();
  }
  static constexpr void backward(const CExpression auto &lhs,
                                 const CExpression auto &rhs, T adj,
                                 const auto &syms, auto &grads) noexcept {
    lhs.backward(syms, adj, grads);
    rhs.backward(syms, adj, grads);
  }
};

template <Numeric T> struct MultiplyOp : BinaryOp<T, std::multiplies<void>, '*'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    auto lmul = lhs.derivative() * rhs;
    auto rmul = lhs * rhs.derivative();
    return std::move(lmul) + std::move(rmul);
  }
  static constexpr void backward(const CExpression auto &lhs,
                                 const CExpression auto &rhs, T adj,
                                 const auto &syms, auto &grads) noexcept {
    lhs.backward(syms, adj * static_cast<T>(rhs), grads);
    rhs.backward(syms, adj * static_cast<T>(lhs), grads);
  }
};

template <Numeric T> struct NegateOp : UnaryOp<T, std::negate<void>, '-'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept {
    auto d = lhs.derivative();
    return MonoExpression<NegateOp<T>, decltype(d)>{std::move(d)};
  }
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) noexcept {
    expr.backward(syms, -adj, grads);
  }
};

template <Numeric T> struct DivideOp : BinaryOp<T, std::divides<void>, '/'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    auto num_l = lhs.derivative() * rhs;
    auto num_r = lhs * rhs.derivative();
    auto numerator = std::move(num_l) - std::move(num_r);
    auto denominator = rhs * rhs;
    return std::move(numerator) / std::move(denominator);
  }
  static constexpr void backward(const CExpression auto &lhs,
                                 const CExpression auto &rhs, T adj,
                                 const auto &syms, auto &grads) noexcept {
    const T b = static_cast<T>(rhs);
    lhs.backward(syms, adj / b, grads);
    rhs.backward(syms, -adj * static_cast<T>(lhs) / (b * b), grads);
  }
};

namespace detail {
struct sine_impl {
  template <Numeric T>
  constexpr T operator()(const T &a) const noexcept { using std::sin; return sin(a); }
};
struct cosine_impl {
  template <Numeric T>
  constexpr T operator()(const T &a) const noexcept { using std::cos; return cos(a); }
};
struct tan_impl {
  template <Numeric T>
  constexpr T operator()(const T &a) const noexcept { using std::tan; return tan(a); }
};
struct log_impl {
  template <Numeric T>
  constexpr T operator()(const T &a) const noexcept { using std::log; return log(a); }
};
struct sqrt_impl {
  template <Numeric T>
  constexpr T operator()(const T &a) const noexcept { using std::sqrt; return sqrt(a); }
};
struct abs_impl {
  template <Numeric T>
  constexpr T operator()(const T &a) const noexcept { using std::abs; return abs(a); }
};
struct asin_impl {
  template <Numeric T>
  constexpr T operator()(const T &a) const noexcept { using std::asin; return asin(a); }
};
struct acos_impl {
  template <Numeric T>
  constexpr T operator()(const T &a) const noexcept { using std::acos; return acos(a); }
};
struct atan_impl {
  template <Numeric T>
  constexpr T operator()(const T &a) const noexcept { using std::atan; return atan(a); }
};
struct sinh_impl {
  template <Numeric T>
  constexpr T operator()(const T &a) const noexcept { using std::sinh; return sinh(a); }
};
struct cosh_impl {
  template <Numeric T>
  constexpr T operator()(const T &a) const noexcept { using std::cosh; return cosh(a); }
};
struct tanh_impl {
  template <Numeric T>
  constexpr T operator()(const T &a) const noexcept { using std::tanh; return tanh(a); }
};
struct exp_impl {
  template <Numeric T>
  constexpr T operator()(const T &a) const noexcept { using std::exp; return exp(a); }
};
} // namespace detail

template <Numeric T> struct SineOp : UnaryOp<T, detail::sine_impl, '$'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) noexcept {
    using std::cos;
    expr.backward(syms, adj * cos(static_cast<T>(expr)), grads);
  }
};

template <Numeric T> struct CosineOp : UnaryOp<T, detail::cosine_impl, '['> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) noexcept {
    using std::sin;
    expr.backward(syms, -adj * sin(static_cast<T>(expr)), grads);
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

template <Numeric T> struct ExpOp : UnaryOp<T, detail::exp_impl, 'e'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept {
    return MonoExpression<ExpOp<T>, std::decay_t<decltype(lhs)>>{lhs} *
           lhs.derivative();
  }
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) noexcept {
    using std::exp;
    expr.backward(syms, adj * exp(static_cast<T>(expr)), grads);
  }
};

template <Numeric T> struct TanOp : UnaryOp<T, detail::tan_impl, 't'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) noexcept {
    using std::cos;
    const T c = cos(static_cast<T>(expr));
    expr.backward(syms, adj / (c * c), grads);
  }
};

template <Numeric T> struct LogOp : UnaryOp<T, detail::log_impl, 'l'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) noexcept {
    expr.backward(syms, adj / static_cast<T>(expr), grads);
  }
};

template <Numeric T> struct SqrtOp : UnaryOp<T, detail::sqrt_impl, 'q'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) noexcept {
    using std::sqrt;
    expr.backward(syms, adj / (T{2} * sqrt(static_cast<T>(expr))), grads);
  }
};

template <Numeric T> struct AbsOp : UnaryOp<T, detail::abs_impl, '|'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) noexcept {
    const T v = static_cast<T>(expr);
    const T sign = v > T{} ? T{1} : v < T{} ? T{-1} : T{};
    expr.backward(syms, adj * sign, grads);
  }
};

template <Numeric T> struct AsinOp : UnaryOp<T, detail::asin_impl, 'S'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) noexcept {
    using std::sqrt;
    const T v = static_cast<T>(expr);
    expr.backward(syms, adj / sqrt(T{1} - v * v), grads);
  }
};

template <Numeric T> struct AcosOp : UnaryOp<T, detail::acos_impl, 'K'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) noexcept {
    using std::sqrt;
    const T v = static_cast<T>(expr);
    expr.backward(syms, -adj / sqrt(T{1} - v * v), grads);
  }
};

template <Numeric T> struct AtanOp : UnaryOp<T, detail::atan_impl, 'N'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) noexcept {
    const T v = static_cast<T>(expr);
    expr.backward(syms, adj / (T{1} + v * v), grads);
  }
};

template <Numeric T> struct SinhOp : UnaryOp<T, detail::sinh_impl, 'H'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) noexcept {
    using std::cosh;
    expr.backward(syms, adj * cosh(static_cast<T>(expr)), grads);
  }
};

template <Numeric T> struct CoshOp : UnaryOp<T, detail::cosh_impl, 'G'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) noexcept {
    using std::sinh;
    expr.backward(syms, adj * sinh(static_cast<T>(expr)), grads);
  }
};

template <Numeric T> struct TanhOp : UnaryOp<T, detail::tanh_impl, 'Y'> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept;
  static constexpr void backward(const CExpression auto &expr, T adj,
                                 const auto &syms, auto &grads) noexcept {
    using std::cosh;
    const T c = cosh(static_cast<T>(expr));
    expr.backward(syms, adj / (c * c), grads);
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
[[nodiscard]] constexpr inline auto Negate(CExpression auto expr) noexcept {
  return MonoExpression<NegateOp<T>, decltype(expr)>{std::move(expr)};
}

} // namespace diff
