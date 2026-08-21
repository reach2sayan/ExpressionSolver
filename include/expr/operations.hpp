#pragma once

#include "expr/expressions.hpp"
#include "expr/unary_math.hpp"
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <functional>
#include <string_view>
#include <utility>

namespace ddx::impl {

// How an op is written down; only expr/format.hpp reads it.
//   Prefix -a   Infix a + b   Function pow(a, b)
enum class Notation : std::uint8_t { Prefix, Infix, Function };

// Binding strength; higher binds tighter.  Function calls and leaves are
// atomic.
inline constexpr int precedence_atom = 100;

template <Numeric T, typename func, CFixedString auto symbol,
          Notation note = Notation::Function, int prec = precedence_atom>
  requires std::regular_invocable<func, const T &> &&
           std::convertible_to<std::invoke_result_t<func, const T &>, T>
struct UnaryOp {
  using value_type = T;
  using func_type = func;
  static constexpr std::string_view label = symbol.view();
  static constexpr Notation notation = note;
  static constexpr int precedence = prec;

  // False lets fill_cache skip the store for every child.  True is the
  // conservative answer.
  static constexpr bool reads_primals = true;
  [[nodiscard]] static constexpr auto
  eval(const CExpression auto &lhs) noexcept {
    using VT = typename std::remove_cvref_t<decltype(lhs)>::value_type;
    return std::invoke(func{}, static_cast<VT>(lhs));
  }
};

template <Numeric T, typename func, CFixedString auto symbol,
          Notation note = Notation::Infix, int prec = precedence_atom>
  requires std::regular_invocable<func, const T &, const T &> &&
           std::convertible_to<std::invoke_result_t<func, const T &, const T &>,
                               T>
struct BinaryOp {
  using value_type = T;
  using func_type = func;
  static constexpr std::string_view label = symbol.view();
  static constexpr Notation notation = note;
  static constexpr int precedence = prec;

  static constexpr bool reads_primals = true;
  [[nodiscard]] static constexpr auto
  eval(const CExpression auto &lhs, const CExpression auto &rhs) noexcept {
    using LT = typename std::remove_cvref_t<decltype(lhs)>::value_type;
    using RT = typename std::remove_cvref_t<decltype(rhs)>::value_type;
    return std::invoke(func{}, static_cast<LT>(lhs), static_cast<RT>(rhs));
  }
};

template <Numeric T>
struct SumOp
    : BinaryOp<T, std::plus<void>, FixedString{"+"}, Notation::Infix, 10> {
  // A sum hands its adjoint to both operands unchanged, and a - b is a + (-b),
  // so this is the biggest source of skippable stores.
  static constexpr bool reads_primals = false;

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
struct MultiplyOp : BinaryOp<T, std::multiplies<void>, FixedString{"*"},
                             Notation::Infix, 20> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    auto lmul = lhs.derivative() * rhs;
    auto rmul = lhs * rhs.derivative();
    return std::move(lmul) + std::move(rmul);
  }
  // For c = a*b the differential is da*b + a*db, so the adjoint reaching `a`
  // multiplies on the LEFT of b and the one reaching `b` on the RIGHT of a.
  // Sided, not complete: a non-commutative scalar would also need a transpose.
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    return {adj * cache[cb[1]], cache[cb[0]] * adj};
  }
};

template <Numeric T>
struct NegateOp
    : UnaryOp<T, std::negate<void>, FixedString{"-"}, Notation::Prefix, 30> {
  static constexpr bool reads_primals = false;

  // Through operator-, so values.hpp's folding ladder sees it: -0 is a Lit.
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept {
    return -lhs.derivative();
  }
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &) noexcept {
    return {-adj};
  }
};

template <Numeric T>
struct DivideOp
    : BinaryOp<T, std::divides<void>, FixedString{"/"}, Notation::Infix, 20> {
  // `a / b` means a * b^-1 -- RIGHT division.  CFieldLike cannot distinguish
  // the two, so the side is a stated convention.  Under it
  // dc = da*b^-1 - a*b^-1*db*b^-1, which does not fold into one division by
  // b*b; the familiar quotient rule is that with the factors commuted, so both
  // spellings are kept and selected by if constexpr.
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    if constexpr (CCommutativeMultiply<T>) {
      auto num_l = lhs.derivative() * rhs;
      auto num_r = lhs * rhs.derivative();
      auto numerator = std::move(num_l) - std::move(num_r);
      auto denominator = rhs * rhs;
      return std::move(numerator) / std::move(denominator);
    } else {
      auto da = lhs.derivative() / rhs;
      auto db = ((lhs / rhs) * rhs.derivative()) / rhs;
      return std::move(da) - std::move(db);
    }
  }
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    const T b = cache[cb[1]];
    const T a = cache[cb[0]];
    if constexpr (CCommutativeMultiply<T>) {
      return {adj / b, -adj * a / (b * b)};
    } else {
      return {adj / b, -((a / b) * adj) / b};
    }
  }
};

// Binary math ops.  derivative() builds a forward symbolic tree; adjoints()
// pushes the partial adjoints.

template <Numeric T> struct PowOp;
template <Numeric T> struct Atan2Op;
template <Numeric T> struct HypotOp;
template <Numeric T>
  requires std::totally_ordered<T>
struct MaxOp;
template <Numeric T>
  requires std::totally_ordered<T>
struct MinOp;

namespace detail {
struct abs_impl {
  constexpr auto operator()(const Numeric auto &a) const noexcept {
    using std::abs;
    return abs(a);
  }
};
// sign(0) is 0: the abs/max/min subgradient convention every engine shares.
// a - a reaches only ±0 and NaN, giving 0 for the first and NaN for the
// second, so a NaN operand poisons the derivative instead of picking a side.
struct sign_impl {
  constexpr auto operator()(const Numeric auto &a) const noexcept {
    using T = std::remove_cvref_t<decltype(a)>;
    return a > T{} ? T{1} : a < T{} ? T{-1} : T{a - a};
  }
};
// `using std::` plus an unqualified call: ADL finds Dual's / TaylorDual's own
// overloads, std::* otherwise.
#define DDX_ADL_BINARY_IMPL(NAME, FN)                                          \
  struct NAME {                                                                \
    constexpr auto operator()(const Numeric auto &a,                           \
                              const Numeric auto &b) const noexcept {          \
      using std::FN;                                                           \
      return FN(a, b);                                                         \
    }                                                                          \
  };
DDX_ADL_BINARY_IMPL(pow_impl, pow)
DDX_ADL_BINARY_IMPL(atan2_impl, atan2)
DDX_ADL_BINARY_IMPL(hypot_impl, hypot)
#undef DDX_ADL_BINARY_IMPL
// A tie averages the operands: the value is unchanged, but a dual or Taylor
// operand carries the mean of the two subgradients.  An unordered pair -- at
// least one NaN -- returns (a-b)*0, NaN in every component from either side.
// Both choices are symmetric, the only convention stable under the
// commutative reordering the graph builder applies.  The tie is spelled
// a + (b-a)/2 so an equal pair at the top of the range cannot overflow.
struct max_impl {
  constexpr auto operator()(const Numeric auto &a,
                            const Numeric auto &b) const noexcept {
    using R = std::remove_cvref_t<decltype(a < b ? b : a)>;
    if (a == b) {
      return R{a + (b - a) / R{2}};
    }
    if (a < b) {
      return R{b};
    }
    if (b < a) {
      return R{a};
    }
    return R{(a - b) * R{}};
  }
};
struct min_impl {
  constexpr auto operator()(const Numeric auto &a,
                            const Numeric auto &b) const noexcept {
    using R = std::remove_cvref_t<decltype(b < a ? b : a)>;
    if (a == b) {
      return R{a + (b - a) / R{2}};
    }
    if (b < a) {
      return R{b};
    }
    if (a < b) {
      return R{a};
    }
    return R{(a - b) * R{}};
  }
};
} // namespace detail

// Generated from the registry: each op pulls value and derivative from its
// descriptor in unary_math.hpp.  Forward mode needs no member here -- it is the
// ordinary eval() sweep seeded with Dual, through the same descriptor.
#define DDX_UNARY_MATH_OP(FN, NAME, LABEL)                                     \
  template <Numeric T>                                                         \
    requires(!detail::needs_real_constants_v<detail::NAME##Fn<T>> ||           \
             std::constructible_from<T, double>)                               \
  struct NAME : UnaryOp<T, detail::NAME##Fn<T>, FixedString{LABEL}> {          \
    [[nodiscard]] static constexpr auto                                        \
    derivative(const CExpression auto &lhs) noexcept {                         \
      return detail::NAME##Fn<T>::deriv(lhs) * lhs.derivative();               \
    }                                                                          \
    template <std::size_t Base, std::size_t... CB>                             \
    static constexpr std::array<T, sizeof...(CB)>                              \
    adjoints(T adj, const auto &cache) noexcept {                              \
      constexpr std::size_t cb[]{CB...};                                       \
      return {adj * detail::NAME##Fn<T>::deriv(cache[cb[0]])};                 \
    }                                                                          \
  };

DDX_UNARY_MATH_TABLE(DDX_UNARY_MATH_OP)
#undef DDX_UNARY_MATH_OP

// Piecewise constant, so its derivative is identically zero.  Not public API:
// it exists so the derivative trees of abs, max and min stay finite at the
// kink, where a u/|u| quotient is 0/0.
template <Numeric T>
  requires std::totally_ordered<T>
struct SignOp : UnaryOp<T, detail::sign_impl, FixedString{"sign"}> {
  static constexpr bool reads_primals = false;
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &) noexcept {
    return Lit<T, 0>{};
  }
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T, const auto &) noexcept {
    return {T{}};
  }
};

// abs is not in the table: its derivative is a sign, not a function of the
// primal.  totally_ordered because the rule branches on a comparison.
template <Numeric T>
  requires std::totally_ordered<T>
struct AbsOp : UnaryOp<T, detail::abs_impl, FixedString{"abs"}> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs) noexcept {
    auto s = MonoExpression<SignOp<T>, std::decay_t<decltype(lhs)>>{lhs};
    return std::move(s) * lhs.derivative();
  }
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    return {adj * detail::sign_impl{}(cache[cb[0]])};
  }
};

// pow(a, b) = a^b.  d(a^b) = b' ln(a) a^b + b a^(b-1) a'.
template <Numeric T>
struct PowOp
    : BinaryOp<T, detail::pow_impl, FixedString{"pow"}, Notation::Function> {
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
struct Atan2Op : BinaryOp<T, detail::atan2_impl, FixedString{"atan2"},
                          Notation::Function> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs, const CExpression auto &rhs) noexcept;
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    using std::hypot;
    const T y = cache[cb[0]];
    const T x = cache[cb[1]];
    // Scaled by hypot rather than divided by x² + y², which overflows past
    // 1e154 and underflows the adjoints to zero.
    const T h = hypot(x, y);
    return {adj * (x / h) / h, -adj * (y / h) / h};
  }
};

// hypot(x, y) = sqrt(x² + y²).  d = (x x' + y y') / hypot.
template <Numeric T>
struct HypotOp : BinaryOp<T, detail::hypot_impl, FixedString{"hypot"},
                          Notation::Function> {
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
    // x / h first: x * adj can overflow where the quotient, in [-1, 1], never
    // does.
    return {adj * (x / h), adj * (y / h)};
  }
};

// totally_ordered because the rules below branch on a comparison.
template <Numeric T>
  requires std::totally_ordered<T>
struct MaxOp
    : BinaryOp<T, detail::max_impl, FixedString{"max"}, Notation::Function> {
  // max(a, b) = (a + b + |a - b|) / 2, so with s = sign(a - b),
  //   max' = (a' + b' + s*(a' - b')) / 2
  // Branch-free of necessity: a runtime conditional would have to choose
  // between two different *types*.  sign(0) = 0 makes a tie the mean of the
  // two branches, the shared convention.
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    const auto d = lhs - rhs;
    const auto s = MonoExpression<SignOp<T>, std::decay_t<decltype(d)>>{d};
    return (lhs.derivative() + rhs.derivative() +
            s * (lhs.derivative() - rhs.derivative())) /
           Lit<T, 2>{};
  }
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    using ret_t = std::array<T, sizeof...(CB)>;
    const T &a = cache[cb[0]];
    const T &b = cache[cb[1]];
    if (a == b) {
      const T half = adj / T{2};
      return ret_t{half, half};
    }
    if (a < b) {
      return ret_t{T{}, adj};
    }
    if (b < a) {
      return ret_t{adj, T{}};
    }
    const T poison = (a - b) * T{}; // unordered: NaN to both operands
    return ret_t{poison, poison};
  }
};

// totally_ordered because the rules below branch on a comparison.
template <Numeric T>
  requires std::totally_ordered<T>
struct MinOp
    : BinaryOp<T, detail::min_impl, FixedString{"min"}, Notation::Function> {
  // min(a, b) = (a + b - |a - b|) / 2; see MaxOp for why this is branch-free
  // and what a tie means.
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    const auto d = lhs - rhs;
    const auto s = MonoExpression<SignOp<T>, std::decay_t<decltype(d)>>{d};
    return (lhs.derivative() + rhs.derivative() -
            s * (lhs.derivative() - rhs.derivative())) /
           Lit<T, 2>{};
  }
  template <std::size_t Base, std::size_t... CB>
  static constexpr auto adjoints(T adj, const auto &cache) noexcept {
    using ret_t = std::array<T, sizeof...(CB)>;
    constexpr std::size_t cb[]{CB...};
    const T &a = cache[cb[0]];
    const T &b = cache[cb[1]];
    if (a == b) {
      const T half = adj / T{2};
      return ret_t{half, half};
    }
    if (b < a) {
      return ret_t{T{}, adj};
    }
    if (a < b) {
      return ret_t{adj, T{}};
    }
    const T poison = (a - b) * T{}; // unordered: NaN to both operands
    return ret_t{poison, poison};
  }
};

template <Numeric T>
constexpr auto PowOp<T>::derivative(const CExpression auto &lhs,
                                    const CExpression auto &rhs) noexcept {
  // Split form, not a^b * (b' ln a + b a'/a): the a'/a quotient is 0/0 at
  // a == 0 where a^(b-1) is exact, and with a constant exponent the b' term
  // folds away without leaving a log node behind.
  return rhs.derivative() * log(lhs) * pow(lhs, rhs) +
         rhs * pow(lhs, rhs - Lit<T, 1>{}) * lhs.derivative();
}

template <Numeric T>
constexpr auto Atan2Op<T>::derivative(const CExpression auto &lhs,
                                      const CExpression auto &rhs) noexcept {
  // Two divisions by hypot, never x² + y²: the square overflows past 1e154.
  return ((rhs * lhs.derivative() - lhs * rhs.derivative()) / hypot(lhs, rhs)) /
         hypot(lhs, rhs);
}

template <Numeric T>
constexpr auto HypotOp<T>::derivative(const CExpression auto &lhs,
                                      const CExpression auto &rhs) noexcept {
  return (lhs * lhs.derivative() + rhs * rhs.derivative()) / hypot(lhs, rhs);
}

} // namespace ddx::impl
