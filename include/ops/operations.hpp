#pragma once

#include "ops/adjoints.hpp"
#include "ops/numeric.hpp"
#include "ops/unary_math.hpp"
#include "util/fixed_string.hpp"
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <functional>
#include <numeric> // std::midpoint
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
  template <CExpression LT, CExpression RT>
  [[nodiscard]] static constexpr auto eval(const LT &lhs,
                                           const RT &rhs) noexcept {
    return std::invoke(func{}, static_cast<typename LT::value_type>(lhs),
                       static_cast<typename RT::value_type>(rhs));
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
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    return detail::SumOpFn<T>::adjoints(adj, cache[cb[0]], cache[cb[1]]);
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
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    return detail::MultiplyOpFn<T>::adjoints(adj, cache[cb[0]], cache[cb[1]]);
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
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    return detail::NegateOpFn<T>::adjoints(adj, cache[cb[0]]);
  }
};

// Both the rewriter and the printer ask this, so it lives with the operation
// rather than once in each.
namespace detail {
template <typename E> inline constexpr bool is_negation_expr_v = false;
template <Numeric T, CExpression C>
inline constexpr bool is_negation_expr_v<Expression<NegateOp<T>, C>> = true;
} // namespace detail

template <Numeric T>
struct DivideOp
    : BinaryOp<T, std::divides<void>, FixedString{"/"}, Notation::Infix, 20> {
  // `a / b` means a * b^-1 -- RIGHT division, which CFieldLike cannot state.
  // Under it dc = da*b^-1 - a*b^-1*db*b^-1, which does not fold into one
  // division by b*b; the familiar quotient rule is that with the factors
  // commuted, so both spellings are kept.
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
    return detail::DivideOpFn<T>::adjoints(adj, cache[cb[0]], cache[cb[1]]);
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
// The value form of the same rule adjoints.hpp differentiates through.
struct sign_impl {
  template <Numeric T> constexpr auto operator()(const T &a) const noexcept {
    return sign(a);
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
// std::midpoint for arithmetic R; Dual's and TaylorDual's own overloads for
// the rest, which is what max/min's tie branch needs.
DDX_ADL_BINARY_IMPL(midpoint_impl, midpoint)
#undef DDX_ADL_BINARY_IMPL
// A tie averages the operands and an unordered pair returns (a-b)*0, NaN from
// either side: both symmetric, and so stable under the graph builder's
// commutative reordering.  On a Dual a == b compares only the value level, so
// the derivative parts are averaged rather than picked from a side.  On an
// arithmetic scalar the only tie that is not simply `a` is the signed zeros,
// following IEEE 754-2019 maximum/minimum -- what llvm.maximum computes -- and
// spelled arithmetically: -0 + +0 is +0, so the sum of two zeros is their
// maximum and the negated sum of their negations their minimum.
template <bool IsMax> struct extremum_impl {
  constexpr auto operator()(const Numeric auto &a,
                            const Numeric auto &b) const noexcept {
    using R = std::remove_cvref_t<decltype(a < b ? b : a)>;
    if (a == b) {
      if constexpr (CArithmetic<R>) {
        if (a == R{}) {
          if constexpr (IsMax) {
            return R{a + b};
          } else {
            return R{-((-a) + (-b))};
          }
        }
        return R{a};
      } else {
        return R{midpoint_impl{}(R{a}, R{b})};
      }
    }
    if (a < b) {
      if constexpr (IsMax) {
        return R{b};
      } else {
        return R{a};
      }
    }
    if (b < a) {
      if constexpr (IsMax) {
        return R{a};
      } else {
        return R{b};
      }
    }
    return R{(a - b) * R{}};
  }
};
using max_impl = extremum_impl<true>;
using min_impl = extremum_impl<false>;
} // namespace detail

// Generated from the registry: each op pulls value and derivative from its
// descriptor.  Forward mode needs no member here -- it is the ordinary eval()
// sweep seeded with Dual, through the same descriptor.
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

// Not public API: it exists so the derivative trees of abs, max and min stay
// finite at the kink, where a u/|u| quotient is 0/0.
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
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    return detail::SignOpFn<T>::adjoints(adj, cache[cb[0]]);
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
    return detail::AbsOpFn<T>::adjoints(adj, cache[cb[0]]);
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
    return detail::PowOpFn<T>::adjoints(adj, cache[cb[0]], cache[cb[1]]);
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
    return detail::Atan2OpFn<T>::adjoints(adj, cache[cb[0]], cache[cb[1]]);
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
    return detail::HypotOpFn<T>::adjoints(adj, cache[cb[0]], cache[cb[1]]);
  }
};

// max(a, b) = (a + b + |a - b|) / 2, so with s = sign(a - b),
//   max' = (a' + b' + s*(a' - b')) / 2
// and min' the same with the sign subtracted.  Branch-free of necessity: a
// runtime conditional would have to choose between two different *types*.
// sign(0) = 0 makes a tie the mean, which adjoints() spells as a half to each
// side.  totally_ordered because both rules branch on a comparison.
template <Numeric T, bool IsMax, typename func, CFixedString auto symbol>
  requires std::totally_ordered<T>
struct ExtremumOp : BinaryOp<T, func, symbol, Notation::Function> {
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    const auto d = lhs - rhs;
    const auto s = MonoExpression<SignOp<T>, std::decay_t<decltype(d)>>{d};
    if constexpr (IsMax) {
      return (lhs.derivative() + rhs.derivative() +
              s * (lhs.derivative() - rhs.derivative())) /
             Lit<T, 2>{};
    } else {
      return (lhs.derivative() + rhs.derivative() -
              s * (lhs.derivative() - rhs.derivative())) /
             Lit<T, 2>{};
    }
  }

  // The one op that does not forward to adjoints.hpp: a compare is cheaper than
  // ExtremumOpFn's sign expansion and can halve a tie.  The graph has no
  // comparisons and must use the expansion.
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
    // The winner takes the whole adjoint, the loser none.
    if (IsMax ? a < b : b < a) {
      return ret_t{T{}, adj};
    }
    if (IsMax ? b < a : a < b) {
      return ret_t{adj, T{}};
    }
    const T poison = (a - b) * T{}; // unordered: NaN to both operands
    return ret_t{poison, poison};
  }
};

template <Numeric T>
  requires std::totally_ordered<T>
struct MaxOp : ExtremumOp<T, true, detail::max_impl, FixedString{"max"}> {};

template <Numeric T>
  requires std::totally_ordered<T>
struct MinOp : ExtremumOp<T, false, detail::min_impl, FixedString{"min"}> {};

template <Numeric T>
constexpr auto PowOp<T>::derivative(const CExpression auto &lhs,
                                    const CExpression auto &rhs) noexcept {
  // Split form, not a^b * (b' ln a + b a'/a): that quotient is 0/0 at a == 0
  // where a^(b-1) is exact, and a constant exponent folds away with no log.
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
