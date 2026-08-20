#pragma once

#include "expr/expressions.hpp"
#include "expr/unary_math.hpp"
#include <array>
#include <concepts>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string_view>
#include <utility>

namespace diff {

template <typename F, typename T>
concept CUnaryOp = std::regular_invocable<F, const T &> &&
                   std::convertible_to<std::invoke_result_t<F, const T &>, T>;

template <typename F, typename T>
concept CBinaryOp =
    std::regular_invocable<F, const T &, const T &> &&
    std::convertible_to<std::invoke_result_t<F, const T &, const T &>, T>;

// How an op is written down.  This is data on the op rather than code in it:
// the printer in expr/format.hpp is the only thing that reads it.
//
//   Prefix    -a          the operator leads its single operand
//   Infix     a + b       the operator sits between its two operands
//   Function  pow(a, b)   name and parentheses, any arity
enum class Notation : std::uint8_t { Prefix, Infix, Function };

// Binding strength; higher binds tighter.  A function call brings its own
// parentheses and a leaf has nothing to split, so both are atomic — there is no
// context in which either needs wrapping.
inline constexpr int precedence_atom = 100;

template <Numeric T, CUnaryOp<T> func, CFixedString auto symbol,
          Notation note = Notation::Function, int prec = precedence_atom>
struct UnaryOp {
  using value_type = T;
  using func_type = func;
  static constexpr std::string_view label = symbol.view();
  static constexpr Notation notation = note;
  static constexpr int precedence = prec;

  // Whether this op's adjoint rule reads the primal values of its operands.
  // The reverse sweep only has to *store* a node's value when some parent will
  // read it back -- no rule ever reads its own slot -- so an op that answers
  // false lets fill_cache skip the store for every child it has.  True by
  // default, so an op that forgets to say otherwise keeps its operands, which
  // is the conservative answer rather than the wrong one.
  static constexpr bool reads_primals = true;
  [[nodiscard]] static constexpr auto
  eval(const CExpression auto &lhs) noexcept {
    using VT = typename std::remove_cvref_t<decltype(lhs)>::value_type;
    return std::invoke(func{}, static_cast<VT>(lhs));
  }
};

template <Numeric T, CBinaryOp<T> func, CFixedString auto symbol,
          Notation note = Notation::Infix, int prec = precedence_atom>
struct BinaryOp {
  using value_type = T;
  using func_type = func;
  static constexpr std::string_view label = symbol.view();
  static constexpr Notation notation = note;
  static constexpr int precedence = prec;

  // Whether this op's adjoint rule reads the primal values of its operands.
  // The reverse sweep only has to *store* a node's value when some parent will
  // read it back -- no rule ever reads its own slot -- so an op that answers
  // false lets fill_cache skip the store for every child it has.  True by
  // default, so an op that forgets to say otherwise keeps its operands, which
  // is the conservative answer rather than the wrong one.
  static constexpr bool reads_primals = true;
  [[nodiscard]] static constexpr auto
  eval(const CExpression auto &lhs, const CExpression auto &rhs) noexcept {
    using LT = typename std::remove_cvref_t<decltype(lhs)>::value_type;
    using RT = typename std::remove_cvref_t<decltype(rhs)>::value_type;
    return std::invoke(func{}, static_cast<LT>(lhs), static_cast<RT>(rhs));
  }
};

template <Numeric T>
struct SumOp : BinaryOp<T, std::plus<void>, FixedString{"+"}, Notation::Infix,
                        10> {
  // adjoints() hands its own adjoint to both operands unchanged, so a sum
  // never looks at what it added.  Addition is the most common node in the
  // library -- a - b is spelled a + (-b) -- so this is the single biggest
  // source of stores the forward sweep can skip.
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
  // Sided to match the product rule above: for c = a*b the differential is
  // da*b + a*db, so the adjoint reaching `a` multiplies on the LEFT of b and
  // the one reaching `b` multiplies on the RIGHT of a.  Writing both as
  // `adj * ...` was only correct because every scalar shipped today commutes;
  // the symbolic path never had the bug, since derivative() already spells
  // lhs.derivative() * rhs and lhs * rhs.derivative().
  //
  // Exact wherever multiplication commutes, so this is bit-identical for
  // double, Dual and TaylorDual.  It is the correctly *sided* rule
  // rather than a complete one for a genuinely non-commutative scalar: a matrix
  // would additionally need a transpose here, and Numeric provides none.
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    return {adj * cache[cb[1]], cache[cb[0]] * adj};
  }
};

template <Numeric T>
struct NegateOp : UnaryOp<T, std::negate<void>, FixedString{"-"},
                          Notation::Prefix, 30> {
  // {-adj}: the operand's value is never consulted either.
  static constexpr bool reads_primals = false;

  // Through operator-, not MonoExpression directly, so that the folding ladder
  // in values.hpp gets to see it: -0 is a Lit, not a node wrapping one.
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
struct DivideOp : BinaryOp<T, std::divides<void>, FixedString{"/"},
                           Notation::Infix, 20> {
  // `a / b` is read as a * b^-1 -- RIGHT division.  A ring with inverses admits
  // two divisions and nothing in CFieldLike distinguishes them, so the side is
  // chosen and stated here rather than left implied: a scalar whose operator/
  // means b^-1 * a gets the wrong rule from both functions below, and cannot be
  // detected.  (Mat2 in the test suite follows this convention.)
  //
  // Under it, c = a*b^-1 differentiates as
  //
  //     dc = da*b^-1 - a*b^-1*db*b^-1
  //
  // so the b term threads db BETWEEN a/b and b^-1 and will not fold into a
  // single division by b*b.  The familiar quotient rule (a'b - ab')/b^2 is that
  // expression with the factors commuted, which is why both spellings appear
  // below: they agree exactly wherever multiplication commutes.
  //
  // Both spellings are kept, selected by if constexpr -- so the choice is made
  // at compile time and each instantiation emits exactly one of them.  There is
  // no runtime branch and no dead code, which is what makes keeping the fast
  // path free rather than a tradeoff against the general one.
  //
  // It earns its place: the sided form costs a second division in a hot path,
  // and a/b/b does not round like a/(b*b).  Taking it unconditionally would
  // slow down and perturb every scalar that ships to fix a case none of them
  // has.  Correctness is not what is being traded here -- the two agree
  // exactly wherever multiplication commutes, which the guard is precisely the
  // condition for.
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
    // The a-side is right division either way: adj * b^-1.
    if constexpr (CCommutativeMultiply<T>) {
      return {adj / b, -adj * a / (b * b)};
    } else {
      return {adj / b, -((a / b) * adj) / b};
    }
  }
};

// --- binary math ops (function-style: pow / atan2 / hypot / min / max) ------
// derivative() builds a forward symbolic derivative tree; backward() pushes the
// partial adjoints for reverse mode.  Op-level impls live in detail:: below.

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
// Binary impls.  ADL resolves pow/atan2/hypot for diff::Dual, TaylorDual and
// TaylorDual (each defines its own) when T is one of those, and to std::* for
// plain arithmetic T.  The `using std::` + unqualified call is the whole
// mechanism, so it is written once.
#define DIFF_ADL_BINARY_IMPL(NAME, FN)                                         \
  struct NAME {                                                                \
    constexpr auto operator()(const Numeric auto &a,                           \
                              const Numeric auto &b) const noexcept {          \
      using std::FN;                                                           \
      return FN(a, b);                                                         \
    }                                                                          \
  };
DIFF_ADL_BINARY_IMPL(pow_impl, pow)
DIFF_ADL_BINARY_IMPL(atan2_impl, atan2)
DIFF_ADL_BINARY_IMPL(hypot_impl, hypot)
#undef DIFF_ADL_BINARY_IMPL
struct max_impl {
  constexpr auto operator()(const Numeric auto &a,
                            const Numeric auto &b) const noexcept {
    return a < b ? b : a;
  }
};
struct min_impl {
  constexpr auto operator()(const Numeric auto &a,
                            const Numeric auto &b) const noexcept {
    return b < a ? b : a;
  }
};
} // namespace detail

// Each unary math op pulls its value + derivative from the shared descriptor
// detail::<Name>Fn (unary_math.hpp).  derivative() builds f'(lhs)·lhs';
// adjoints pushes adj·f'(value).  Forward mode needs no member here: it is the
// ordinary eval() sweep seeded with Dual, and the same descriptor drives
// unary_dual_combine in dual.hpp.
//
// Generated from the registry in unary_math.hpp -- the name list is not
// repeated here.
#define DIFF_UNARY_MATH_OP(FN, NAME, LABEL)                                    \
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

DIFF_UNARY_MATH_TABLE(DIFF_UNARY_MATH_OP)
#undef DIFF_UNARY_MATH_OP

// abs is not in the table: its derivative is a sign, not a function of the
// primal, so it has no descriptor to generate from.
// Ordering is a real requirement, not an implied one: the rules below branch on
// operand comparison, and CFieldLike asks only for closure under the five
// operators.  Saying so here fails a bad scalar at the point of use with a
// readable message, instead of deep inside a sweep.
template <Numeric T>
  requires std::totally_ordered<T>
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

// pow(a, b) = a^b.  d(a^b) = a^b * (b' ln a + b a'/a).
template <Numeric T>
struct PowOp : BinaryOp<T, detail::pow_impl, FixedString{"pow"}, Notation::Function> {
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
struct Atan2Op : BinaryOp<T, detail::atan2_impl, FixedString{"atan2"}, Notation::Function> {
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
struct HypotOp : BinaryOp<T, detail::hypot_impl, FixedString{"hypot"}, Notation::Function> {
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

// Ordering is a real requirement, not an implied one: the rules below branch on
// operand comparison, and CFieldLike asks only for closure under the five
// operators.  Saying so here fails a bad scalar at the point of use with a
// readable message, instead of deep inside a sweep.
template <Numeric T>
  requires std::totally_ordered<T>
struct MaxOp : BinaryOp<T, detail::max_impl, FixedString{"max"}, Notation::Function> {
  // max(a, b) = (a + b + |a - b|) / 2, so with s = (a - b)/|a - b|,
  //   max' = (a' + b' + s*(a' - b')) / 2
  // which is a' where a > b and b' where a < b.  It has to be branch-free: a
  // runtime conditional would have to choose between lhs.derivative() and
  // rhs.derivative(), and those are two different *types*.  NaN at a == b,
  // where max is genuinely not differentiable -- the same convention abs
  // already follows at 0.
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    const auto d = lhs - rhs;
    const auto s = d / MonoExpression<AbsOp<T>, std::decay_t<decltype(d)>>{d};
    return (lhs.derivative() + rhs.derivative() +
            s * (lhs.derivative() - rhs.derivative())) /
           Lit<T, 2>{};
  }
  template <std::size_t Base, std::size_t... CB>
  static constexpr std::array<T, sizeof...(CB)>
  adjoints(T adj, const auto &cache) noexcept {
    constexpr std::size_t cb[]{CB...};
    using ret_t = std::array<T, sizeof...(CB)>;
    return (cache[cb[0]] < cache[cb[1]]) ? ret_t{T{}, adj} : ret_t{adj, T{}};
  }
};

// Ordering is a real requirement, not an implied one: the rules below branch on
// operand comparison, and CFieldLike asks only for closure under the five
// operators.  Saying so here fails a bad scalar at the point of use with a
// readable message, instead of deep inside a sweep.
template <Numeric T>
  requires std::totally_ordered<T>
struct MinOp : BinaryOp<T, detail::min_impl, FixedString{"min"}, Notation::Function> {
  // min(a, b) = (a + b - |a - b|) / 2; see MaxOp for why this is branch-free.
  [[nodiscard]] static constexpr auto
  derivative(const CExpression auto &lhs,
             const CExpression auto &rhs) noexcept {
    const auto d = lhs - rhs;
    const auto s = d / MonoExpression<AbsOp<T>, std::decay_t<decltype(d)>>{d};
    return (lhs.derivative() + rhs.derivative() -
            s * (lhs.derivative() - rhs.derivative())) /
           Lit<T, 2>{};
  }
  template <std::size_t Base, std::size_t... CB>
  static constexpr auto adjoints(T adj, const auto &cache) noexcept {
    using ret_t = std::array<T, sizeof...(CB)>;
    constexpr std::size_t cb[]{CB...};
    return cache[cb[1]] < cache[cb[0]] ? ret_t{T{}, adj} : ret_t{adj, T{}};
  }
};

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
