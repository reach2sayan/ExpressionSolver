#pragma once

#include "expressions.hpp" // Numeric

#include <cmath>
#include <numbers>

namespace diff::detail {

#define DIFF_UNARY_MATH_FNS                                                    \
  using std::sin, std::cos, std::tan, std::exp, std::log, std::log10,          \
      std::sqrt, std::cbrt, std::asin, std::acos, std::atan, std::sinh,        \
      std::cosh, std::tanh, std::asinh, std::acosh, std::atanh, std::erf
#define DIFF_UNARY_MATH_DESC(NAME, VAL, ...)                                   \
  template <Numeric T> struct NAME {                                          \
    constexpr auto operator()(const auto &u) const noexcept {                  \
      DIFF_UNARY_MATH_FNS;                                                     \
      return (VAL);                                                            \
    }                                                                          \
    static constexpr auto deriv(const auto &u) noexcept {                      \
      DIFF_UNARY_MATH_FNS;                                                     \
      return (__VA_ARGS__);                                                    \
    }                                                                          \
  };

// Same, plus a derivative written in terms of the *already computed* value.
// The plain `deriv(u)` form re-derives the primal from the argument, so a node
// like exp evaluates std::exp twice — and because both functors recurse, that
// doubling squares with nesting depth (four std::exp calls for one exp of a
// dual2nd).  glibc's exp is not `const` without -fno-math-errno, so the
// compiler cannot fold the duplicate away.  A descriptor that can express its
// derivative from f(u) defines `deriv_from_value` and the Dual combiner picks
// it up; `deriv` stays for callers that only have the argument in hand.
#define DIFF_UNARY_MATH_DESC_FV(NAME, VAL, DERIV, ...)                         \
  template <Numeric T> struct NAME {                                          \
    constexpr auto operator()(const auto &u) const noexcept {                  \
      DIFF_UNARY_MATH_FNS;                                                     \
      return (VAL);                                                            \
    }                                                                          \
    static constexpr auto deriv(const auto &u) noexcept {                      \
      DIFF_UNARY_MATH_FNS;                                                     \
      return (DERIV);                                                          \
    }                                                                          \
    static constexpr auto deriv_from_value(const auto &u,                      \
                                           const auto &fu) noexcept {          \
      DIFF_UNARY_MATH_FNS;                                                     \
      (void)u;                                                                 \
      return (__VA_ARGS__);                                                    \
    }                                                                          \
  };

DIFF_UNARY_MATH_DESC(SineOpFn, sin(u), cos(u))
DIFF_UNARY_MATH_DESC(CosineOpFn, cos(u), -sin(u))
DIFF_UNARY_MATH_DESC_FV(ExpOpFn, exp(u), exp(u), fu)
// sec^2 = 1 + tan^2 — reassociates, so not bit-identical to 1/cos^2.
DIFF_UNARY_MATH_DESC_FV(TanOpFn, tan(u), T{1} / (cos(u) * cos(u)),
                        T{1} + fu * fu)
DIFF_UNARY_MATH_DESC(LogOpFn, log(u), T{1} / u)
DIFF_UNARY_MATH_DESC_FV(SqrtOpFn, sqrt(u), T{1} / (T{2} * sqrt(u)),
                        T{1} / (T{2} * fu))
DIFF_UNARY_MATH_DESC(AsinOpFn, asin(u), T{1} / sqrt(T{1} - u * u))
DIFF_UNARY_MATH_DESC(AcosOpFn, acos(u), T{-1} / sqrt(T{1} - u * u))
DIFF_UNARY_MATH_DESC(AtanOpFn, atan(u), T{1} / (T{1} + u * u))
DIFF_UNARY_MATH_DESC(SinhOpFn, sinh(u), cosh(u))
DIFF_UNARY_MATH_DESC(CoshOpFn, cosh(u), sinh(u))
// sech^2 = 1 - tanh^2 — reassociates, so not bit-identical to 1/cosh^2.
DIFF_UNARY_MATH_DESC_FV(TanhOpFn, tanh(u), T{1} / (cosh(u) * cosh(u)),
                        T{1} - fu * fu)
DIFF_UNARY_MATH_DESC(Log10OpFn, log10(u),
                     T{1} / (u * static_cast<T>(std::numbers::ln10)))
DIFF_UNARY_MATH_DESC_FV(CbrtOpFn, cbrt(u), T{1} / (T{3} * cbrt(u) * cbrt(u)),
                        T{1} / (T{3} * fu * fu))
DIFF_UNARY_MATH_DESC(AsinhOpFn, asinh(u), T{1} / sqrt(u * u + T{1}))
DIFF_UNARY_MATH_DESC(AcoshOpFn, acosh(u), T{1} / sqrt(u * u - T{1}))
DIFF_UNARY_MATH_DESC(AtanhOpFn, atanh(u), T{1} / (T{1} - u * u))
DIFF_UNARY_MATH_DESC(ErfOpFn, erf(u),
                     static_cast<T>(2.0 * std::numbers::inv_sqrtpi) *
                         exp(-(u * u)))
#undef DIFF_UNARY_MATH_DESC_FV
#undef DIFF_UNARY_MATH_DESC
#undef DIFF_UNARY_MATH_FNS

// True when a descriptor can build its derivative from the computed value.
// A requires-expression rather than a tag type, so a descriptor opts in just by
// defining the member.
template <typename Fn, Numeric T>
inline constexpr bool has_deriv_from_value_v =
    requires(const T &u, const T &fu) { Fn::deriv_from_value(u, fu); };

} // namespace diff::detail
