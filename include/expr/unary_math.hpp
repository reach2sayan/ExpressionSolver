#pragma once

#include "expr/expressions.hpp" // Numeric

#include <cmath>
#include <numbers>

namespace diff::impl::detail {

#define DIFF_UNARY_MATH_FNS                                                    \
  using std::sin, std::cos, std::tan, std::exp, std::log, std::log10,          \
      std::sqrt, std::cbrt, std::asin, std::acos, std::atan, std::sinh,        \
      std::cosh, std::tanh, std::asinh, std::acosh, std::atanh, std::erf
#define DIFF_UNARY_MATH_DESC(NAME, VAL, ...)                                   \
  template <Numeric T> struct NAME {                                           \
    constexpr auto operator()(const auto &u) const noexcept {                  \
      DIFF_UNARY_MATH_FNS;                                                     \
      return (VAL);                                                            \
    }                                                                          \
    static constexpr auto deriv(const auto &u) noexcept {                      \
      DIFF_UNARY_MATH_FNS;                                                     \
      return (__VA_ARGS__);                                                    \
    }                                                                          \
  };

#define DIFF_UNARY_MATH_DESC_FV(NAME, VAL, DERIV, ...)                         \
  template <Numeric T> struct NAME {                                           \
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

// clang-format off
DIFF_UNARY_MATH_DESC(SineOpFn,    sin(u), cos(u))
DIFF_UNARY_MATH_DESC(CosineOpFn,  cos(u), -sin(u))
DIFF_UNARY_MATH_DESC_FV(ExpOpFn,  exp(u), exp(u), fu)
// sec^2 = 1 + tan^2 — reassociates, so not bit-identical to 1/cos^2.
DIFF_UNARY_MATH_DESC_FV(TanOpFn,  tan(u),  T{1} / (cos(u) * cos(u)), T{1} + fu * fu)
DIFF_UNARY_MATH_DESC(LogOpFn,     log(u),  T{1} / u)
DIFF_UNARY_MATH_DESC_FV(SqrtOpFn, sqrt(u), T{1} / (T{2} * sqrt(u)), T{1} / (T{2} * fu))
DIFF_UNARY_MATH_DESC(AsinOpFn,    asin(u), T{1} / sqrt(T{1} - u * u))
DIFF_UNARY_MATH_DESC(AcosOpFn,    acos(u), T{-1} / sqrt(T{1} - u * u))
DIFF_UNARY_MATH_DESC(AtanOpFn,    atan(u), T{1} / (T{1} + u * u))
DIFF_UNARY_MATH_DESC(SinhOpFn,    sinh(u), cosh(u))
DIFF_UNARY_MATH_DESC(CoshOpFn,    cosh(u), sinh(u))
// sech^2 = 1 - tanh^2 — reassociates, so not bit-identical to 1/cosh^2.
DIFF_UNARY_MATH_DESC_FV(TanhOpFn, tanh(u),  T{1} / (cosh(u) * cosh(u)), T{1} - fu * fu)
DIFF_UNARY_MATH_DESC(Log10OpFn,   log10(u), T{1} / (u * static_cast<T>(std::numbers::ln10)))
DIFF_UNARY_MATH_DESC_FV(CbrtOpFn, cbrt(u),  T{1} / (T{3} * cbrt(u) * cbrt(u)), T{1} / (T{3} * fu * fu))
DIFF_UNARY_MATH_DESC(AsinhOpFn,   asinh(u), T{1} / sqrt(u * u + T{1}))
DIFF_UNARY_MATH_DESC(AcoshOpFn,   acosh(u), T{1} / sqrt(u * u - T{1}))
DIFF_UNARY_MATH_DESC(AtanhOpFn,   atanh(u), T{1} / (T{1} - u * u))
DIFF_UNARY_MATH_DESC(ErfOpFn,     erf(u),   static_cast<T>(2.0 * std::numbers::inv_sqrtpi) * exp(-(u * u)))
#undef DIFF_UNARY_MATH_DESC_FV
#undef DIFF_UNARY_MATH_DESC
#undef DIFF_UNARY_MATH_FNS
// clang-format on

// A descriptor's `deriv` needs strictly more of T than its value does (sin
// needs cos, asin needs sqrt, ...), and that is not expressible as a
// constraint: `deriv` returns auto, so a failure is outside the immediate
// context and hard-errors rather than SFINAE-ing.
//
// log10 and erf are the exception: they carry irrational constants and so need
// static_cast<T>(double), which constructible_from<T, int> does not give.  That
// one IS a constraint, applied by the generator below.
template <typename Fn> inline constexpr bool needs_real_constants_v = false;
template <Numeric T>
inline constexpr bool needs_real_constants_v<Log10OpFn<T>> = true;
template <Numeric T>
inline constexpr bool needs_real_constants_v<ErfOpFn<T>> = true;

template <typename Fn, Numeric T>
inline constexpr bool has_deriv_from_value_v =
    requires(const T &u, const T &fu) { Fn::deriv_from_value(u, fu); };

} // namespace diff::impl::detail

// `abs` is absent: its derivative is a sign, not a function of the primal.
// clang-format off
#define DIFF_UNARY_MATH_TABLE(X)                                               \
  X(sin,   SineOp,   "sin")                                                    \
  X(cos,   CosineOp, "cos")                                                    \
  X(exp,   ExpOp,    "exp")                                                    \
  X(tan,   TanOp,    "tan")                                                    \
  X(log,   LogOp,    "log")                                                    \
  X(log10, Log10Op,  "log10")                                                  \
  X(sqrt,  SqrtOp,   "sqrt")                                                   \
  X(cbrt,  CbrtOp,   "cbrt")                                                   \
  X(asin,  AsinOp,   "asin")                                                   \
  X(acos,  AcosOp,   "acos")                                                   \
  X(atan,  AtanOp,   "atan")                                                   \
  X(sinh,  SinhOp,   "sinh")                                                   \
  X(cosh,  CoshOp,   "cosh")                                                   \
  X(tanh,  TanhOp,   "tanh")                                                   \
  X(asinh, AsinhOp,  "asinh")                                                  \
  X(acosh, AcoshOp,  "acosh")                                                  \
  X(atanh, AtanhOp,  "atanh")                                                  \
  X(erf,   ErfOp,    "erf")
// clang-format on
