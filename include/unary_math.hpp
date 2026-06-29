#pragma once

#include <cmath>
#include <numbers>

namespace diff::detail {

#define DIFF_UNARY_MATH_FNS                                                    \
  using std::sin, std::cos, std::tan, std::exp, std::log, std::log10,          \
      std::sqrt, std::cbrt, std::asin, std::acos, std::atan, std::sinh,        \
      std::cosh, std::tanh, std::asinh, std::acosh, std::atanh, std::erf
#define DIFF_UNARY_MATH_DESC(NAME, VAL, ...)                                   \
  template <typename T> struct NAME {                                          \
    constexpr auto operator()(const auto &u) const noexcept {                  \
      DIFF_UNARY_MATH_FNS;                                                     \
      return (VAL);                                                            \
    }                                                                          \
    static constexpr auto deriv(const auto &u) noexcept {                      \
      DIFF_UNARY_MATH_FNS;                                                     \
      return (__VA_ARGS__);                                                    \
    }                                                                          \
  };

DIFF_UNARY_MATH_DESC(SineOpFn, sin(u), cos(u))
DIFF_UNARY_MATH_DESC(CosineOpFn, cos(u), -sin(u))
DIFF_UNARY_MATH_DESC(ExpOpFn, exp(u), exp(u))
DIFF_UNARY_MATH_DESC(TanOpFn, tan(u), T{1} / (cos(u) * cos(u)))
DIFF_UNARY_MATH_DESC(LogOpFn, log(u), T{1} / u)
DIFF_UNARY_MATH_DESC(SqrtOpFn, sqrt(u), T{1} / (T{2} * sqrt(u)))
DIFF_UNARY_MATH_DESC(AsinOpFn, asin(u), T{1} / sqrt(T{1} - u * u))
DIFF_UNARY_MATH_DESC(AcosOpFn, acos(u), T{-1} / sqrt(T{1} - u * u))
DIFF_UNARY_MATH_DESC(AtanOpFn, atan(u), T{1} / (T{1} + u * u))
DIFF_UNARY_MATH_DESC(SinhOpFn, sinh(u), cosh(u))
DIFF_UNARY_MATH_DESC(CoshOpFn, cosh(u), sinh(u))
DIFF_UNARY_MATH_DESC(TanhOpFn, tanh(u), T{1} / (cosh(u) * cosh(u)))
DIFF_UNARY_MATH_DESC(Log10OpFn, log10(u),
                     T{1} / (u * static_cast<T>(std::numbers::ln10)))
DIFF_UNARY_MATH_DESC(CbrtOpFn, cbrt(u), T{1} / (T{3} * cbrt(u) * cbrt(u)))
DIFF_UNARY_MATH_DESC(AsinhOpFn, asinh(u), T{1} / sqrt(u * u + T{1}))
DIFF_UNARY_MATH_DESC(AcoshOpFn, acosh(u), T{1} / sqrt(u * u - T{1}))
DIFF_UNARY_MATH_DESC(AtanhOpFn, atanh(u), T{1} / (T{1} - u * u))
DIFF_UNARY_MATH_DESC(ErfOpFn, erf(u),
                     static_cast<T>(2.0 * std::numbers::inv_sqrtpi) *
                         exp(-(u * u)))
#undef DIFF_UNARY_MATH_DESC
#undef DIFF_UNARY_MATH_FNS

} // namespace diff::detail
