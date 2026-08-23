#pragma once
// Everything tests_math.cpp and dual/tests_math_dual.cpp both build on: the
// includes, the model expressions and the local helpers.  Split out so the
// two suites share one definition of each rather than a copy apiece.

#include "tests_common.hpp"

// ===========================================================================
// New math functions: log10, cbrt, asinh, acosh,
// atanh, erf (unary) and pow, atan2, hypot, min, max (binary), exercised
// across all three mechanisms — expression templates, lazy dual, Taylor.
// ===========================================================================

namespace {
constexpr double kLn10 = std::numbers::ln10;
constexpr double k2OverSqrtPi = 2.0 * std::numbers::inv_sqrtpi;
} // namespace

// ---------------------------------------------------------------------------
// operator[] — the subscript spelling of get/set.  It is an alias, so what
// these pin is that it stays one: same slot, same errors, same ownership
// rules, plus the assignment form that get/set cannot express.
// ---------------------------------------------------------------------------
