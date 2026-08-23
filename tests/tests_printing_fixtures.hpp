#pragma once
// Everything tests_printing.cpp and dual/tests_printing_dual.cpp both build on: the
// includes, the model expressions and the local helpers.  Split out so the
// two suites share one definition of each rather than a copy apiece.

#include "tests_common.hpp"

// ===========================================================================
// Expression printing.  The assertions are on std::format, since operator<< is
// a forward to it.
// ===========================================================================

namespace {
constexpr auto px = ddx::var<"x">;
constexpr auto py = ddx::var<"y">;
constexpr auto pz = ddx::var<"z">;
} // namespace
