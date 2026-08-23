#pragma once
// Shared by tests_printing.cpp and dual/tests_printing_dual.cpp.

#include "tests_common.hpp"

namespace {
constexpr auto px = ddx::var<"x">;
constexpr auto py = ddx::var<"y">;
constexpr auto pz = ddx::var<"z">;
} // namespace
