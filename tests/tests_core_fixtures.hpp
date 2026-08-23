#pragma once
// Shared by tests_core.cpp and dual/tests_core_dual.cpp.

#include "tests_common.hpp"

namespace {

using ddx::impl::Map;
using ddx::impl::NamedValue;

constexpr auto kMapX = var<"x">;
constexpr auto kMapN = var<"n", int>;

} // namespace
