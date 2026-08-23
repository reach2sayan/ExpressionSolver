#pragma once
// Shared by tests_core.cpp and dual/tests_core_dual.cpp.

#include "tests_common.hpp"

namespace {

using ddx::impl::Entry;
using ddx::impl::Record;

constexpr auto kRecordX = var<"x">;
constexpr auto kRecordN = var<"n", int>;

} // namespace
