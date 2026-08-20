#pragma once

#include "dual/dual.hpp"
#include "dual/taylor_dual.hpp"
#include "dual/vector_dual.hpp"
#include "expr/bound.hpp"
#include "interop/eigen_interop.hpp"
#include "drivers/forward_driver.hpp"
#include "drivers/gradient.hpp"
#include "drivers/seeded_energy.hpp"
#include "drivers/vforward_driver.hpp"
#include "expr/equation.hpp"
#include "expr/format.hpp"
#include "expr/operations.hpp"
#include "expr/traits.hpp"
#include "expr/values.hpp"
#include "md/layouts.hpp"
#include "md/tensor.hpp"
#include "util/scope_guard.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <gtest/gtest.h>
#include <numbers>
#include <random>
#include <ranges>
#include <set>
#include <sstream>
#include <stdexcept>

using namespace diff;
using namespace diff::literals; // "x"_s

// ===========================================================================
// Math function tests — ported from autodiff's test suite
// Covers: tan, log, sqrt, abs, asin, acos, atan, sinh, cosh, tanh,
//         identity checks, and reverse/forward mode coverage for all.
// ===========================================================================
