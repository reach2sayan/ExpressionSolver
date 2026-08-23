#pragma once
// Dual-free by construction: this is what the suite that builds without
// forward mode includes.  The second-order helpers live in
// dual/tests_dual_common.hpp beside the tests that need them.
#include "ddx.hpp"

#include "symbolic/seeded_energy.hpp"
#include "symbolic/sweep.hpp"
#include "symbolic/bound.hpp"
#include "symbolic/equation.hpp"
#include "symbolic/format.hpp"
#include "ops/operations.hpp"
#include "symbolic/traits.hpp"
#include "symbolic/values.hpp"
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
#include <tuple>

using namespace ddx::impl;
using namespace ddx::literals; // "x"_s

// The drivers answer with result<T> wherever the point is a span or a range,
// so every accessor below takes either.  A test that is not about the error
// path reads exactly as it did; one that is checks .error().code directly.
template <typename T>
[[nodiscard]] constexpr const T &deref(const T &x) noexcept {
  return x;
}
template <typename T>
[[nodiscard]] constexpr const T &deref(const ddx::result<T> &r) noexcept {
  return *r;
}

[[nodiscard]] constexpr const double *raw(const double *p) noexcept {
  return p;
}
template <std::ranges::contiguous_range R>
[[nodiscard]] constexpr const double *raw(const R &r) noexcept {
  return std::ranges::data(r);
}
[[nodiscard]] inline const double *
raw(const std::unique_ptr<double[]> &p) noexcept {
  return p.get();
}

[[nodiscard]] constexpr const double *grad_ptr(const auto &h) noexcept {
  return raw(deref(h).gradient);
}

[[nodiscard]] constexpr double grad_at(const auto &h, std::size_t i) noexcept {
  return grad_ptr(h)[i];
}

[[nodiscard]] constexpr double val_of(const auto &h) noexcept {
  return deref(h).value;
}
