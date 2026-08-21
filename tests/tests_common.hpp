#pragma once
#include "ddx.hpp"

#include "drivers/hessian.hpp"
#include "drivers/numeric.hpp"
#include "drivers/seeded_energy.hpp"
#include "drivers/symbolic.hpp"
#include "dual/dual.hpp"
#include "dual/taylor_dual.hpp"
#include "expr/bound.hpp"
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
#include <tuple>

using namespace ddx::impl;
using namespace ddx::literals; // "x"_s

// Test-side conveniences over the drivers' two second-order result shapes.
// Both name the same members, so the only thing that differs is whether the
// buffers are owned (unique_ptr, runtime arity) or inline (array, static N).
template <typename T>
concept CHessianResult = requires(const T &H) {
  { H.value } -> std::convertible_to<double>;
  H.gradient;
  H.hessian;
  { T::arity } -> std::convertible_to<std::size_t>;
} || requires(const T &H) {
  { H.value } -> std::convertible_to<double>;
  { H.arity } -> std::convertible_to<std::size_t>;
};

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

[[nodiscard]] constexpr const double *hess_ptr(const auto &h) noexcept {
  return raw(deref(h).hessian);
}

[[nodiscard]] constexpr std::size_t hess_n(const auto &h) noexcept {
  return deref(h).arity;
}

[[nodiscard]] constexpr double hess_at(const auto &h, std::size_t i,
                                       std::size_t j) noexcept {
  return hess_ptr(h)[i * hess_n(h) + j]; // row-major, as the drivers document
}

[[nodiscard]] constexpr double grad_at(const auto &h, std::size_t i) noexcept {
  return grad_ptr(h)[i];
}

[[nodiscard]] constexpr double val_of(const auto &h) noexcept {
  return deref(h).value;
}
