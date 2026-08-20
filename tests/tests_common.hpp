#pragma once
#include "ddx.hpp"

#include "dual/dual.hpp"
#include "dual/taylor_dual.hpp"
#include "expr/bound.hpp"
#include "drivers/numeric.hpp"
#include "drivers/symbolic.hpp"
#include "drivers/seeded_energy.hpp"
#include "drivers/hessian.hpp"
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
#include <tuple>
#include <numbers>
#include <random>
#include <ranges>
#include <set>
#include <sstream>
#include <stdexcept>

using namespace ddx::impl;
using namespace ddx::literals; // "x"_s

// Test-side conveniences for the drivers' plain-tuple results.  They live here,
// not in the library: they are the accessors the tuple return exists to avoid.
//
//   owning form: {value, gradient, hessian, extent}   (runtime arity)
//   static form: {value, gradient, hessian}           (extent is the array size)
template <CTupleLike T>
[[nodiscard]] constexpr const double *grad_ptr(const T &H) noexcept {
  if constexpr (std::tuple_size_v<std::remove_cvref_t<T>> == 4) {
    return std::get<1>(H).get();
  } else {
    return std::get<1>(H).data();
  }
}

template <CTupleLike T>
[[nodiscard]] constexpr const double *hess_ptr(const T &H) noexcept {
  if constexpr (std::tuple_size_v<std::remove_cvref_t<T>> == 4) {
    return std::get<2>(H).get();
  } else {
    return std::get<2>(H).data();
  }
}

template <CTupleLike T>
[[nodiscard]] constexpr std::size_t hess_n(const T &H) noexcept {
  if constexpr (std::tuple_size_v<std::remove_cvref_t<T>> == 4) {
    return std::get<3>(H);
  } else {
    return std::tuple_size_v<std::remove_cvref_t<decltype(std::get<1>(H))>>;
  }
}

template <CTupleLike T>
[[nodiscard]] constexpr double hess_at(const T &H, std::size_t i,
                                       std::size_t j) noexcept {
  return hess_ptr(H)[i * hess_n(H) + j]; // row-major, as the drivers document
}

template <CTupleLike T>
[[nodiscard]] constexpr double grad_at(const T &H, std::size_t i) noexcept {
  return grad_ptr(H)[i];
}

template <CTupleLike T> [[nodiscard]] constexpr double val_of(const T &H) noexcept {
  return std::get<0>(H);
}

