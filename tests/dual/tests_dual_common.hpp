#pragma once
// What the forward-mode suite adds to tests_common.hpp: the dual scalars
// themselves, the drivers only they can answer, and the two shapes a
// second-order result comes back in.

#include "tests_common.hpp"

#include "dual/dual.hpp"
#include "dual/hessian.hpp"
#include "dual/numeric.hpp"
#include "dual/taylor_dual.hpp"

// Test-side conveniences over the drivers' two second-order result shapes.
// Both name the same members, so the only thing that differs is whether the
// buffers are owned (unique_ptr, runtime arity) or inline (array, static N).
template <typename T>
concept CHessianResult = requires(const T &H) {
  { H.value } -> std::convertible_to<double>;
  H.jacobian;
  H.hessian;
  { T::arity } -> std::convertible_to<std::size_t>;
} || requires(const T &H) {
  { H.value } -> std::convertible_to<double>;
  { H.arity } -> std::convertible_to<std::size_t>;
};

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
