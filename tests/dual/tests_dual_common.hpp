#pragma once
// What the forward-mode suite adds to tests_common.hpp.

#include "tests_common.hpp"

#include "dual/dual.hpp"
#include "dual/hessian.hpp"
#include "dual/numeric.hpp"
#include "dual/taylor_dual.hpp"

// The drivers' two second-order result shapes name the same members; they
// differ only in owned (unique_ptr) vs inline (array) buffers.
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
  return deref(h).hessian_view()[i, j];
}
