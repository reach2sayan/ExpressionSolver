#pragma once
// Shared by tests_drivers.cpp and dual/tests_drivers_dual.cpp.

#include "tests_common.hpp"

// Energy exercising +,-,*,/,log,exp and scalar*dual.
namespace {
template <Numeric T> T vf_sample(const T *y, std::size_t n) {
  using std::exp, std::log; // ADL selects the dual overloads
  T g = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    g = g + y[i] * log(y[i]);
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double c =
          0.1 * static_cast<double>(i + 1) - 0.05 * static_cast<double>(j);
      g = g + c * (y[i] * y[j]) / (T{1} + y[i]);
    }
  }
  g = g + exp(y[0] * y[n - 1]);
  return g;
}
} // namespace

namespace {
// CSC triple -> dense row-major, as a consumer would read it.
template <typename Sparse>
  requires requires {
    Sparse::rows;
    Sparse::outer();
    Sparse::inner();
  }
std::vector<double> densify(const Sparse &h) {
  std::vector<double> dense(Sparse::rows * Sparse::rows, 0.0);
  const auto outer = Sparse::outer();
  const auto inner = Sparse::inner();
  const auto values = h.values();
  for (std::size_t j = 0; j < Sparse::rows; ++j) {
    for (auto k = static_cast<std::size_t>(outer[j]);
         k < static_cast<std::size_t>(outer[j + 1]); ++k) {
      dense[static_cast<std::size_t>(inner[k]) * Sparse::rows + j] = values[k];
    }
  }
  return dense;
}
} // namespace

namespace {
// Not separable and not symmetric, so a stale cell cannot coincidentally be
// the right answer.
auto reuse_energy = [](const auto *q) {
  using std::exp, std::log;
  return q[0] * q[0] * q[1] + exp(q[0] * q[2]) + q[1] * log(q[1] + 2.0);
};
} // namespace
