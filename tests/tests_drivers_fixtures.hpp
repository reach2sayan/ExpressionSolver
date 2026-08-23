#pragma once
// Everything tests_drivers.cpp and dual/tests_drivers_dual.cpp both build on: the
// includes, the model expressions and the local helpers.  Split out so the
// two suites share one definition of each rather than a copy apiece.

#include "tests_common.hpp"

// ===========================================================================
// The public hessian() router on a raw callable.  It has one answer — the
// scalar O(m^2) probe driver — and this pins that across a spread of m, on an
// energy exercising +,-,*,/,log,exp and scalar*dual.
// ===========================================================================
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




// ===========================================================================
// Memory-ownership contract.
//
// Every result the library hands back is owned by the caller: the drivers
// return a HessianResult whose vectors are its own, and the symbolic API
// returns std::array by value.  The one raw pointer in the design points the
// other way — the driver lends the energy callable a `const Dof *` that is
// valid only for the call and carries no length.  These tests pin down both
// halves: the borrow is bounds-checked at the router, and the accessors that
// alias member storage stay usable on temporaries by copying out.
// ===========================================================================






// ===========================================================================
// Compile-time Hessian sparsity (coupling.hpp).  The pattern drives which
// entries the compressed driver writes, so a pattern that WRONGLY drops a pair
// silently zeroes a real Hessian entry — these pin the derivation down.
// ===========================================================================






// ===========================================================================
// Sparse Hessian.  The sparsity is a property of the expression TYPE, so the
// path writes only the entries the compile-time pattern predicts: a pattern
// that is wrong shows up here as a missing entry rather than as a wrong number.
// These compare it against the structure-blind dense driver.
//
// There is no linear-algebra library on this boundary.  What the library hands
// over is a compressed-column triple (outer, inner, values) plus the extent,
// and `densify` below is the whole of what a caller does with it -- it stands
// in for the one-line matrix map a client would write instead.
// ===========================================================================
namespace {
// CSC triple -> dense row-major, exactly as a consumer would read it.
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








// ===========================================================================
// Reusing driver overloads.  The point of these is that a caller sweeping many
// points allocates once rather than once per call; the tests here are about the
// buffers staying *correct* under that reuse, which is the part a benchmark
// cannot see.
// ===========================================================================

namespace {
// Deliberately not separable and not symmetric in the variables, so a stale
// cell from a previous call cannot coincidentally be the right answer.
auto reuse_energy = [](const auto *q) {
  using std::exp, std::log;
  return q[0] * q[0] * q[1] + exp(q[0] * q[2]) + q[1] * log(q[1] + 2.0);
};
} // namespace
