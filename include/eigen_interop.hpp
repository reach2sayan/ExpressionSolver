#pragma once

// ===========================================================================
// The Eigen boundary — the ONLY header in this library that includes Eigen.
//
// The library computes derivatives; it does not do linear algebra.  What it
// owes a downstream consumer is a matrix in a form that consumer can factorize
// directly, which is what lives here: zero-copy views of the dense
// HessianResult, and a sparse Hessian whose structure comes from the
// compile-time coupling pass.
//
// Nothing else in include/ includes Eigen, and nothing here is reachable from
// the constexpr symbolic core.  That separation is structural, not incidental:
// Eigen types cannot be constant-evaluated, and the symbolic path
// (derivative_tensor, Equation, VectorDual, TaylorDual, coupling) is constexpr
// end to end — see ConstexprContract in tests.cpp, which fails to COMPILE if
// anything non-literal leaks inward.
//
// This header is deliberately NOT pulled in by expression_differentiator.hpp.
// Including it is opt-in, so a caller who only wants derivatives does not pay
// Eigen's header cost.
//
// Eigen itself is optional (CMake: -DDIFF_USE_EIGEN=OFF).  With it off the
// library is header-only with no third-party dependencies again, and this
// header is the only thing that goes away — everything it builds on
// (sparse_layout, sparse_slots, hessian_values_sparse) is Eigen-free and stays.
// ===========================================================================

#ifndef DIFF_USE_EIGEN
#error                                                                         \
    "eigen_interop.hpp needs Eigen: configure with -DDIFF_USE_EIGEN=ON (the default), or define DIFF_USE_EIGEN and put Eigen on the include path."
#endif

#include "coupling.hpp"
#include "expressions.hpp"
#include "forward_driver.hpp"
#include "vforward_driver.hpp"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace diff {

// Row-major, because that is how HessianResult stores it.  The Hessian is
// symmetric (every driver symmetrizes), so the storage order is immaterial for
// it — being explicit keeps these helpers correct if they are ever pointed at
// something that is not symmetric, such as a Jacobian.
using EigenDenseMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Views, not conversions: HessianResult already holds a flat row-major buffer,
// so these cost nothing and alias the result's storage.  Both dangle if the
// HessianResult they came from is destroyed.
[[nodiscard]] inline Eigen::Map<const EigenDenseMatrix>
as_matrix(const HessianResult &h) noexcept {
  const auto n = static_cast<Eigen::Index>(h.n());
  return {h.hessian.data(), n, n};
}

[[nodiscard]] inline Eigen::Map<const Eigen::VectorXd>
as_vector(const HessianResult &h) noexcept {
  return {h.gradient.data(), static_cast<Eigen::Index>(h.gradient.size())};
}

// A sparse Hessian whose sparsity is a property of the expression TYPE.
//
// The compressed index arrays are static constexpr — one copy in the binary,
// shared by every instance and every call — so constructing one allocates only
// the nnz values.  For the tridiagonal-plus-corner chain energy that is ~3n
// doubles instead of n^2.
//
// Because the structure is fixed for the type, a consumer may call
// analyzePattern() once and factorize() per point; that is the expensive half
// of a sparse direct solve, and here it is hoistable by construction.
template <CExpression Expr> class SparseHessian {
  using E = std::remove_cvref_t<Expr>;
  static constexpr std::size_t kN =
      mpl::mp_size(extract_symbols_from_expr_t<E>{});
  static constexpr auto kLayout = sparse_layout<E>();

  std::vector<double> values_;

public:
  static constexpr std::size_t rows = kN;
  static constexpr std::size_t nnz = decltype(kLayout)::nnz;

  explicit SparseHessian(std::vector<double> values) noexcept
      : values_(std::move(values)) {}

  // Valid while this object is alive.  Eigen::Map over a compressed matrix,
  // with the index arrays owned by the program image rather than by us.
  [[nodiscard]] Eigen::Map<const Eigen::SparseMatrix<double>>
  matrix() const noexcept {
    const auto n = static_cast<Eigen::Index>(kN);
    return {n, n, static_cast<Eigen::Index>(nnz), kLayout.outer.data(),
            kLayout.inner.data(), values_.data()};
  }

  [[nodiscard]] const std::vector<double> &values() const noexcept {
    return values_;
  }
};

// The sparse counterpart of hessian(graph, x).  Only an expression graph can
// take this path: the pattern comes from the type, and a runtime callable has
// no type to inspect.
template <CExpression Expr>
[[nodiscard]] SparseHessian<Expr> sparse_hessian(const Expr &expr,
                                                 std::span<const double> x) {
  return SparseHessian<Expr>{detail::hessian_values_sparse(expr, x)};
}

} // namespace diff
