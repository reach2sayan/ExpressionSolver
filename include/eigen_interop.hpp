#pragma once

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

// Views, not conversions
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

  [[nodiscard]] Eigen::Map<const Eigen::SparseMatrix<double>>
  matrix() const noexcept {
    const auto n = static_cast<Eigen::Index>(kN);
    return {n,
            n,
            static_cast<Eigen::Index>(nnz),
            kLayout.outer.data(),
            kLayout.inner.data(),
            values_.data()};
  }

  [[nodiscard]] const std::vector<double> &values() const noexcept {
    return values_;
  }
};

// The sparse counterpart of hessian(graph, x).
template <CExpression Expr>
[[nodiscard]] SparseHessian<Expr> sparse_hessian(const Expr &expr,
                                                 std::span<const double> x) {
  return SparseHessian<Expr>{detail::hessian_values_sparse(expr, x)};
}

} // namespace diff
