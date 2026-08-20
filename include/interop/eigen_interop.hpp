#pragma once

#ifndef DIFF_USE_EIGEN
#error                                                                         \
    "eigen_interop.hpp needs Eigen: configure with -DDIFF_USE_EIGEN=ON (the default), or define DIFF_USE_EIGEN and put Eigen on the include path."
#endif

#include "drivers/coupling.hpp"
#include "drivers/forward_driver.hpp"
#include "drivers/hessian.hpp"
#include "expr/expressions.hpp"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <array>
#include <cstddef>
#include <span>

namespace diff {

using EigenDenseMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Non-owning views onto buffers the CALLER owns.
//
// A pointer and an extent rather than a result type, so these work with either
// shape the drivers return -- the unique_ptr one and the std::array one -- and
// so the ownership is unambiguous: the map borrows, the caller holds. The
// buffer must outlive the map, which is now visible in the signature instead of
// being a property of a struct that had to be kept alive.
//
// The drivers document their Hessian buffer as row-major, and Eigen::RowMajor
// here is that same statement; the two have to be changed together.
[[nodiscard]] inline Eigen::Map<const EigenDenseMatrix>
as_matrix(const double *const hess, const std::size_t n) noexcept {
  return {hess, static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n)};
}

[[nodiscard]] inline Eigen::Map<const Eigen::VectorXd>
as_vector(const double *const grad, const std::size_t n) noexcept {
  return {grad, static_cast<Eigen::Index>(n)};
}

// A sparse Hessian whose sparsity is a property of the expression TYPE.
template <CExpression Expr> class SparseHessian {
  using E = std::remove_cvref_t<Expr>;
  static constexpr std::size_t kN =
      detail::expr_arity_v<E>;
  static constexpr auto kLayout = sparse_layout<E>();
  static constexpr std::size_t kNnz = decltype(kLayout)::nnz;

  std::array<double, kNnz + 1> values_;

public:
  static constexpr std::size_t rows = kN;
  static constexpr std::size_t nnz = kNnz;

  explicit SparseHessian(std::array<double, kNnz + 1> values) noexcept
      : values_(values) {}

  [[nodiscard]] Eigen::Map<const Eigen::SparseMatrix<double>>
  matrix() const & noexcept {
    const auto n = static_cast<Eigen::Index>(kN);
    return {n,
            n,
            static_cast<Eigen::Index>(nnz),
            kLayout.outer.data(),
            kLayout.inner.data(),
            values_.data()};
  }
  auto matrix() const && = delete;

  // The compressed buffer read as the dense matrix it stands for.
  [[nodiscard]] auto view() const & noexcept {
    return sparse_matrix_view<E>(values_);
  }
  auto view() const && = delete;
  [[nodiscard]] double operator[](std::size_t i, std::size_t j) const noexcept {
    return view()[i, j];
  }
  // Whether (i, j) is in the pattern at all, as opposed to reading 0.0 because
  // the structure says it cannot be anything else.
  [[nodiscard]] static constexpr bool structural(std::size_t i,
                                                 std::size_t j) noexcept {
    return typename layout_sparse_pattern<E>::template mapping<
               md::extents<std::size_t, kN, kN>>{}
        .contains(i, j);
  }

  // The nnz values, without the sink cell the view appends.
  [[nodiscard]] std::span<const double> values() const & noexcept {
    return std::span<const double>{values_}.first(nnz);
  }
  auto values() const && = delete;
};

// The sparse counterpart of hessian(graph, x).
template <CExpression Expr>
[[nodiscard]] SparseHessian<Expr> sparse_hessian(const Expr &expr,
                                                 std::span<const double> x) {
  return SparseHessian<Expr>{detail::hessian_values_sparse(expr, x)};
}

} // namespace diff
