#pragma once

#ifndef DIFF_USE_EIGEN
#error                                                                         \
    "eigen_interop.hpp needs Eigen: configure with -DDIFF_USE_EIGEN=ON (the default), or define DIFF_USE_EIGEN and put Eigen on the include path."
#endif

#include "drivers/coupling.hpp"
#include "drivers/forward_driver.hpp"
#include "drivers/vforward_driver.hpp"
#include "expr/expressions.hpp"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace diff {

using EigenDenseMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Views, not conversions.  Built from HessianResult::matrix() rather than from
// the raw buffer, so the row-major claim above is inherited from that view's
// md::layout_right instead of being asserted a second time here.
[[nodiscard]] inline Eigen::Map<const EigenDenseMatrix>
as_matrix(const HessianResult &h) noexcept {
  const auto m = h.matrix();
  static_assert(std::same_as<typename decltype(m)::layout_type,
                             md::layout_right>,
                "as_matrix maps Eigen::RowMajor onto the Hessian's own view; "
                "if that view stops being row-major this mapping is wrong");
  return {m.data_handle(), static_cast<Eigen::Index>(m.extent(0)),
          static_cast<Eigen::Index>(m.extent(1))};
}

[[nodiscard]] inline Eigen::Map<const Eigen::VectorXd>
as_vector(const HessianResult &h) noexcept {
  return {h.gradient.data(), static_cast<Eigen::Index>(h.gradient.size())};
}

// A sparse Hessian whose sparsity is a property of the expression TYPE.
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

  // The compressed buffer read as the dense matrix it stands for.  The
  // pattern is a compile-time property of Expr, so it is a layout mapping;
  // structural zeros share one sink cell and read back as exactly 0.0, which
  // is what they are.  Nothing here is Eigen — sparse_matrix_view lives in
  // coupling.hpp and works with DIFF_USE_EIGEN=OFF.
  [[nodiscard]] auto view() const noexcept {
    return sparse_matrix_view<E>(values_);
  }
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
  [[nodiscard]] std::span<const double> values() const noexcept {
    return std::span<const double>{values_}.first(nnz);
  }
};

// The sparse counterpart of hessian(graph, x).
template <CExpression Expr>
[[nodiscard]] SparseHessian<Expr> sparse_hessian(const Expr &expr,
                                                 std::span<const double> x) {
  return SparseHessian<Expr>{detail::hessian_values_sparse(expr, x)};
}

} // namespace diff
