#pragma once

#include "expr/expressions.hpp"
#include "expr/operations.hpp"
#include "expr/traits.hpp"
#include "md/md.hpp"
#include "util/mpl.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <functional>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>
#include <tuple>
#include <type_traits>
#include <utility>

// Hessian sparsity, derived from the expression type: d2f/dxi dxj can only be
// nonzero if xi and xj meet under something with curvature.
namespace diff::impl {

template <COperation Op> inline constexpr bool is_linear_op_v = false;
template <Numeric T> inline constexpr bool is_linear_op_v<SumOp<T>> = true;
template <Numeric T> inline constexpr bool is_linear_op_v<NegateOp<T>> = true;

// a*b: curvature across the operands only.
template <COperation Op> inline constexpr bool is_product_op_v = false;
template <Numeric T> inline constexpr bool is_product_op_v<MultiplyOp<T>> = true;

// a/b: across the operands, plus within the denominator; linear in the numerator.
template <COperation Op> inline constexpr bool is_quotient_op_v = false;
template <Numeric T> inline constexpr bool is_quotient_op_v<DivideOp<T>> = true;

template <std::size_t N> using symbol_set = std::bitset<N>;
template <std::size_t N> using coupling_rows = std::array<symbol_set<N>, N>;

namespace detail {

template <std::size_t N> struct coupling_info {
  symbol_set<N> symbols{};  // symbols occurring in this subtree
  coupling_rows<N> rows{};  // rows[i][j]: (i,j) may be nonzero
};

// Both ways: the pattern is symmetric.
template <std::size_t N>
consteval void couple(coupling_rows<N> &rows, const symbol_set<N> &a,
                      const symbol_set<N> &b) noexcept {

  // enumerate hands out a signed difference_type; bitset indexes by size_t.
  for (auto &&[n, row] : rows | std::views::enumerate) {
    const auto i = static_cast<std::size_t>(n);
    if (a[i]) {
      row |= b;
    }
    if (b[i]) {
      row |= a;
    }
  }
}

// Propagate (symbols, coupled pairs) upward.  Recursion is on the type: a
// storing node is not default-constructible.
template <CExpression E, CSymbolList Syms, std::size_t N>
consteval coupling_info<N> coupling_of() noexcept {
  using U = std::remove_cvref_t<E>;
  coupling_info<N> info{};

  if constexpr (is_variable_v<U>) {
    // A frozen variable differentiates to zero.
    if constexpr (!U::frozen) {
      info.symbols.set(mpl::mp_find<U::label>(Syms{}));
    }
  } else if constexpr (CExpressionNode<U>) {
    using Kids = typename U::children_t;
    constexpr std::size_t K = std::tuple_size_v<Kids>;

    const std::array kids =
        [&]<std::size_t... I>(std::index_sequence<I...>) {
          return std::array{coupling_of<std::tuple_element_t<I, Kids>, Syms,
                                        N>()...};
        }(std::make_index_sequence<K>{});

    for (const auto &kid : kids) {
      info.symbols |= kid.symbols;
      std::ranges::transform(info.rows, kid.rows, info.rows.begin(),
                             std::bit_or{});
    }

    using Op = typename U::op_type;
    if constexpr (is_linear_op_v<Op>) {
      // No curvature of its own.
    } else if constexpr (is_product_op_v<Op> && K == 2) {
      couple<N>(info.rows, kids[0].symbols, kids[1].symbols);
    } else if constexpr (is_quotient_op_v<Op> && K == 2) {
      couple<N>(info.rows, kids[0].symbols, kids[1].symbols);
      couple<N>(info.rows, kids[1].symbols, kids[1].symbols);
    } else {
      // Anything else is assumed to couple its whole support with itself.
      couple<N>(info.rows, info.symbols, info.symbols);
    }
  }
  return info;
}

} // namespace detail

// rows[i][j]: d2f/dxi dxj may be nonzero.  A conservative superset.
template <CExpression Expr,
          std::size_t N = detail::expr_arity_v<Expr>>
consteval coupling_rows<N> hessian_pattern() noexcept {
  using Syms = detail::expr_symbols_t<std::remove_cvref_t<Expr>>;
  return detail::coupling_of<std::remove_cvref_t<Expr>, Syms, N>().rows;
}

// Instantiated once per Expr, so the walk above runs once for all the tables.
template <CExpression Expr>
inline constexpr auto hessian_pattern_v = hessian_pattern<Expr>();

template <std::size_t N> struct column_coloring {
  std::array<std::size_t, N> color{};
  std::size_t count = 0;
};

// Two columns may share a seed only if no row has a nonzero in both -- the
// distance-1 colouring of the column intersection graph that CPR uses.  Greedy.
template <std::size_t N>
consteval column_coloring<N>
color_columns(const coupling_rows<N> &rows) noexcept {
  column_coloring<N> coloring{};
  for (const auto j : std::views::iota(0uz, N)) {
    std::bitset<N> taken{};
    for (const auto k : std::views::iota(0uz, j)) {
      if ((rows[j] & rows[k]).any()) {
        taken.set(coloring.color[k]);
      }
    }
    // At most j colours are taken, so the search always lands inside [0, N).
    const auto pick = *std::ranges::find_if(
        std::views::iota(0uz, N), [&taken](std::size_t c) { return !taken[c]; });
    coloring.color[j] = pick;
    coloring.count = std::max(coloring.count, pick + 1);
  }
  return coloring;
}

// No column of this colour writes into this row.
inline constexpr std::size_t no_column = static_cast<std::size_t>(-1);

// map[colour][row]: where that row's sweep result belongs.
template <std::size_t N>
using scatter_map = std::array<std::array<std::size_t, N>, N>;

template <std::size_t N> consteval scatter_map<N> unmapped() noexcept {
  scatter_map<N> map{};
  for (auto &row : map) {
    row.fill(no_column);
  }
  return map;
}

// The colouring guarantees at most one column per (colour, row), so the driver
// scatters in one pass over rows and the sweep loop carries no search.
template <std::size_t N>
consteval scatter_map<N>
scatter_targets(const coupling_rows<N> &rows,
                const column_coloring<N> &coloring) noexcept {
  auto targets = unmapped<N>();
  for (const auto i : std::views::iota(0uz, N)) {
    for (const auto j : std::views::iota(0uz, N)) {
      if (rows[i][j]) {
        targets[coloring.color[j]][i] = j;
      }
    }
  }
  return targets;
}

// Compressed-column layout for the sparsity pattern.
template <CExpression Expr,
          std::size_t N = detail::expr_arity_v<Expr>>
consteval std::size_t hessian_nnz() noexcept {
  return std::ranges::fold_left(
      hessian_pattern_v<Expr>, 0uz,
      [](std::size_t nnz, const symbol_set<N> &row) {
        return nnz + row.count();
      });
}

template <std::size_t N, std::size_t NNZ> struct sparse_layout_t {
  static constexpr std::size_t rows = N;
  static constexpr std::size_t nnz = NNZ;
  std::array<int, N + 1> outer{}; // column j occupies [outer[j], outer[j+1])
  std::array<int, NNZ> inner{};   // row index of each nonzero, ascending
};

// Sorted CSC.  The pattern is symmetric, so by column and by row agree.
template <CExpression Expr,
          std::size_t N = detail::expr_arity_v<Expr>,
          std::size_t NNZ = hessian_nnz<Expr>()>
consteval sparse_layout_t<N, NNZ> sparse_layout() noexcept {
  constexpr auto &pattern = hessian_pattern_v<Expr>;
  sparse_layout_t<N, NNZ> layout{};

  // Exclusive prefix sum over N + 1 inputs, so it emits N + 1 outputs.
  std::array<int, N + 1> counts{};
  std::ranges::transform(pattern, counts.begin(),
                         [](const symbol_set<N> &column) {
                           return static_cast<int>(column.count());
                         });
  std::exclusive_scan(counts.begin(), counts.end(), layout.outer.begin(), 0);

  auto out = layout.inner.begin();
  for (const symbol_set<N> &column : pattern) {
    auto occupied = std::views::iota(0uz, N) |
                    std::views::filter(
                        [&column](std::size_t i) { return column[i]; });
    for (const auto i : occupied) {
      *out++ = static_cast<int>(i);
    }
  }
  return layout;
}

namespace detail {

// For each column j, every stored entry k and the row i it sits at.  Both
// tables below are that walk, differing only in where they write (i, j) -> k.
struct compressed_entry {
  std::size_t column, row, slot;
};
template <std::size_t N, std::size_t NNZ>
consteval auto compressed_entries(const sparse_layout_t<N, NNZ> &layout) {
  std::vector<compressed_entry> out;
  for (std::size_t j = 0; j < N; ++j) {
    for (auto k = static_cast<std::size_t>(layout.outer[j]);
         k < static_cast<std::size_t>(layout.outer[j + 1]); ++k) {
      out.push_back({j, static_cast<std::size_t>(layout.inner[k]), k});
    }
  }
  return out;
}

} // namespace detail

// The sparse counterpart of scatter_targets: the sweep writes straight into
// compressed storage, so the dense N x N matrix is never materialised.
template <CExpression Expr,
          std::size_t N = detail::expr_arity_v<Expr>>
consteval scatter_map<N> sparse_slots() noexcept {
  constexpr auto coloring = color_columns<N>(hessian_pattern_v<Expr>);
  constexpr auto layout = sparse_layout<Expr>();

  auto slots = unmapped<N>();
  for (const auto [j, i, k] : detail::compressed_entries(layout)) {
    slots[coloring.color[j]][i] = k;
  }
  return slots;
}

// layout_sparse_pattern<Expr>: the compressed Hessian, indexed as if dense.
//
// An mdspan mapping must be total, and a structural zero has no slot -- so the
// span is nnz + 1 cells and every structural zero maps to the last one, which
// the owner keeps at zero.  Reads give 0.0; writes are discarded.  That costs
// uniqueness, hence is_always_unique() == false.
template <CExpression Expr,
          std::size_t N = detail::expr_arity_v<Expr>,
          std::size_t NNZ = hessian_nnz<Expr>()>
consteval std::array<std::size_t, N * N> sparse_slot_table() noexcept {
  constexpr auto layout = sparse_layout<Expr>();
  std::array<std::size_t, N * N> table{};
  table.fill(NNZ);
  for (const auto [j, i, k] : detail::compressed_entries(layout)) {
    table[i * N + j] = k;
  }
  return table;
}

template <CExpression Expr> struct layout_sparse_pattern {
  using expr_type = std::remove_cvref_t<Expr>;
  static constexpr std::size_t kN =
      detail::expr_arity_v<expr_type>;
  static constexpr std::size_t kNnz = hessian_nnz<expr_type>();
  static constexpr auto kTable = sparse_slot_table<expr_type>();

  template <md::CExtents Ext> class mapping {
    static_assert(Ext::rank() == 2);

  public:
    using extents_type = Ext;
    using index_type = typename Ext::index_type;
    using rank_type = typename Ext::rank_type;
    using layout_type = layout_sparse_pattern;

    constexpr mapping() noexcept = default;
    constexpr explicit mapping(const Ext &e) noexcept : ext_(e) {}

    [[nodiscard]] constexpr const Ext &extents() const noexcept { return ext_; }

    [[nodiscard]] constexpr index_type required_span_size() const noexcept {
      return static_cast<index_type>(kNnz + 1);
    }

    [[nodiscard]] constexpr index_type operator()(index_type i,
                                                  index_type j) const noexcept {
      return static_cast<index_type>(
          kTable[static_cast<std::size_t>(i) * kN +
                 static_cast<std::size_t>(j)]);
    }

    // In the pattern, rather than aliasing the sink.
    [[nodiscard]] constexpr bool contains(index_type i,
                                          index_type j) const noexcept {
      return static_cast<std::size_t>((*this)(i, j)) != kNnz;
    }

    static constexpr bool is_always_unique() noexcept { return false; }
    static constexpr bool is_always_exhaustive() noexcept { return false; }
    static constexpr bool is_always_strided() noexcept { return false; }
    [[nodiscard]] constexpr bool is_unique() const noexcept { return false; }
    [[nodiscard]] constexpr bool is_exhaustive() const noexcept {
      return false;
    }
    [[nodiscard]] constexpr bool is_strided() const noexcept { return false; }

    [[nodiscard]] friend constexpr bool operator==(const mapping &a,
                                                   const mapping &b) noexcept {
      return a.ext_ == b.ext_;
    }

  private:
    Ext ext_{};
  };
};

template <CExpression Expr>
[[nodiscard]] constexpr auto sparse_matrix_view(std::span<const double> values)
    noexcept {
  using L = layout_sparse_pattern<std::remove_cvref_t<Expr>>;
  using Ext = md::extents<std::size_t, L::kN, L::kN>;
  return md::mdspan<const double, Ext, L>(values.data(),
                                          typename L::template mapping<Ext>{});
}

} // namespace diff::impl
