#pragma once

#include "expressions.hpp"
#include "mpl.hpp"
#include "operations.hpp"
#include "traits.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <functional>
#include <numeric>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

// Hessian sparsity, derived from the expression *type*.
//
// d2f/dxi dxj can only be nonzero if xi and xj meet under something with
// curvature.  The graph is a compile-time type, so that question is answerable
// at compile time.
namespace diff {

// Curvature classification of an operation.
template <typename Op> inline constexpr bool is_linear_op_v = false;
template <Numeric T> inline constexpr bool is_linear_op_v<SumOp<T>> = true;
template <Numeric T> inline constexpr bool is_linear_op_v<NegateOp<T>> = true;

// a*b: curvature only ACROSS the operands (d2/da db = 1), never within one.
template <typename Op> inline constexpr bool is_product_op_v = false;
template <Numeric T> inline constexpr bool is_product_op_v<MultiplyOp<T>> = true;

// a/b: across the operands, plus within the denominator (d2/db2 = 2a/b^3).
// Linear in the numerator, so a-with-a is genuinely absent.
template <typename Op> inline constexpr bool is_quotient_op_v = false;
template <Numeric T> inline constexpr bool is_quotient_op_v<DivideOp<T>> = true;

// Row i of the pattern: the set of j for which d2f/dxi dxj may be nonzero.
template <std::size_t N> using symbol_set = std::bitset<N>;
template <std::size_t N> using coupling_rows = std::array<symbol_set<N>, N>;

namespace detail {

template <std::size_t N> struct coupling_info {
  symbol_set<N> symbols{};  // symbols occurring in this subtree
  coupling_rows<N> rows{};  // rows[i][j]: (i,j) may be nonzero
};

// Mark every (a,b) pair as coupled, both ways — the pattern is symmetric.
template <std::size_t N>
consteval void couple(coupling_rows<N> &rows, const symbol_set<N> &a,
                      const symbol_set<N> &b) noexcept {

  for (auto &&[i, row] : rows | std::views::enumerate) {
    if (a[i]) {
      row |= b;
    }
    if (b[i]) {
      row |= a;
    }
  }
}

// Walk the expression type, propagating (symbols, coupled pairs) upward.
// Recursion is on the type — a storing node is not default-constructible, so
// there is no object to walk.
template <typename E, typename Syms, std::size_t N>
consteval coupling_info<N> coupling_of() noexcept {
  using U = std::remove_cvref_t<E>;
  coupling_info<N> info{};

  if constexpr (is_variable_v<U>) {
    // A frozen variable differentiates to zero, so it contributes no symbol.
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
      // No curvature of its own; the children's coupling is already merged.
    } else if constexpr (is_product_op_v<Op> && K == 2) {
      couple<N>(info.rows, kids[0].symbols, kids[1].symbols);
    } else if constexpr (is_quotient_op_v<Op> && K == 2) {
      couple<N>(info.rows, kids[0].symbols, kids[1].symbols);
      couple<N>(info.rows, kids[1].symbols, kids[1].symbols);
    } else {
      // Any other op (unary math, pow, hypot, ...) is assumed to couple its
      // whole support with itself.
      couple<N>(info.rows, info.symbols, info.symbols);
    }
  }
  // Constant / Lit: no symbols, no coupling.
  return info;
}

} // namespace detail

// The Hessian sparsity pattern of `Expr`: rows[i][j] means d2f/dxi dxj may be
// nonzero.  A conservative superset of the true pattern.
template <typename Expr,
          std::size_t N = mpl::mp_size(
              extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{})>
consteval coupling_rows<N> hessian_pattern() noexcept {
  using Syms = extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  return detail::coupling_of<std::remove_cvref_t<Expr>, Syms, N>().rows;
}

// A variable template is instantiated once per Expr, so the type walk above
// runs once however many of the derived tables below ask for the pattern.
template <typename Expr>
inline constexpr auto hessian_pattern_v = hessian_pattern<Expr>();

// A partition of the columns into groups that can be seeded together.
template <std::size_t N> struct column_coloring {
  std::array<std::size_t, N> color{};
  std::size_t count = 0;
};

// Columns j and k may share a seed only if no row has a nonzero in both —
// otherwise their contributions would add up in that row and be
// indistinguishable.  That is a distance-1 colouring of the column
// intersection graph; greedy is enough, and is what CPR does.
//
// The payoff is structural: a banded Hessian needs a fixed number of colours no
// matter how large n gets, so the driver's sweep count stops growing with n.
// Tridiagonal-plus-corner colours in 5.
template <std::size_t N>
consteval column_coloring<N>
color_columns(const coupling_rows<N> &rows) noexcept {
  column_coloring<N> coloring{};
  for (const auto j : std::views::iota(0uz, N)) {
    // Colours already taken by a column this one conflicts with.
    std::bitset<N> taken{};
    for (const auto k : std::views::iota(0uz, j)) {
      if ((rows[j] & rows[k]).any()) {
        taken.set(coloring.color[k]);
      }
    }
    // The lowest free colour.  At most j colours are taken, so the search
    // always lands inside [0, N) and the dereference is well defined.
    const auto pick = *std::ranges::find_if(
        std::views::iota(0uz, N), [&taken](std::size_t c) { return !taken[c]; });
    coloring.color[j] = pick;
    coloring.count = std::max(coloring.count, pick + 1);
  }
  return coloring;
}

// No column of this colour writes into this row.
inline constexpr std::size_t no_column = static_cast<std::size_t>(-1);

// map[colour][row]: where that row's sweep result belongs, or no_column when
// this colour writes nothing into the row.
template <std::size_t N>
using scatter_map = std::array<std::array<std::size_t, N>, N>;

template <std::size_t N> consteval scatter_map<N> unmapped() noexcept {
  scatter_map<N> map{};
  for (auto &row : map) {
    row.fill(no_column);
  }
  return map;
}

// For each colour, the column each row takes its entry from — the colouring
// guarantees there is at most one, so the driver's scatter is a single pass
// over rows instead of a scan of the whole N x N grid per colour.  Resolving it
// here means the sweep loop carries no search at all.
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

// ===========================================================================
// Compressed-column layout for the sparsity pattern.
//
// The pattern is a compile-time constant, so the index arrays of a compressed
// sparse matrix are too: they can be emitted once as static constexpr data and
// shared by every call, leaving only the nonzero *values* to be computed at run
// time.  No triplet assembly, no sorting, no structure discovery, and — because
// the structure never changes between calls — a consumer may hoist a sparse
// factorization's symbolic analysis out of its loop.
//
// Indices are `int` to match Eigen's default StorageIndex; nothing here
// includes Eigen, this header only produces data Eigen can consume.
// ===========================================================================

template <typename Expr,
          std::size_t N = mpl::mp_size(
              extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{})>
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

// Column-major, row indices ascending within each column — the sorted,
// compressed form Eigen expects.  The pattern is symmetric, so reading it by
// column or by row gives the same structure.
template <typename Expr,
          std::size_t N = mpl::mp_size(
              extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{}),
          std::size_t NNZ = hessian_nnz<Expr>()>
consteval sparse_layout_t<N, NNZ> sparse_layout() noexcept {
  constexpr auto &pattern = hessian_pattern_v<Expr>;
  sparse_layout_t<N, NNZ> layout{};

  // outer is the exclusive prefix sum of the per-column counts.  The scan runs
  // over N + 1 inputs so that it emits N + 1 outputs; the extra count is never
  // added to any of them, and the last output is the total.
  std::array<int, N + 1> counts{};
  std::ranges::transform(pattern, counts.begin(),
                         [](const symbol_set<N> &column) {
                           return static_cast<int>(column.count());
                         });
  std::exclusive_scan(counts.begin(), counts.end(), layout.outer.begin(), 0);

  // Row indices of column j, ascending — by symmetry, the set bits of row j.
  auto out = layout.inner.begin();
  for (const symbol_set<N> &column : pattern) {
    // filter_view caches its begin, so it is not a const-iterable range.
    auto occupied = std::views::iota(0uz, N) |
                    std::views::filter(
                        [&column](std::size_t i) { return column[i]; });
    for (const auto i : occupied) {
      *out++ = static_cast<int>(i);
    }
  }
  return layout;
}

// Per colour, the slot in the VALUE array each row's sweep result belongs to —
// the sparse counterpart of scatter_targets, so the sweep writes straight into
// compressed storage and the dense N x N matrix is never materialised.
template <typename Expr,
          std::size_t N = mpl::mp_size(
              extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{})>
consteval scatter_map<N> sparse_slots() noexcept {
  constexpr auto coloring = color_columns<N>(hessian_pattern_v<Expr>);
  constexpr auto layout = sparse_layout<Expr>();

  auto slots = unmapped<N>();
  for (const auto j : std::views::iota(0uz, N)) {
    for (const auto k : std::views::iota(static_cast<std::size_t>(layout.outer[j]),
                                         static_cast<std::size_t>(layout.outer[j + 1]))) {
      slots[coloring.color[j]][static_cast<std::size_t>(layout.inner[k])] = k;
    }
  }
  return slots;
}

} // namespace diff
