#pragma once

#include "md/md.hpp"
#include "ops/operations.hpp"
#include "rt/builder.hpp"
#include "rt/opcode.hpp"
#include "util/export.hpp"

#include <boost/describe/class.hpp>
#include <boost/dynamic_bitset.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <optional>
#include <ranges>
#include <vector>

// The runtime analogue of drivers/coupling.hpp; `n` is run-time here, so its
// bitset rows become words.
namespace ddx::rt {

inline constexpr std::size_t no_column = static_cast<std::size_t>(-1);

// coupling_pattern's opcode arms mirror what the ops declare of themselves;
// pinned, so the two walks cannot answer different sparsities.
static_assert(impl::SumOp<double>::curvature == impl::Curvature::None &&
              impl::NegateOp<double>::curvature == impl::Curvature::None &&
              impl::MultiplyOp<double>::curvature ==
                  impl::Curvature::Bilinear &&
              impl::DivideOp<double>::curvature == impl::Curvature::Quotient);

// boost::dynamic_bitset, not vector<bool>: the colouring's pairwise overlap
// test is the hot loop, and `intersects` is a word-at-a-time AND.
using SymbolSet = boost::dynamic_bitset<>;
using CouplingRows = std::vector<SymbolSet>;

// Always colours-major and n wide, said once so `c * n + i` is not respelled.
[[nodiscard]] constexpr auto by_color(auto &&flat, std::size_t colors,
                                      std::size_t n) {
  return impl::md::mdspan{std::ranges::data(flat),
                          impl::md::dextents<std::size_t, 2>{colors, n}};
}

// Which cells of a derivative block exist, CSR by row.  Read off a swept row
// rather than propagated: a partial the sweep folded to the zero constant *is*
// structurally zero, so this is exact where `coupling_pattern` is conservative.
struct Sparsity {
  std::vector<std::size_t> rowptr; // rows + 1
  std::vector<std::uint32_t> col;  // nonzeros(), ascending within a row
  std::size_t rows = 0;
  std::size_t columns = 0;

  [[nodiscard]] constexpr std::size_t nonzeros() const noexcept {
    return col.size();
  }

  // What a caller iterates, rather than testing every j.
  [[nodiscard]] constexpr std::span<const std::uint32_t>
  row(std::size_t i) const {
    return std::span{col}.subspan(rowptr[i], rowptr[i + 1] - rowptr[i]);
  }

  // Where (i, j) sits in the compressed block, or nullopt for a cell the
  // structure says cannot be nonzero.
  [[nodiscard]] constexpr std::optional<std::size_t>
  at(std::size_t i, std::size_t j) const noexcept {
    const auto present = row(i);
    const auto found =
        std::ranges::lower_bound(present, static_cast<std::uint32_t>(j));
    return found != present.end() && *found == j
               ? std::optional{rowptr[i] + static_cast<std::size_t>(
                                               found - present.begin())}
               : std::nullopt;
  }

  // Every cell present.
  [[nodiscard]] constexpr bool dense() const noexcept {
    return nonzeros() == rows * columns;
  }
  BOOST_DESCRIBE_CLASS(Sparsity, (), (rowptr, col, rows, columns), (), ())
};

// Two columns share a colour only when no row couples them, so one sweep can
// seed all of a colour's columns and the results still separate.
struct Coloring {
  std::vector<std::size_t> color; // per symbol
  std::size_t count = 0;
  std::vector<std::size_t> scatter; // count * n; column a (colour, row) owns
  // count * n again, but the other question: *where* (colour, row) is stored,
  // or no_column for a cell no column of that colour owns.  Those are the ones
  // a sweep computes for nobody, so they get no storage and no output column.
  std::vector<std::size_t> cell;
  std::size_t cells = 0; // how many there are: the compressed block's width
  [[nodiscard]] constexpr std::size_t target(std::size_t c,
                                             std::size_t row) const {
    return by_color(scatter, count, color.size())[c, row];
  }
  [[nodiscard]] constexpr std::size_t column(std::size_t c,
                                             std::size_t row) const {
    return by_color(cell, count, color.size())[c, row];
  }
  // Where H(i, j) is stored, or nullopt for a cell the colouring calls zero.
  [[nodiscard]] constexpr std::optional<std::size_t>
  cell_of(std::size_t i, std::size_t j) const {
    const std::size_t c = color[j];
    return target(c, i) == j ? std::optional{column(c, i)} : std::nullopt;
  }
  BOOST_DESCRIBE_CLASS(Coloring, (), (color, count, scatter, cell, cells), (),
                       ())
};

namespace detail {

inline void for_each_set(const SymbolSet &s,
                         std::invocable<std::size_t> auto &&f) {
  for (auto i = s.find_first(); i != SymbolSet::npos; i = s.find_next(i)) {
    std::invoke(f, i);
  }
}

inline void couple(CouplingRows &rows, const SymbolSet &a, const SymbolSet &b) {
  for_each_set(a, [&](std::size_t i) {
    for_each_set(b, [&](std::size_t j) {
      rows[i].set(j);
      rows[j].set(i);
    });
  });
}

} // namespace detail

// rows[i][j] is true when d2f/dxi dxj may be nonzero.  Conservative and
// symmetric, as hessian_pattern is: a superset costs sweeps, never correctness.
template <impl::Numeric T>
[[nodiscard]] CouplingRows coupling_pattern(const Builder<T> &b, NodeId root) {
  const std::size_t n = b.symbols().size();
  CouplingRows rows(n, SymbolSet(n));

  // Only what the root reaches: another expression sharing the builder would
  // otherwise contribute couplings this one does not have.
  const auto live = detail::reachable(b, std::span{&root, 1});

  std::vector<SymbolSet> support(b.size());
  for (const NodeId v : detail::live_ids(live)) {
    const auto &node = b[v];
    switch (arity_of(node.op)) {
    case 0:
      support[v] = SymbolSet(n);
      if (node.op == OpCode::Var) {
        support[v].set(node.slot);
      }
      break;
    case 1:
      support[v] = support[node.a];
      // Negation is linear; anything else couples its argument with itself.
      if (node.op != OpCode::Neg) {
        detail::couple(rows, support[v], support[v]);
      }
      break;
    case 3:
      // As linear as an Add: it chooses between two derivatives rather than
      // combining them.  The condition is not differentiated, so it carries no
      // support at all.
      support[v] = support[node.b];
      support[v] |= support[node.c];
      break;
    default:
      // Zero derivative everywhere it exists, so no support and no coupling --
      // the conservative arm below cost a colour or two per tested quantity.
      if (node.op == OpCode::Lt || node.op == OpCode::Le) {
        support[v] = SymbolSet(n);
        break;
      }
      support[v] = support[node.a];
      support[v] |= support[node.b];
      if (node.op == OpCode::Add) {
        break; // linear: contributes nothing
      } else if (node.op == OpCode::Mul) {
        detail::couple(rows, support[node.a], support[node.b]);
      } else if (node.op == OpCode::Div) {
        detail::couple(rows, support[node.a], support[node.b]);
        detail::couple(rows, support[node.b], support[node.b]);
      } else {
        detail::couple(rows, support[v], support[v]);
      }
      break;
    }
  }
  return rows;
}

// Columns j and k conflict iff their coupling rows overlap -- the CPR colouring
// symbolic/coupling.hpp runs at compile time, and carries the citations.
[[nodiscard]] DDX_API Coloring color_columns(const CouplingRows &rows);

} // namespace ddx::rt
