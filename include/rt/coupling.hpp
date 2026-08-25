#pragma once

#include "md/md.hpp"
#include "rt/builder.hpp"
#include "rt/opcode.hpp"
#include "util/export.hpp"

#include <boost/dynamic_bitset.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <vector>

// The runtime analogue of drivers/coupling.hpp; `n` is run-time here, so its
// bitset rows become words.
namespace ddx::rt {

inline constexpr std::size_t no_column = static_cast<std::size_t>(-1);

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

// Two columns share a colour only when no row couples them, so one sweep can
// seed all of a colour's columns and the results still separate.
struct Coloring {
  std::vector<std::size_t> color; // per symbol
  std::size_t count = 0;
  std::vector<std::size_t> scatter; // count * n; column a (colour, row) owns
  [[nodiscard]] constexpr std::size_t target(std::size_t c,
                                             std::size_t row) const {
    return by_color(scatter, count, color.size())[c, row];
  }
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
  const auto live = detail::reachable(b.size(), std::span{&root, 1},
                                      [&](NodeId v, auto &&mark) {
                                        const auto &node = b[v];
                                        if (arity_of(node.op) >= 1) {
                                          mark(node.a);
                                        }
                                        if (arity_of(node.op) == 2) {
                                          mark(node.b);
                                        }
                                      });

  std::vector<SymbolSet> support(b.size());
  const auto ids = std::views::iota(NodeId{0}, static_cast<NodeId>(b.size()));
  const auto is_alive_node = [&live](NodeId u) { return live[u]; };

  for (const NodeId v : ids | std::views::filter(is_alive_node)) {
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
    default:
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
// symbolic/coupling.hpp runs at compile time, and carries the citations.  An
// invalid colouring corrupts the Hessian rather than degrading it: two columns
// sharing a colour sum into one cell.  In src/rt/coupling.cpp because it
// carries no `T`.
[[nodiscard]] DDX_API Coloring color_columns(const CouplingRows &rows);

} // namespace ddx::rt
