#pragma once

#include "md/md.hpp"
#include "rt/builder.hpp"
#include "rt/opcode.hpp"

#include <boost/dynamic_bitset.hpp>
#include <boost/graph/adjacency_list.hpp>
// shared_array_property_map first: smallest_last_ordering.hpp uses
// make_shared_array_property_map without including it, still true as of 1.92.
// Not an unused include, and not safe to sort into the block below.
#include <boost/property_map/shared_array_property_map.hpp>

#include <boost/graph/sequential_vertex_coloring.hpp>
#include <boost/graph/smallest_last_ordering.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <vector>

// The runtime analogue of drivers/coupling.hpp, which answers the same question
// `consteval` from the expression type.  `n` is a run-time value here, so its
// std::bitset<N> rows become packed words.
namespace ddx::rt {

inline constexpr std::size_t no_column = static_cast<std::size_t>(-1);

// boost::dynamic_bitset, not vector<bool>: the colouring's pairwise overlap
// test is the hot loop, and `intersects` is a word-at-a-time AND where a
// vector<bool> scan would be O(n) per pair.  find_first/find_next likewise walk
// only the set bits -- a term in an energy touches two species, not five
// hundred.
using SymbolSet = boost::dynamic_bitset<>;
using CouplingRows = std::vector<SymbolSet>;

// A colouring of the Hessian's columns: two columns share a colour only when no
// row couples them, so one sweep can seed all of a colour's columns at once and
// the results still separate.
// Undirected: the conflict relation is symmetric.
using ConflictGraph =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;

// Compressed-by-colour storage is always colours-major and n wide -- the
// scatter table, and every Hessian block harvested against it.  Saying the
// shape once here is what keeps `c * n + i` from being written out at each of
// the places that read or fill one.  Constness follows the range's.
[[nodiscard]] constexpr auto by_color(auto &&flat, std::size_t colors,
                                      std::size_t n) {
  return impl::md::mdspan{std::ranges::data(flat),
                          impl::md::dextents<std::size_t, 2>{colors, n}};
}

struct Coloring {
  std::vector<std::size_t> color; // per symbol
  std::size_t count = 0;
  std::vector<std::size_t> scatter; // count * n; column a (colour, row) owns
  [[nodiscard]] std::size_t target(std::size_t c, std::size_t row) const {
    return by_color(scatter, count, color.size())[c, row];
  }
};

namespace detail {

// Iterating the set bits only, which is what find_first/find_next are for.
inline void for_each_set(const SymbolSet &s,
                         std::invocable<std::size_t> auto &&f) {
  for (auto i = s.find_first(); i != SymbolSet::npos; i = s.find_next(i)) {
    f(i);
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
  for (NodeId v = 0; v < b.size(); ++v) {
    if (!live[v]) {
      continue;
    }
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
      }
      if (node.op == OpCode::Mul) {
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
// color_columns runs at compile time (drivers/coupling.hpp), which carries the
// citations.
//
// An invalid colouring does not degrade the Hessian, it corrupts it: two
// columns sharing a colour have their second derivatives summed into one cell
// with no way to separate them.  Hence the test that a complete graph colours
// in n.
[[nodiscard]] inline Coloring color_columns(const CouplingRows &rows) {
  const std::size_t n = rows.size();
  Coloring out{
      .color = std::vector<std::size_t>(n, 0), .count = 0, .scatter = {}};
  if (n == 0) {
    return out;
  }

  ConflictGraph conflicts(n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = j + 1; k < n; ++k) {
      if (rows[j].intersects(rows[k])) {
        boost::add_edge(j, k, conflicts);
      }
    }
  }

  // Smallest-last rather than natural vertex order: greedy colouring is only as
  // good as the order it walks, and this is the one that pairs with it.
  const auto order = boost::smallest_last_vertex_ordering(conflicts);
  out.count = static_cast<std::size_t>(boost::sequential_vertex_coloring(
      conflicts,
      boost::make_iterator_property_map(order.begin(),
                                        boost::identity_property_map()),
      boost::make_iterator_property_map(
          out.color.begin(), boost::get(boost::vertex_index, conflicts))));

  // scatter[colour][row] is the column that row's sweep result belongs to.  The
  // colouring guarantees at most one such column, so the harvest needs no
  // search.
  out.scatter.assign(out.count * n, no_column);
  for (std::size_t j = 0; j < n; ++j) {
    detail::for_each_set(rows[j],
                         [&, scatter = by_color(out.scatter, out.count, n)](
                             std::size_t i) { scatter[out.color[j], i] = j; });
  }
  return out;
}

} // namespace ddx::rt
