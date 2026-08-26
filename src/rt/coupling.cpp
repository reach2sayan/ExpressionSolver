// The Hessian column colouring.  Here rather than in rt/coupling.hpp because it
// names no `T`: a header would emit one copy of the Boost.Graph machinery per
// scalar, on every consumer's include path.

#include "rt/coupling.hpp"

#include <boost/graph/adjacency_list.hpp>
// First, and not sortable into the block below: smallest_last_ordering.hpp
// uses make_shared_array_property_map without including it, as of Boost 1.92.
#include <boost/property_map/shared_array_property_map.hpp>

#include <boost/graph/sequential_vertex_coloring.hpp>
#include <boost/graph/smallest_last_ordering.hpp>

#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

namespace ddx::rt {
namespace {

// Undirected: the conflict relation is symmetric.
using ConflictGraph =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;

} // namespace

Coloring color_columns(const CouplingRows &rows) {
  const std::size_t n = rows.size();
  Coloring out{.color = std::vector<std::size_t>(n, 0),
               .count = 0,
               .scatter = {},
               .cell = {},
               .cells = 0};
  if (n == 0) {
    return out;
  }

  ConflictGraph conflicts{n};
  const auto conflicting = [&rows, n](const std::size_t j) {
    return std::views::iota(j + 1, n) |
           std::views::filter([&rows, j](const std::size_t k) {
             return rows[j].intersects(rows[k]);
           }) |
           std::views::transform(
               [j](const std::size_t k) { return std::pair{j, k}; });
  };
  for (const auto [j, k] : std::views::iota(0uz, n) |
                               std::views::transform(conflicting) |
                               std::views::join) {
    boost::add_edge(j, k, conflicts);
  }

  // Greedy colouring is only as good as the order it walks.
  const auto order = boost::smallest_last_vertex_ordering(conflicts);
  // clang-format off
  out.count = static_cast<std::size_t>(
    boost::sequential_vertex_coloring(
      conflicts,
      boost::make_iterator_property_map(order.begin(), boost::identity_property_map()),
      boost::make_iterator_property_map(out.color.begin(), boost::get(boost::vertex_index, conflicts))));
  // clang-format on

  // scatter[colour][row] is the column that row's sweep result belongs to; the
  // colouring guarantees at most one, so the harvest needs no search.
  out.scatter.assign(out.count * n, no_column);
  const auto scatter = by_color(out.scatter, out.count, n);
  for (const auto [index, row] : rows | std::views::enumerate) {
    const auto j = static_cast<std::size_t>(index);
    detail::for_each_set(row,
                         [&](std::size_t i) { scatter[out.color[j], i] = j; });
  }

  // Number the owned cells in (colour, row) order, so the compressed block
  // stays deterministic and a caller can index it the way it always could.
  out.cell.assign(out.count * n, no_column);
  const auto cell = by_color(out.cell, out.count, n);
  for (const auto [c, i] : std::views::cartesian_product(
           std::views::iota(0uz, out.count), std::views::iota(0uz, n))) {
    if (scatter[c, i] != no_column) {
      cell[c, i] = out.cells++;
    }
  }
  return out;
}

} // namespace ddx::rt
