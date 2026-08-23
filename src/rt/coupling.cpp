// The Hessian column colouring.  Compiled once here rather than inlined into
// rt/coupling.hpp: it names no `T`, so a header definition would have every
// scalar the runtime graph is instantiated at emit its own copy of the same
// Boost.Graph machinery.  Keeping it out also keeps the conflict graph, the
// smallest-last ordering and the graph algorithms off the include path of every
// translation unit that only wants to evaluate an expression.

#include "rt/coupling.hpp"

#include <boost/graph/adjacency_list.hpp>
// shared_array_property_map first: smallest_last_ordering.hpp uses
// make_shared_array_property_map without including it, still true as of 1.92.
// Not an unused include, and not safe to sort into the block below.
#include <boost/property_map/shared_array_property_map.hpp>

#include <boost/graph/sequential_vertex_coloring.hpp>
#include <boost/graph/smallest_last_ordering.hpp>

#include <cstddef>
#include <vector>

namespace ddx::rt {
namespace {

// Undirected: the conflict relation is symmetric.
using ConflictGraph =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;

} // namespace

Coloring color_columns(const CouplingRows &rows) {
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
