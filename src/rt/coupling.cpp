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

// Four stages, each feeding the next off shared state; a step per member keeps
// that state out of the parameter lists and the order in one place.
class ColumnColoring {
public:
  explicit ColumnColoring(const CouplingRows &rows)
      : rows_{rows}, n_{rows.size()}, conflicts_{n_},
        out_{.color = std::vector<std::size_t>(n_, 0),
             .count = 0,
             .scatter = {},
             .cell = {},
             .cells = 0} {}

  [[nodiscard]] Coloring run() && {
    link_conflicts();
    pick_colors();
    map_scatter();
    number_cells();
    return std::move(out_);
  }

private:
  void link_conflicts() {
    const auto conflicting = [this](const std::size_t j) {
      return std::views::iota(j + 1, n_) |
             std::views::filter([this, j](const std::size_t k) {
               return rows_[j].intersects(rows_[k]);
             }) |
             std::views::transform(
                 [j](const std::size_t k) { return std::pair{j, k}; });
    };
    for (const auto [j, k] : std::views::iota(0uz, n_) |
                                 std::views::transform(conflicting) |
                                 std::views::join) {
      boost::add_edge(j, k, conflicts_);
    }
  }

  void pick_colors() {
    // Greedy colouring is only as good as the order it walks.
    const auto order = boost::smallest_last_vertex_ordering(conflicts_);
    // clang-format off
    out_.count = static_cast<std::size_t>(
      boost::sequential_vertex_coloring(
        conflicts_,
        boost::make_iterator_property_map(order.begin(), boost::identity_property_map()),
        boost::make_iterator_property_map(out_.color.begin(), boost::get(boost::vertex_index, conflicts_))));
    // clang-format on
  }

  // scatter[colour][row] is the column that row's sweep result belongs to; the
  // colouring guarantees at most one, so the harvest needs no search.
  void map_scatter() {
    out_.scatter.assign(out_.count * n_, no_column);
    const auto scatter = by_color(out_.scatter, out_.count, n_);
    for (const auto [index, row] : rows_ | std::views::enumerate) {
      const auto j = static_cast<std::size_t>(index);
      detail::for_each_set(
          row, [&](std::size_t i) { scatter[out_.color[j], i] = j; });
    }
  }

  // Number the owned cells in (colour, row) order, so the compressed block
  // stays deterministic and a caller can index it the way it always could.
  void number_cells() {
    out_.cell.assign(out_.count * n_, no_column);
    const auto cell = by_color(out_.cell, out_.count, n_);
    for (const auto [c, i] : std::views::cartesian_product(
             std::views::iota(0uz, out_.count), std::views::iota(0uz, n_))) {
      if (out_.target(c, i) != no_column) {
        cell[c, i] = out_.cells++;
      }
    }
  }

  const CouplingRows &rows_;
  std::size_t n_;
  ConflictGraph conflicts_;
  Coloring out_;
};

} // namespace

Coloring color_columns(const CouplingRows &rows) {
  if (rows.empty()) {
    return {};
  }
  return ColumnColoring{rows}.run();
}

} // namespace ddx::rt
