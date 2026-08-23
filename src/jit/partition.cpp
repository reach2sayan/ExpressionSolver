#include "partition.hpp"

#include <boost/property_map/property_map.hpp>

#include <algorithm>
#include <concepts>
#include <numeric>
#include <ranges>

namespace ddx::jit::detail {
namespace {

// Vertex-indexed storage addressed the way the rest of the project addresses
// this graph -- rt/coupling.cpp's colouring reads its ordering and writes its
// colours through the same pair.
template <std::semiregular V>
[[nodiscard]] auto by_vertex(std::vector<V> &v) {
  return boost::make_iterator_property_map(v.begin(),
                                           boost::identity_property_map{});
}

[[nodiscard]] auto all_edges(const rt::Graph<double> &g) {
  const auto [first, last] = boost::edges(g.children());
  return std::ranges::subrange(first, last);
}

// Last position in `order` whose operands name each node.  Outputs are not
// counted: a slab stores the outputs it defines before it ends, so being an
// output is not on its own a reason to carry a value forward.
//
// One pass over the edges rather than over each vertex's operands -- an edge
// *is* a use, and the graph is an EdgeListGraph, so the traversal is the
// library's rather than a walk written here.  An edge out of a dead vertex is
// not a use of anything.
[[nodiscard]] std::vector<std::size_t>
last_read(const rt::Graph<double> &g, std::span<const rt::NodeId> order) {
  std::vector<std::size_t> position(g.size(), 0);
  for (const auto [i, v] : order | std::views::enumerate) {
    position[v] = static_cast<std::size_t>(i);
  }
  const auto at = by_vertex(position);

  std::vector<std::size_t> last(g.size(), 0);
  auto seen = by_vertex(last);
  const auto &adj = g.children();
  for (const auto &e : all_edges(g)) {
    const auto consumer = boost::source(e, adj);
    if (!g.live(static_cast<rt::NodeId>(consumer))) {
      continue;
    }
    const auto operand = boost::target(e, adj);
    boost::put(seen, operand,
               std::max(boost::get(seen, operand), boost::get(at, consumer)));
  }
  return last;
}

} // namespace

std::vector<std::size_t> wavefront(const rt::Graph<double> &g,
                                   std::span<const rt::NodeId> order) {
  const std::size_t m = order.size();
  auto last = last_read(g, order);
  // For pressure, an output is live until the body stores it, which is the end
  // of the body it is emitted in.
  std::ranges::for_each(g.outputs(), [&](rt::NodeId o) { last[o] = m; });

  // A difference array rather than a set: each value contributes to a
  // contiguous run of positions, so one pass builds every width.
  std::vector<long long> delta(m + 2, 0);
  for (const auto [i, v] : order | std::views::enumerate) {
    const auto defined = static_cast<std::size_t>(i);
    if (last[v] > defined) {
      ++delta[defined];
      --delta[last[v]];
    }
  }

  std::partial_sum(delta.begin(), delta.end(), delta.begin());
  return delta | std::views::take(m) |
         std::views::transform([](long long v) {
           return static_cast<std::size_t>(v);
         }) |
         std::ranges::to<std::vector<std::size_t>>();
}

Partition partition(const rt::Graph<double> &g, std::size_t target) {
  Partition p;
  p.order.assign(g.live_order().begin(), g.live_order().end());
  p.slot.assign(g.size(), Partition::no_slot);

  const std::size_t m = p.order.size();
  const auto width = wavefront(g, p.order);
  p.peak_wavefront = width.empty() ? 0 : *std::ranges::max_element(width);

  // Cut positions: about every `target` nodes, moved to the narrowest point in
  // a window around it, since a few hundred nodes either way is free and the
  // wavefront there is what the cut costs.
  std::vector<std::size_t> cuts;
  if (target > 0 && m > target) {
    // A few hundred nodes either side of the nominal boundary is free, and the
    // wavefront there is exactly what the cut costs, so take the narrowest.
    for (const std::size_t at : std::views::iota(std::size_t{1}, m / target + 1) |
                                    std::views::transform([&](std::size_t k) {
                                      return k * target;
                                    }) |
                                    std::views::take_while(
                                        [m](std::size_t at) { return at < m; })) {
      const auto window = std::views::iota(at - target / 4,
                                           std::min(at + target / 4, m - 1));
      cuts.push_back(*std::ranges::min_element(
                         window, {}, [&](std::size_t i) { return width[i]; }) +
                     1);
    }
  }
  cuts.push_back(m);

  // Which slab defines each node, so a crossing is a comparison rather than a
  // search.
  std::vector<std::size_t> slab_of(g.size(), 0);
  {
    std::size_t s = 0;
    for (const auto [i, v] : p.order | std::views::enumerate) {
      while (static_cast<std::size_t>(i) >= cuts[s]) {
        ++s;
      }
      slab_of[v] = s;
    }
  }

  const auto last = last_read(g, p.order);

  // A value crosses if anything after its own slab reads it.  Outputs are
  // stored by the slab that defines them, so they never need a slot.
  std::size_t next_slot = 0;
  std::vector<std::vector<rt::NodeId>> outs(cuts.size());
  std::vector<std::vector<rt::NodeId>> ins(cuts.size());
  for (const rt::NodeId v : p.order) {
    const std::size_t defined_in = slab_of[v];
    const std::size_t read_until = slab_of[p.order[last[v]]];
    if (read_until > defined_in) {
      p.slot[v] = next_slot++;
      outs[defined_in].push_back(v);
      std::ranges::for_each(std::views::iota(defined_in + 1, read_until + 1),
                            [&](std::size_t s) { ins[s].push_back(v); });
    }
  }
  p.scratch = next_slot;

  std::size_t begin = 0;
  for (const auto [s, end] : cuts | std::views::enumerate) {
    const auto i = static_cast<std::size_t>(s);
    p.slabs.push_back({.nodes = std::span{p.order}.subspan(begin, end - begin),
                       .live_in = std::move(ins[i]),
                       .live_out = std::move(outs[i])});
    begin = end;
  }
  return p;
}

} // namespace ddx::jit::detail
