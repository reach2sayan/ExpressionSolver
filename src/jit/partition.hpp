#pragma once

// Cutting a graph into slabs that can be emitted, compiled and register
// allocated independently.
//
// The one quantity that matters is the *wavefront*: how many values are live
// across a given point of the emission order.  It is what a cut has to carry
// through memory, and it is what the register allocator is superlinear in --
// which is where a large graph's compile time goes once the vectoriser is out
// of the way.
//
// Two properties of the wavefront decide the whole design, and neither is a
// property of any particular model:
//
//   It is intrinsic.  Reordering does not help -- a Boost.Graph DFS post-order,
//   which finishes each subexpression before starting the next, moves the peak
//   by 0-6% on every graph measured.  Reverse mode holds an adjoint live per
//   variable and no schedule avoids that, so a cut costs what it costs.
//
//   It grows slower than the graph.  For a model whose cost is quadratic in its
//   variable count the wavefront is linear in it, so cutting gets *cheaper* as
//   the graph grows: traffic per node is 2 * wavefront / slab, and the ratio
//   improves with size.  That is why this exists for large graphs and stays out
//   of the way of small ones.

#include "rt/graph.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace ddx::jit::detail {

// One slab of the emission order, plus what crosses into and out of it.
struct Slab {
  std::span<const rt::NodeId> nodes; // in emission order, into Partition::order
  std::vector<rt::NodeId> live_in;   // read from scratch on entry
  std::vector<rt::NodeId> live_out;  // written to scratch on exit
};

struct Partition {
  std::vector<rt::NodeId> order; // the emission order, all slabs concatenated
  std::vector<Slab> slabs;
  std::vector<std::size_t> slot; // scratch slot per node, or npos
  std::size_t scratch = 0;       // slots needed, the widest cut
  std::size_t peak_wavefront = 0;

  static constexpr std::size_t no_slot = ~std::size_t{0};
};

// Values live across each position of `order`: what a cut there must carry.
[[nodiscard]] std::vector<std::size_t>
wavefront(const rt::Graph<double> &g, std::span<const rt::NodeId> order);

// Cut `g` into slabs of about `target` nodes, each cut placed where the
// wavefront is narrowest nearby.  A `target` at or above the graph's size
// returns a single slab and no scratch, which is what keeps a small graph
// emitting exactly what it emitted before.
[[nodiscard]] Partition partition(const rt::Graph<double> &g,
                                  std::size_t target);

} // namespace ddx::jit::detail
