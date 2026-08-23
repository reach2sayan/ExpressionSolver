#pragma once

// Cutting a graph into slabs that emit, compile and register allocate
// independently.  A cut costs its *wavefront* -- the values live across it --
// which is also what the register allocator is superlinear in.
//
// Two measured facts shape this, neither specific to any model.  The wavefront
// is intrinsic: a Boost.Graph DFS post-order moves the peak by 0-6%, because
// reverse mode holds an adjoint live per variable whatever the schedule.  And
// it grows slower than the graph, so traffic per node -- 2 * wavefront / slab
// -- improves with size, which is why this is for large graphs only.

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
