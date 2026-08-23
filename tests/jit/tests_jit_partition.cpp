// The partitioner on its own, before anything is emitted from it.  Its
// invariants are what make a split kernel equal an unsplit one: the slabs cover
// the order once and in order, every value read by a later slab has a scratch
// slot, and nothing is read before the slab that writes it.
//
// partition.cpp is compiled into this target rather than reached through
// libddx: it names no LLVM type and is not part of the public surface, so the
// test compiles the translation unit it is testing.

#include "partition.hpp"

#include "jit/kernel.hpp"

#include "rt/derivative.hpp"
#include "rt/expressions.hpp"
#include "rt/graph.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <ranges>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
using ddx::jit::detail::Partition;
using ddx::rt::Builder;
using ddx::rt::NodeId;
using ddx::rt::no_node;
using RE = ddx::rt::RTExpression<double>;

// A graph with enough coupling to have a wavefront worth cutting: every
// variable reaches every other through the pairwise term.
ddx::rt::Graph<double> coupled(std::size_t n) {
  Builder<> b;
  std::vector<RE> x;
  x.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    x.push_back(var(b, "x" + std::to_string(i)));
  }
  RE f = x[0] * log(x[0]);
  for (std::size_t i = 1; i < n; ++i) {
    f = f + x[i] * log(x[i]);
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      f = f + 0.5 * x[i] * x[j];
    }
  }
  const NodeId root = f.id(b);
  const auto row = ddx::rt::jacobian<ddx::impl::DiffMode::Reverse>(b, root);
  return ddx::rt::GraphBuilder{b}
      .values_from(std::span<const NodeId>{&root, 1})
      .jacobian_from(row.partial)
      .build();
}

TEST(Partition, ATargetOverTheGraphSizeDoesNotCut) {
  const auto g = coupled(6);
  const auto p = ddx::jit::detail::partition(g, g.live_count() * 2);

  ASSERT_EQ(p.slabs.size(), 1u);
  EXPECT_EQ(p.scratch, 0u);
  EXPECT_TRUE(std::ranges::equal(p.order, g.live_order()));
  EXPECT_TRUE(p.slabs[0].live_in.empty());
  EXPECT_TRUE(p.slabs[0].live_out.empty());
  // Nothing goes through memory, which is what makes a small graph's emission
  // identical to what it was before there was a partitioner.
  EXPECT_TRUE(std::ranges::all_of(
      p.slot, [](std::size_t s) { return s == Partition::no_slot; }));
}

TEST(Partition, SlabsCoverTheOrderOnceAndInOrder) {
  const auto g = coupled(12);
  const auto p = ddx::jit::detail::partition(g, 64);
  ASSERT_GT(p.slabs.size(), 1u);

  std::vector<NodeId> seen;
  for (const auto &slab : p.slabs) {
    seen.insert(seen.end(), slab.nodes.begin(), slab.nodes.end());
  }
  EXPECT_TRUE(std::ranges::equal(seen, p.order));
  EXPECT_TRUE(std::ranges::equal(p.order, g.live_order()));
}

TEST(Partition, EveryCrossingValueHasASlotAndIsCarried) {
  const auto g = coupled(12);
  const auto p = ddx::jit::detail::partition(g, 64);

  std::vector<std::size_t> slab_of(g.size(), 0);
  for (const auto [s, slab] : p.slabs | std::views::enumerate) {
    for (const NodeId v : slab.nodes) {
      slab_of[v] = static_cast<std::size_t>(s);
    }
  }

  for (const auto [s, slab] : p.slabs | std::views::enumerate) {
    for (const NodeId v : slab.nodes) {
      for (const NodeId u : g.operands(v)) {
        if (u == no_node || slab_of[u] == static_cast<std::size_t>(s)) {
          continue;
        }
        // Read from an earlier slab: it must have a slot, its own slab must
        // publish it, and this one must ask for it.
        EXPECT_NE(p.slot[u], Partition::no_slot);
        EXPECT_LT(p.slot[u], p.scratch);
        EXPECT_NE(std::ranges::find(p.slabs[slab_of[u]].live_out, u),
                  p.slabs[slab_of[u]].live_out.end());
        EXPECT_NE(std::ranges::find(slab.live_in, u), slab.live_in.end());
      }
    }
  }
}

TEST(Partition, NothingIsReadBeforeTheSlabThatWritesIt) {
  const auto g = coupled(12);
  const auto p = ddx::jit::detail::partition(g, 64);

  std::unordered_set<NodeId> published;
  for (const auto &slab : p.slabs) {
    for (const NodeId v : slab.live_in) {
      EXPECT_TRUE(published.contains(v)) << "slab reads " << v << " unwritten";
    }
    for (const NodeId v : slab.live_out) {
      published.insert(v);
    }
  }
}

TEST(Partition, SlotsAreDistinctPerCarriedValue) {
  const auto g = coupled(12);
  const auto p = ddx::jit::detail::partition(g, 64);

  std::unordered_set<std::size_t> used;
  for (std::size_t v = 0; v < p.slot.size(); ++v) {
    if (p.slot[v] != Partition::no_slot) {
      EXPECT_TRUE(used.insert(p.slot[v]).second) << "slot reused for " << v;
    }
  }
  EXPECT_EQ(used.size(), p.scratch);
}

// The difference array is an optimisation of a definition; the definition is
// cheap enough to check against on a small graph.
TEST(Partition, WavefrontMatchesTheDefinition) {
  const auto g = coupled(5);
  const auto order = g.live_order();
  const auto fast = ddx::jit::detail::wavefront(g, order);
  ASSERT_EQ(fast.size(), order.size());

  std::vector<std::size_t> position(g.size(), 0);
  for (const auto [i, v] : order | std::views::enumerate) {
    position[v] = static_cast<std::size_t>(i);
  }
  const std::unordered_set<NodeId> outputs(g.outputs().begin(),
                                           g.outputs().end());

  for (std::size_t at = 0; at < order.size(); ++at) {
    std::size_t live = 0;
    for (const NodeId v : order) {
      if (position[v] > at) {
        continue;
      }
      const bool used_later =
          outputs.contains(v) ||
          std::ranges::any_of(order, [&](NodeId w) {
            if (position[w] <= at) {
              return false;
            }
            const auto ops = g.operands(w);
            return ops[0] == v || ops[1] == v;
          });
      live += used_later ? 1 : 0;
    }
    EXPECT_EQ(fast[at], live) << "at " << at;
  }
}

} // namespace

// --- the split kernel against the unsplit one --------------------------------
// The gate this whole idea has to pass.  A value crossing a slab boundary
// travels through memory as a double, which is exact -- but the two kernels are
// optimised as different functions, so anything the optimiser could do across
// the cut in one and not the other shows up here.

namespace {

ddx::jit::Compiler &compiler() {
  static ddx::jit::result<ddx::jit::Compiler> c = ddx::jit::Compiler::create();
  EXPECT_TRUE(c.has_value()) << (c ? "" : c.error().detail);
  return *c;
}

struct Batch {
  std::size_t n, points;
  std::vector<std::vector<double>> in, partial;
  std::vector<double> value;
  std::vector<const double *> xs;
  std::vector<double *> gs;

  Batch(std::size_t vars, std::size_t count)
      : n(vars), points(count), in(vars, std::vector<double>(count)),
        partial(vars, std::vector<double>(count)), value(count), xs(vars),
        gs(vars) {
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t p = 0; p < points; ++p) {
        in[i][p] = 0.05 + 0.9 * static_cast<double>((i * 37 + p * 11) % 97) / 97.0;
      }
      xs[i] = in[i].data();
      gs[i] = partial[i].data();
    }
  }

  void run(const ddx::jit::Kernel &k) {
    double *v = value.data();
    k(xs, std::span<double *const>{&v, 1}, gs, {}, points);
  }
};

} // namespace

TEST(SplitKernel, MatchesTheUnsplitKernelBitForBit) {
  const auto g = coupled(24);

  ddx::jit::Options whole;
  ddx::jit::Options split;
  split.split_above = 1;   // always
  split.min_slab = 128;    // and finely, to cross as many boundaries as we can
  split.slab_factor = 1;

  ddx::jit::CompileReport plain_report, split_report;
  auto a = compiler().compile(g, whole, &plain_report);
  auto b = compiler().compile(g, split, &split_report);
  ASSERT_TRUE(a.has_value()) << (a ? "" : a.error().detail);
  ASSERT_TRUE(b.has_value()) << (b ? "" : b.error().detail);
  ASSERT_GT(split_report.slabs, 1u) << "the split kernel was not split";
  EXPECT_EQ(plain_report.slabs, 1u);
  EXPECT_EQ(a->slabs(), 0u);
  EXPECT_EQ(b->slabs(), split_report.slabs);

  Batch whole_batch(24, 16), split_batch(24, 16);
  whole_batch.run(*a);
  split_batch.run(*b);

  for (std::size_t p = 0; p < 16; ++p) {
    EXPECT_EQ(whole_batch.value[p], split_batch.value[p]) << "value at " << p;
  }
  for (std::size_t i = 0; i < 24; ++i) {
    for (std::size_t p = 0; p < 16; ++p) {
      EXPECT_EQ(whole_batch.partial[i][p], split_batch.partial[i][p])
          << "d/dx" << i << " at point " << p;
    }
  }
}

TEST(SplitKernel, ScratchSizeMatchesWhatTheCallerMustSupply) {
  const auto g = coupled(24);
  ddx::jit::Options split;
  split.split_above = 1;
  split.min_slab = 128;
  split.slab_factor = 1;

  ddx::jit::CompileReport rep;
  auto k = compiler().compile(g, split, &rep);
  ASSERT_TRUE(k.has_value()) << (k ? "" : k.error().detail);
  EXPECT_EQ(k->scratch_size(16), rep.scratch_slots * 16);

  // The caller-supplied overload and the thread_local one agree.
  Batch owned(24, 16), given(24, 16);
  owned.run(*k);
  std::vector<double> scratch(k->scratch_size(16));
  double *v = given.value.data();
  (*k)(given.xs, std::span<double *const>{&v, 1}, given.gs, {}, 16, scratch);
  for (std::size_t i = 0; i < 24; ++i) {
    for (std::size_t p = 0; p < 16; ++p) {
      EXPECT_EQ(owned.partial[i][p], given.partial[i][p]);
    }
  }
}
