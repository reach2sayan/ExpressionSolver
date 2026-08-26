#pragma once

#include "rt/builder.hpp"
#include "rt/expressions.hpp"
#include "rt/opcode.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

namespace ddx::rt {

// Blocked summation over the graph's reduction spines.
//
// A reduction of k terms is k *dependent* adds.  Both the sweep and the kernel
// pay that as latency rather than throughput -- at four cycles per fma an
// rss/64 spine of two thousand is thousands of cycles no vector width can hide,
// because every lane is stalled on the same chain.  Splitting a spine into
// `blocks` interleaved partial sums, combined pairwise at the end, turns one
// chain into that many independent ones.
//
// **This reassociates, deliberately.**  `+` is associative under the algebra
// the library already assumes (see NOTES.md, *What algebra the rewrites may
// assume*), and the rewrite happens in the arena *before any freeze*, so the
// arena walk, every interpreter width and every codegen level still agree with
// each other to the bit -- which is the invariant.  What moves is the recorded
// bit-exactness hash, once.  Multiplication is left alone: its commutativity is
// not assumed.

namespace detail {

// The terms of a maximal single-use Add spine, left to right, so the blocked
// sum is a reassociation of the same sequence rather than of a different one.
template <impl::Numeric T>
[[nodiscard]] std::vector<NodeId>
spine_terms(const Builder<T> &b, NodeId top,
            const std::vector<std::uint32_t> &uses,
            const std::vector<bool> &root) {
  // A root is read from outside the graph, so it is never a spine's interior --
  // a separate flag rather than an extra use, since a root nothing else reads
  // would otherwise look single-use and swallow the spine below it.
  const auto links = [&](NodeId u) {
    return b[u].op == OpCode::Add && uses[u] == 1 && !root[u];
  };
  std::vector<NodeId> terms, stack{top};
  while (!stack.empty()) {
    const NodeId u = stack.back();
    stack.pop_back();
    if (b[u].op != OpCode::Add || (u != top && !links(u))) {
      terms.push_back(u);
      continue;
    }
    const auto [a, c] = b.operands(u);
    stack.push_back(c); // right pushed first, so the left operand pops first
    stack.push_back(a);
  }
  return terms;
}

// `blocks` interleaved partial sums, then a pairwise combine of those.
template <impl::Numeric T>
[[nodiscard]] NodeId blocked_sum(Builder<T> &b, std::span<const NodeId> terms,
                                 std::size_t blocks) {
  const auto as_expr = [&b](NodeId t) { return RTExpression<T>{b, t}; };
  auto partials =
      std::views::iota(0uz, blocks) |
      std::views::transform([&](std::size_t j) {
        return std::ranges::fold_left_first(
            terms | std::views::drop(j) | std::views::stride(blocks) |
                std::views::transform(as_expr),
            std::plus<>{});
      }) |
      std::views::filter([](const auto &o) { return o.has_value(); }) |
      std::views::transform([](const auto &o) { return *o; }) |
      impl::to<std::vector<RTExpression<T>>>();

  // Pairwise rather than left to right: the combine is itself a chain, and at
  // four accumulators a tree is one add of depth two instead of three.
  while (partials.size() > 1) {
    partials = partials | std::views::chunk(2) |
               std::views::transform([](auto pair) {
                 return std::ranges::fold_left_first(pair, std::plus<>{})
                     .value();
               }) |
               impl::to<std::vector<RTExpression<T>>>();
  }
  return partials.front().id(b);
}

} // namespace detail

// Rewrites every reduction spine of at least `least` terms under `roots`, and
// answers where those roots moved to.  Nodes are immutable and interned, so a
// spine cannot be edited in place: the arena is rebuilt bottom-up instead, and
// interning means every subtree that did not change maps straight back to
// itself.  The spines' old interior nodes are left behind for `mark_live` to
// drop at the freeze.
//
// Run it *before* differentiating, so the reverse sweep inherits the shape.
//
// Sixteen accumulators, measured: on rss/64 at one lane the scalar kernel goes
// 3958 ns -> 3388 at four, 3245 at eight, 3118 at sixteen, against 3958 unblocked.
// It is still improving there, but the return per accumulator has flattened and
// a spine shorter than `least` is left alone rather than split into singletons.
template <impl::Numeric T>
[[nodiscard]] std::vector<NodeId> rebalance(Builder<T> &b,
                                            std::span<const NodeId> roots,
                                            std::size_t blocks = 16,
                                            std::size_t least = 16) {
  const auto size = static_cast<NodeId>(b.size());
  const std::vector<bool> live =
      detail::reachable(size, roots, [&b](NodeId v, auto &&mark) {
        for (const NodeId u : b.operands(v)) {
          if (u != no_node) {
            mark(u);
          }
        }
      });

  std::vector<std::uint32_t> uses(size, 0);
  for (const NodeId v : std::views::iota(NodeId{0}, size) |
                            std::views::filter([&](NodeId v) { return live[v]; })) {
    for (const NodeId u : b.operands(v)) {
      if (u != no_node) {
        ++uses[u];
      }
    }
  }
  std::vector<bool> is_root(size, false);
  for (const NodeId r : roots) {
    is_root[r] = true;
  }

  std::vector<NodeId> remap(size, no_node);
  for (const NodeId v : std::views::iota(NodeId{0}, size)) {
    if (!live[v]) {
      continue;
    }
    // By value: building below appends, and the node vector may reallocate.
    const Node<T> node = b[v];
    if (is_leaf(node.op)) {
      remap[v] = v;
      continue;
    }
    const bool spine_top =
        node.op == OpCode::Add && (uses[v] != 1 || is_root[v]);
    if (spine_top) {
      auto terms = detail::spine_terms(b, v, uses, is_root);
      if (terms.size() >= least) {
        // Every term has a lower id, so it is already remapped.
        std::ranges::transform(terms, terms.begin(),
                               [&remap](NodeId t) { return remap[t]; });
        remap[v] = detail::blocked_sum(b, terms, blocks);
        continue;
      }
    }
    remap[v] = arity_of(node.op) == 1
                   ? b.make(node.op, remap[node.a])
                   : b.make(node.op, remap[node.a], remap[node.b]);
  }

  return roots | std::views::transform([&remap](NodeId r) { return remap[r]; }) |
         impl::to<std::vector<NodeId>>();
}

} // namespace ddx::rt
