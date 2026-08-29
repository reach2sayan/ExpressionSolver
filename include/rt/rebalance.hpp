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

// Blocked summation over the reduction spines: k terms are k *dependent* adds,
// which a kernel pays as latency no lane width can hide.
//
// This reassociates, and it runs in the arena before any freeze so that every
// consumer of the *rewritten* roots agrees with every other.  What it does not
// do is make the rewrite universal: `Equation` keeps both root sets and builds
// two graphs from them -- the sweep reads the original, the kernel the blocked
// one -- so on a spine long enough to rewrite, the interpreted and compiled
// answers differ in their last bits (2 ULP at forty terms, 4 at eighty).  That
// is the one place in the library where the two paths disagree at all; the
// Hessian lane shares a single graph and so is exempt.  `*` is left alone: its
// commutativity is not assumed, `+`'s associativity is.

namespace detail {

// A maximal single-use Add spine's terms, left to right: the blocked sum has to
// reassociate the same sequence, not a different one.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId>
spine_terms(const Builder<T> &b, NodeId top,
            const std::vector<std::uint32_t> &uses,
            const std::vector<bool> &root) {
  // `root` is a flag, not an extra use: a root nothing else reads would then
  // look single-use and be swallowed as spine interior.
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
    const auto [a, c, absent] = b.operands(u); // Add is binary
    stack.push_back(c); // right pushed first, so the left operand pops first
    stack.push_back(a);
  }
  return terms;
}

// `blocks` interleaved partial sums, combined pairwise.
template <impl::Numeric T>
[[nodiscard]] constexpr NodeId
blocked_sum(Builder<T> &b, std::span<const NodeId> terms, std::size_t blocks) {
  const auto as_expr = [&b](NodeId t) { return RTExpression<T>{b, t}; };
  auto partials =
      std::views::iota(0uz, blocks) | std::views::transform([&](std::size_t j) {
        return std::ranges::fold_left_first(terms | std::views::drop(j) |
                                                std::views::stride(blocks) |
                                                std::views::transform(as_expr),
                                            std::plus<>{});
      }) |
      std::views::filter([](const auto &o) { return o.has_value(); }) |
      std::views::transform([](const auto &o) { return *o; }) |
      impl::to<std::vector<RTExpression<T>>>();

  // Pairwise: the combine is itself a chain.
  while (partials.size() > 1) {
    partials =
        partials | std::views::chunk(2) | std::views::transform([](auto pair) {
          return std::ranges::fold_left_first(pair, std::plus<>{}).value();
        }) |
        impl::to<std::vector<RTExpression<T>>>();
  }
  return partials.front().id(b);
}

// Where `roots` moved to.  Nodes are interned and immutable, so a spine cannot
// be edited in place: the arena is rebuilt bottom-up and interning maps every
// unchanged subtree back to itself.  Run it before differentiating, so the
// reverse sweep inherits the shape.
//
// Sixteen: rss/64's scalar kernel is 3958 ns unblocked, 3388 at four, 3245 at
// eight, 3118 at sixteen.
template <impl::Numeric T>
[[nodiscard]] constexpr std::vector<NodeId>
rebalance(Builder<T> &b, std::span<const NodeId> roots, std::size_t blocks = 16,
          std::size_t least = 16) {
  const auto size = static_cast<NodeId>(b.size());
  const std::vector<bool> live = detail::reachable(b, roots);

  std::vector<std::uint32_t> uses(size, 0);
  for (const NodeId v : detail::live_ids(live)) {
    std::ranges::for_each(detail::operands_of(b, v),
                          [&uses](NodeId u) { ++uses[u]; });
  }
  std::vector<bool> is_root(size, false);
  for (const NodeId r : roots) {
    is_root[r] = true;
  }

  std::vector<NodeId> remap(size, no_node);
  for (const NodeId v : detail::live_ids(live)) {

    // By value: building below may reallocate.
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
        std::ranges::transform(terms, terms.begin(),
                               [&remap](NodeId t) { return remap[t]; });
        remap[v] = detail::blocked_sum(b, terms, blocks);
        continue;
      }
    }
    // Every operand the node has, not the two a binary one has: rebuilding a
    // select as a pair leaves its third `no_node`, which the next walker
    // dereferences.
    switch (arity_of(node.op)) {
    case 1:
      remap[v] = b.make(node.op, remap[node.a]);
      break;
    case 3:
      remap[v] = b.make(node.op, remap[node.a], remap[node.b], remap[node.c]);
      break;
    default:
      remap[v] = b.make(node.op, remap[node.a], remap[node.b]);
      break;
    }
  }

  return roots |
         std::views::transform([&remap](NodeId r) { return remap[r]; }) |
         impl::to<std::vector<NodeId>>();
}

} // namespace detail
} // namespace ddx::rt
