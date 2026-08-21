#pragma once

#include "drivers/symbolic.hpp" // DiffMode
#include "rt/coupling.hpp"
#include "rt/expr.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <ranges>
#include <utility>
#include <vector>

namespace ddx::rt {

namespace detail {

// A template, so `if constexpr` actually discards: several descriptors define
// deriv_from_value, which reuses the primal node instead of recomputing it --
// d(exp)/du becomes the exp node itself rather than a second one.
template <typename Fn, impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> rule(const RTExpression<T> &u,
                                             const RTExpression<T> &fu) {
  if constexpr (impl::detail::has_deriv_from_value_v<Fn, RTExpression<T>>) {
    return Fn::deriv_from_value(u, fu);
  } else {
    return Fn::deriv(u);
  }
}

// d/du of a one-argument op.  The eighteen transcendentals come from the
// descriptors in expr/unary_math.hpp instantiated at RTExpression: the rule
// bodies are written against Numeric, so at T = RTExpression they build nodes
// instead of computing.  `neg` and `abs` come from rt/opcode.hpp's table, which
// carries their partials for the same reason -- there is no second copy of the
// chain rule anywhere.
template <impl::Numeric S>
[[nodiscard]] constexpr RTExpression<S>
partial(OpCode op, const RTExpression<S> &u, const RTExpression<S> &f) {
  using T = RTExpression<S>;
  switch (op) {
#define DDX_RT_PARTIAL(fn, Op, label, functor, dU)                             \
  case OpCode::Op:                                                             \
    return dU;
    DDX_RT_UNARY_TABLE(DDX_RT_PARTIAL)
#undef DDX_RT_PARTIAL
#define DDX_RT_PARTIAL(fn, Op, label)                                          \
  case OpCode::Op:                                                             \
    return rule<impl::detail::Op##Fn<RTExpression<S>>>(u, f);
    DDX_UNARY_MATH_TABLE(DDX_RT_PARTIAL)
#undef DDX_RT_PARTIAL
  default:
    return RTExpression<S>{0};
  }
}

// (d/dl, d/dr) of a two-argument op, given the node for the result.
template <impl::Numeric S>
[[nodiscard]] constexpr std::pair<RTExpression<S>, RTExpression<S>>
partials(OpCode op, const RTExpression<S> &l, const RTExpression<S> &r,
         const RTExpression<S> &f) {
  using T = RTExpression<S>;
  switch (op) {
#define DDX_RT_PARTIALS(fn, Op, label, functor, dL, dR)                        \
  case OpCode::Op:                                                             \
    return {dL, dR};
    DDX_RT_BINARY_TABLE(DDX_RT_PARTIALS)
#undef DDX_RT_PARTIALS
  default:
    return {RTExpression<S>{0}, RTExpression<S>{0}};
  }
}

} // namespace detail

struct Gradient {
  NodeId value = no_node;      // the node for f itself
  std::vector<NodeId> partial; // one per symbol, in Builder::symbols() order
};

// One reverse sweep over the whole graph, pushing adjoints from each node to
// its children -- the structural analogue of reverse_sweep in
// drivers/symbolic.hpp, accumulating *nodes* rather than values.  The result
// shares every subexpression it can, because the builder interns.
//
// The sweep appends to the same builder it reads: new nodes land above the
// snapshot, so reverse id order stays a topological order of what came before.
template <impl::Numeric T>
[[nodiscard]] constexpr Gradient reverse_gradient(Builder<T> &b, NodeId root) {
  const auto n = static_cast<NodeId>(b.size());
  std::vector<NodeId> adj(n, no_node);
  adj[root] = b.constant(T{1});

  const auto add_to = [&](NodeId child, const RTExpression<T> &contribution) {
    const NodeId c = contribution.id(b);
    adj[child] = adj[child] == no_node
                     ? c
                     : (RTExpression{b, adj[child]} + RTExpression{b, c}).id(b);
  };

  // Explicitly indexed, not a filtered view: the body writes adjoints into
  // entries this traversal has not reached yet, and a lazy filter would be
  // reading state the loop is still changing.
  for (NodeId v = n; v-- > 0;) {
    if (adj[v] == no_node) {
      continue;
    }
    const auto node = b[v]; // by value: building below may reallocate
    if (is_leaf(node.op)) {
      continue;
    }
    const RTExpression<T> a{b, adj[v]};
    const RTExpression<T> self{b, v};
    if (arity_of(node.op) == 1) {
      const RTExpression<T> u{b, node.a};
      add_to(node.a, a * detail::partial(node.op, u, self));
    } else {
      const RTExpression<T> l{b, node.a};
      const RTExpression<T> r{b, node.b};
      const auto [dl, dr] = detail::partials(node.op, l, r, self);
      add_to(node.a, a * dl);
      add_to(node.b, a * dr);
    }
  }

  // One pass to collect the leaves, rather than re-scanning the graph per
  // symbol.
  Gradient g{.value = root,
             .partial = std::vector<NodeId>(b.symbols().size(), no_node)};
  for (const auto [v, node] : std::views::enumerate(b.nodes().first(n)) |
                                  std::views::filter([](const auto &entry) {
                                    return std::get<1>(entry).op == OpCode::Var;
                                  })) {
    g.partial[node.slot] = adj[v];
  }
  // A symbol the expression never mentions has no leaf, so no adjoint reached
  // it; those partials are the literal zero.
  std::ranges::replace(g.partial, no_node, b.constant(T{0}));
  return g;
}

// Forward accumulation: one pass per symbol, carrying d[v]/dx_s up the graph.
// Here to check Reverse against, not to compute with -- on the graph Reverse
// produces fewer nodes on everything except a single-variable expression, where
// it loses by exactly one.  (README's "Symbolic wins at n=3" is a different
// cost model: there Symbolic evaluates pre-folded partial trees against a sweep
// that pays for a node_cache_t, and on a graph both are simply nodes.)
template <impl::Numeric T>
[[nodiscard]] constexpr Gradient symbolic_gradient(Builder<T> &b, NodeId root) {
  const auto nsym = b.symbols().size();
  Gradient g{.value = root, .partial = std::vector<NodeId>(nsym, no_node)};

  for (std::uint32_t s = 0; s < nsym; ++s) {
    // Snapshot per pass: the previous pass appended nodes, and those are not
    // part of the function being differentiated.
    const auto n = static_cast<NodeId>(b.size());
    std::vector<NodeId> d(n, no_node);

    for (NodeId v = 0; v < n; ++v) {
      const auto node = b[v]; // by value: building below may reallocate
      if (is_leaf(node.op)) {
        d[v] = node.op == OpCode::Var && node.slot == s ? b.constant(T{1})
                                                        : b.constant(T{0});
        continue;
      }
      const RTExpression<T> self{b, v};
      if (arity_of(node.op) == 1) {
        const RTExpression<T> u{b, node.a};
        d[v] =
            (detail::partial(node.op, u, self) * RTExpression<T>{b, d[node.a]})
                .id(b);
      } else {
        const RTExpression<T> l{b, node.a};
        const RTExpression<T> r{b, node.b};
        const auto [dl, dr] = detail::partials(node.op, l, r, self);
        d[v] = (dl * RTExpression<T>{b, d[node.a]} +
                dr * RTExpression<T>{b, d[node.b]})
                   .id(b);
      }
    }
    g.partial[s] = d[root];
  }
  return g;
}

// A system: m functions over the same symbols, so m sweeps sharing one graph.
// The partials are row-major by function, matching Equation::jacobian's
// md_tensor<value_type, extents<m, n>>.
struct Jacobian {
  std::vector<NodeId> value;   // m
  std::vector<NodeId> partial; // m * n, row-major
  std::size_t rows = 0;
  std::size_t columns = 0;

  [[nodiscard]] constexpr NodeId at(std::size_t k, std::size_t j) const {
    return partial[k * columns + j];
  }
};

template <impl::Numeric T>
[[nodiscard]] constexpr Jacobian jacobian(Builder<T> &b,
                                          std::span<const NodeId> roots) {
  Jacobian j{.value = {roots.begin(), roots.end()},
             .partial = {},
             .rows = roots.size(),
             .columns = b.symbols().size()};
  j.partial.reserve(j.rows * j.columns);
  // Each sweep snapshots a builder that already holds the previous rows' nodes.
  // Those are unreachable from this root, so their adjoints stay unset and the
  // sweep skips them; what they do provide is subexpressions to share.
  for (const NodeId r : roots) {
    const auto g = reverse_gradient(b, r);
    j.partial.insert(j.partial.end(), g.partial.begin(), g.partial.end());
  }
  return j;
}

// Explicit template arguments are required, so this never competes with the
// plain spelling below.
template <impl::DiffMode Mode, impl::Numeric T>
[[nodiscard]] constexpr Gradient gradient(Builder<T> &b, NodeId root) {
  if constexpr (Mode == impl::DiffMode::Symbolic) {
    return symbolic_gradient(b, root);
  } else {
    return reverse_gradient(b, root);
  }
}

template <impl::Numeric T>
[[nodiscard]] constexpr Gradient gradient(Builder<T> &b, NodeId root) {
  return reverse_gradient(b, root);
}

// The Hessian, one sweep per colour rather than one per variable.  Colouring
// survives a sweep that builds expressions rather than scattering numbers:
// sweeping from the sum of a colour's partials gives, for row i, the sum over j
// in the colour of d2f/dxi dxj, of which the colouring guarantees at most one
// term is not structurally zero.  The rest fold away as the nodes are formed.
struct Hessian {
  NodeId value = no_node;
  std::vector<NodeId> partial;    // n
  std::vector<NodeId> compressed; // colours * n
  Coloring coloring;
  NodeId zero = no_node;

  // Scattered on read, so a caller never sees the compressed form.
  [[nodiscard]] NodeId at(std::size_t i, std::size_t j) const {
    const std::size_t c = coloring.color[j];
    return coloring.target(c, i) == j ? by_color(compressed, coloring.count,
                                                 coloring.color.size())[c, i]
                                      : zero;
  }
  [[nodiscard]] std::size_t colors() const { return coloring.count; }
};

template <impl::Numeric T>
[[nodiscard]] Hessian hessian(Builder<T> &b, NodeId root) {
  const auto g = reverse_gradient(b, root);
  const auto rows = coupling_pattern(b, root);
  Hessian h{.value = g.value,
            .partial = g.partial,
            .compressed = {},
            .coloring = color_columns(rows),
            .zero = b.constant(T{0})};

  const std::size_t n = h.partial.size();
  h.compressed.assign(h.coloring.count * n, h.zero);
  for (const std::size_t c : std::views::iota(0uz, h.coloring.count)) {
    // Summing a colour's partials before the sweep is what makes one sweep do
    // the work of |colour| of them.
    const auto seed = std::ranges::fold_left(
        std::views::iota(0uz, n) | std::views::filter([&](std::size_t j) {
          return h.coloring.color[j] == c;
        }) | std::views::transform([&](std::size_t j) {
          return RTExpression<T>{b, h.partial[j]};
        }),
        RTExpression<T>{0}, std::plus<>{});
    const auto row = reverse_gradient(b, seed.id(b));
    const auto out = by_color(h.compressed, h.coloring.count, n);
    for (const auto [i, p] : std::views::enumerate(row.partial)) {
      out[c, static_cast<std::size_t>(i)] = p;
    }
  }
  return h;
}

} // namespace ddx::rt
