#pragma once

#include "ops/mode.hpp" // DiffMode
#include "rt/coupling.hpp"
#include "rt/expressions.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <ranges>
#include <utility>
#include <vector>

namespace ddx::rt {

namespace detail {

// A template, so `if constexpr` actually discards: deriv_from_value reuses the
// primal node, and d(exp)/du becomes the exp node itself.
template <typename Fn, impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> rule(const RTExpression<T> &u,
                                             const RTExpression<T> &fu) {
  if constexpr (impl::detail::has_deriv_from_value_v<Fn, RTExpression<T>>) {
    return Fn::deriv_from_value(u, fu);
  } else {
    return Fn::deriv(u);
  }
}

// The rule bodies are written against Numeric, so at T = RTExpression the same
// descriptors build nodes instead of computing.
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

// One root's row of the Jacobian.  jacobian() below is an overload set rather
// than two names: the number of roots picks the shape, as output_dim does.
struct JacobianRow {
  NodeId value = no_node;      // the node for f itself
  std::vector<NodeId> partial; // one per symbol, in Builder::symbols() order
};

// reverse_sweep, accumulating *nodes*.  It appends to the builder it reads: new
// nodes land above the snapshot, so reverse id order stays topological.
template <impl::Numeric T>
[[nodiscard]] constexpr JacobianRow reverse_jacobian(Builder<T> &b,
                                                     NodeId root) {
  const auto n = static_cast<NodeId>(b.size());
  std::vector<NodeId> adj(n, no_node);
  adj[root] = b.constant(T{1});

  const auto add_to = [&](NodeId child, const RTExpression<T> &contribution) {
    const NodeId c = contribution.id(b);
    adj[child] = adj[child] == no_node
                     ? c
                     : (RTExpression{b, adj[child]} + RTExpression{b, c}).id(b);
  };

  // Not a filtered view: the body writes adjoints into entries this traversal
  // has not reached.
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

  JacobianRow g{.value = root,
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

// One pass per symbol, carrying d[v]/dx_s up the graph.  Here to check Reverse
// against: Reverse produces fewer nodes on everything but a single variable.
template <impl::Numeric T>
[[nodiscard]] constexpr JacobianRow symbolic_jacobian(Builder<T> &b,
                                                      NodeId root) {
  const auto nsym = b.symbols().size();
  JacobianRow g{.value = root, .partial = std::vector<NodeId>(nsym, no_node)};

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

// m functions over the same symbols: m sweeps sharing one graph, row-major by
// function to match Equation::jacobian.
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
  // The previous rows' nodes are unreachable from this root, so the sweep
  // skips them; what they provide is subexpressions to share.
  for (const NodeId r : roots) {
    const auto g = reverse_jacobian(b, r);
    j.partial.insert(j.partial.end(), g.partial.begin(), g.partial.end());
  }
  return j;
}

// One root rather than a span: a span is not constructible from a NodeId, so
// this and the m-root overload never compete, and DiffMode does not satisfy
// Numeric, which keeps it clear of the plain spelling below.
template <impl::DiffMode Mode, impl::Numeric T>
[[nodiscard]] constexpr JacobianRow jacobian(Builder<T> &b, NodeId root) {
  if constexpr (Mode == impl::DiffMode::Symbolic) {
    return symbolic_jacobian(b, root);
  } else {
    return reverse_jacobian(b, root);
  }
}

template <impl::Numeric T>
[[nodiscard]] constexpr JacobianRow jacobian(Builder<T> &b, NodeId root) {
  return reverse_jacobian(b, root);
}

// One sweep per colour.  Sweeping from the sum of a colour's partials gives,
// for row i, the sum over j in the colour of d2f/dxi dxj -- of which the
// colouring guarantees at most one is not structurally zero.
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
  const auto g = reverse_jacobian(b, root);
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
    const auto row = reverse_jacobian(b, seed.id(b));
    const auto out = by_color(h.compressed, h.coloring.count, n);
    for (const auto [i, p] : std::views::enumerate(row.partial)) {
      out[c, static_cast<std::size_t>(i)] = p;
    }
  }
  return h;
}

} // namespace ddx::rt
