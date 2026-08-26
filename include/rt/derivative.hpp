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
// descriptors build nodes instead of computing.  They return the adjoint
// contribution rather than the bare partial: on the compile-time side the
// association of `adj * dU` is what the BitExactness hash records, so the
// multiply has to live inside the shared rule, not at the call site.
//
// Nothing here passes `f` to a binary rule.  The descriptors recompute it --
// pow(l, r), hypot(l, r), l / r -- which is free on a graph, because
// Builder::make interns the recomputed node straight back onto the primal.
template <impl::Numeric S>
[[nodiscard]] constexpr RTExpression<S>
contribution(OpCode op, const RTExpression<S> &adj, const RTExpression<S> &u,
             const RTExpression<S> &f) {
  using T = RTExpression<S>;
  switch (op) {
#define DDX_RT_CONTRIB(fn, Op, label, functor, Desc)                           \
  case OpCode::Op:                                                             \
    return impl::detail::Desc<T>::adjoints(adj, u)[0];
    DDX_RT_UNARY_TABLE(DDX_RT_CONTRIB)
#undef DDX_RT_CONTRIB
    // The eighteen keep `adj * f'(u)`, already the association
    // DDX_UNARY_MATH_OP's generated adjoints() uses, and `rule` still prefers
    // the deriv_from_value spelling where one exists.
#define DDX_RT_CONTRIB(fn, Op, label)                                          \
  case OpCode::Op:                                                             \
    return adj * rule<impl::detail::Op##Fn<T>>(u, f);
    DDX_UNARY_MATH_TABLE(DDX_RT_CONTRIB)
#undef DDX_RT_CONTRIB
  default:
    return T{0};
  }
}

// The pair reaching a two-argument op's operands.
template <impl::Numeric S>
[[nodiscard]] constexpr std::pair<RTExpression<S>, RTExpression<S>>
contributions(OpCode op, const RTExpression<S> &adj, const RTExpression<S> &l,
              const RTExpression<S> &r) {
  using T = RTExpression<S>;
  switch (op) {
#define DDX_RT_CONTRIB(fn, Op, label, functor, Desc)                           \
  case OpCode::Op: {                                                           \
    const auto c = impl::detail::Desc<T>::adjoints(adj, l, r);                 \
    return {c[0], c[1]};                                                       \
  }
    DDX_RT_BINARY_TABLE(DDX_RT_CONTRIB)
#undef DDX_RT_CONTRIB
  default:
    return {T{0}, T{0}};
  }
}

// The bare partial is the same rule seeded with a unit adjoint: Builder::make
// fires the x*1 rule before a node exists, so this is the partial itself and
// not a multiply by one.
template <impl::Numeric S>
[[nodiscard]] constexpr RTExpression<S>
partial(OpCode op, const RTExpression<S> &u, const RTExpression<S> &f) {
  return contribution(op, RTExpression<S>{1}, u, f);
}

template <impl::Numeric S>
[[nodiscard]] constexpr std::pair<RTExpression<S>, RTExpression<S>>
partials(OpCode op, const RTExpression<S> &l, const RTExpression<S> &r) {
  return contributions(op, RTExpression<S>{1}, l, r);
}

} // namespace detail

// One root's row of the Jacobian.  build_jacobian_impl() below is an overload
// set rather than two names: the number of roots picks the shape, as output_dim
// does.
struct JacobianRow {
  NodeId value = no_node;      // the node for f itself
  std::vector<NodeId> partial; // one per symbol, in Builder::symbols() order
};

// reverse_sweep, accumulating *nodes*.  It appends to the builder it reads: new
// nodes land above the snapshot, so reverse id order stays topological.
template <impl::Numeric T>
[[nodiscard]] constexpr JacobianRow build_reverse_jacobian(Builder<T> &b,
                                                           NodeId root) {
  const auto n = static_cast<NodeId>(b.size());
  std::vector<NodeId> adj(n, no_node);
  adj[root] = b.constant(T{1});

  const auto add_to = [&](NodeId child, const RTExpression<T> &contribution) {
    const NodeId c = contribution.id(b);
    // What sign() -- and so abs() -- hands back.  Storing it would mark the
    // child live and carry a zero adjoint down its whole cone, where every
    // rule folds it away again; the traversal and the interning are the cost.
    if (b.is_constant(c, T{0})) {
      return;
    }
    adj[child] = adj[child] == no_node
                     ? c
                     : (RTExpression{b, adj[child]} + RTExpression{b, c}).id(b);
  };

  // Not a filtered view: the body writes adjoints into entries this traversal
  // has not reached.
  for (NodeId v = n; v-- > 0;) {
    // no_node is "nothing reached it"; a folded Const 0 is "what reached it
    // cancelled".  Neither contributes below, and ids are topological, so by
    // the time the descending pass arrives every parent has had its say.
    if (adj[v] == no_node || b.is_constant(adj[v], T{0})) {
      continue;
    }
    const auto node = b[v]; // by value: building below may reallocate
    if (is_leaf(node.op)) {
      continue;
    }
    const RTExpression<T> a{b, adj[v]};
    if (arity_of(node.op) == 1) {
      const RTExpression<T> u{b, node.a};
      const RTExpression<T> self{b, v};
      add_to(node.a, detail::contribution(node.op, a, u, self));
    } else {
      const RTExpression<T> l{b, node.a};
      const RTExpression<T> r{b, node.b};
      const auto [cl, cr] = detail::contributions(node.op, a, l, r);
      add_to(node.a, cl);
      add_to(node.b, cr);
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
[[nodiscard]] constexpr JacobianRow build_symbolic_jacobian(Builder<T> &b,
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
      if (arity_of(node.op) == 1) {
        const RTExpression<T> u{b, node.a};
        const RTExpression<T> self{b, v};
        d[v] =
            (detail::partial(node.op, u, self) * RTExpression<T>{b, d[node.a]})
                .id(b);
      } else {
        const RTExpression<T> l{b, node.a};
        const RTExpression<T> r{b, node.b};
        const auto [dl, dr] = detail::partials(node.op, l, r);
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
};

template <impl::Numeric T>
[[nodiscard]] constexpr Jacobian
build_jacobian_impl(Builder<T> &b, std::span<const NodeId> roots) {
  Jacobian j{.value = {roots.begin(), roots.end()},
             .partial = {},
             .rows = roots.size(),
             .columns = b.symbols().size()};
  j.partial.reserve(j.rows * j.columns);
  // The previous rows' nodes are unreachable from this root, so the sweep
  // skips them; what they provide is subexpressions to share.
  for (const NodeId r : roots) {
    const auto g = build_reverse_jacobian(b, r);
    j.partial.insert(j.partial.end(), g.partial.begin(), g.partial.end());
  }
  return j;
}

// One root rather than a span: a span is not constructible from a NodeId, so
// this and the m-root overload never compete, and DiffMode does not satisfy
// Numeric, which keeps it clear of the plain spelling below.
template <impl::DiffMode Mode, impl::Numeric T>
[[nodiscard]] constexpr JacobianRow build_jacobian_impl(Builder<T> &b,
                                                        NodeId root) {
  if constexpr (Mode == impl::DiffMode::Symbolic) {
    return build_symbolic_jacobian(b, root);
  } else {
    return build_reverse_jacobian(b, root);
  }
}

template <impl::Numeric T>
[[nodiscard]] constexpr JacobianRow build_jacobian_impl(Builder<T> &b,
                                                        NodeId root) {
  return build_reverse_jacobian(b, root);
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

  // Scattered on read, so a caller never sees the compressed form.  Total over
  // every (i, j): a cell no column owns has no storage at all, and answers the
  // structural zero the colouring already promised it was.
  [[nodiscard]] constexpr NodeId at(std::size_t i, std::size_t j) const {
    const std::size_t c = coloring.color[j];
    return coloring.target(c, i) == j ? compressed[coloring.column(c, i)]
                                      : zero;
  }
  [[nodiscard]] constexpr std::size_t colors() const { return coloring.count; }
};

template <impl::Numeric T>
[[nodiscard]] Hessian build_hessian_impl(Builder<T> &b, NodeId root) {
  const auto g = build_reverse_jacobian(b, root);
  const auto rows = coupling_pattern(b, root);
  Hessian h{.value = g.value,
            .partial = g.partial,
            .compressed = {},
            .coloring = color_columns(rows),
            .zero = b.constant(T{0})};

  const std::size_t n = h.partial.size();
  // Only the cells a column owns.  The rest were swept and thrown away by
  // Hessian::at, which cost an output column each and a whole cone of nodes
  // behind it -- on an arrow-shaped Hessian that is most of the graph.
  h.compressed.assign(h.coloring.cells, h.zero);
  for (const std::size_t c : std::views::iota(0uz, h.coloring.count)) {
    // Summing a colour's partials before the sweep is what makes one sweep do
    // the work of |colour| of them.
    auto match_coloring = [&](std::size_t j) {
      return h.coloring.color[j] == c;
    };
    auto make_partial_expression = [&](std::size_t j) {
      return RTExpression<T>{b, h.partial[j]};
    };
    const auto seed = std::ranges::fold_left(
        std::views::iota(0uz, n) |
            std::views::filter(std::move(match_coloring)) |
            std::views::transform(std::move(make_partial_expression)),
        RTExpression<T>{0}, std::plus<>{});
    const auto row = build_reverse_jacobian(b, seed.id(b));
    for (const auto [i, p] : row.partial | std::views::enumerate) {
      const auto k = h.coloring.column(c, static_cast<std::size_t>(i));
      if (k != no_column) {
        h.compressed[k] = p;
      }
    }
  }
  return h;
}

} // namespace ddx::rt
