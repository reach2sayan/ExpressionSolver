#pragma once

#include "rt/builder.hpp"
#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/opcode.hpp"
#include "util/ranges.hpp"

#include <boost/graph/compressed_sparse_row_graph.hpp>

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <iterator>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ddx::impl {
// Redeclared rather than relied on through rt/expressions.hpp: GraphBuilder
// befriends it below.
template <typename... Ts> class Equation;
} // namespace ddx::impl

namespace ddx::py {
// The Python equation freezes three lanes off one sweep as the facade does, so
// it hands over a Hessian for the same reason.
class PyEquation;
} // namespace ddx::py

namespace ddx::rt {

// Befriended by Graph: the CSR itself is the writer's business.
template <impl::Numeric T> class Dot;

// At namespace scope rather than inside Graph because it carries no `T`: every
// scalar freezes to the same CSR, so the graphviz writer compiles once.
using Adjacency =
    boost::compressed_sparse_row_graph<boost::directedS, boost::no_property,
                                       std::uint32_t>;
using Vertex = boost::graph_traits<Adjacency>::vertex_descriptor;

// A builder frozen into CSR, the form codegen walks.  Operand position rides
// along as an edge attribute, because a CSR row is a set.
template <impl::Numeric T = double> class Graph {
public:
  using value_type = T;

  // Codegen and the caller agree on a column's meaning from this, not by
  // positional convention.
  struct Layout {
    std::size_t values = 0;   // m
    std::size_t jacobian = 0; // the pattern's nonzeros, row-major by function
    std::size_t hessian = 0;  // colours * n, compressed
  };

  struct Property {
    OpCode op = OpCode::Const;
    T value{};
    std::uint32_t slot = 0;
  };

  // `contract` is settled here, not per sweep: it decides the arithmetic, so
  // changing it afterwards means freezing again.
  [[nodiscard]] static Graph freeze(const Builder<T> &b,
                                    std::span<const NodeId> outputs,
                                    Layout layout = {}, Coloring coloring = {},
                                    Sparsity jacobian = {},
                                    bool contract = true) {
    Graph g;
    // A default layout means every output is a value; a stated one has to
    // account for every output, or codegen and the caller disagree on a
    // column's meaning.
    const bool unstated =
        layout.values == 0 && layout.jacobian == 0 && layout.hessian == 0;
    g.layout_ = unstated ? Layout{.values = outputs.size()} : layout;
    assert(g.layout_.values + g.layout_.jacobian + g.layout_.hessian ==
           outputs.size());
    g.coloring_ = std::move(coloring);
    g.jacobian_ = std::move(jacobian);
    g.symbols_.assign(b.symbols().begin(), b.symbols().end());
    g.outputs_.assign(outputs.begin(), outputs.end());
    g.properties_.reserve(b.size());

    std::vector<std::pair<std::size_t, std::size_t>> edges;
    std::vector<std::uint32_t> slots;
    edges.reserve(b.size() * 3);
    slots.reserve(b.size() * 3);

    for (const auto [v, n] : b.nodes() | std::views::enumerate) {
      g.properties_.push_back({.op = n.op, .value = n.value, .slot = n.slot});
      const auto id = static_cast<std::size_t>(v);
      if (arity_of(n.op) >= 1) {
        edges.emplace_back(id, n.a);
        slots.push_back(0);
      }
      if (arity_of(n.op) >= 2) {
        edges.emplace_back(id, n.b);
        slots.push_back(1);
      }
      if (arity_of(n.op) == 3) {
        edges.emplace_back(id, n.c);
        slots.push_back(2);
      }
    }

    g.children_ =
        adjacency_type(boost::edges_are_unsorted_multi_pass, edges.begin(),
                       edges.end(), slots.begin(), b.size());
    g.mark_live();
    g.contract(contract);
    // Read off what survived rather than taken from the builder, so a lane
    // that kept no seed keeps the kernel and the digest it had.
    for (const NodeId v : g.contracted_order_) {
      if (g.properties_[v].op == OpCode::Seed) {
        g.seeds_ = std::max(g.seeds_, std::size_t{g.properties_[v].slot} + 1);
      }
    }
    return g;
  }

  [[nodiscard]] std::size_t size() const { return properties_.size(); }
  [[nodiscard]] const Property &operator[](NodeId v) const {
    return properties_[v];
  }
  [[nodiscard]] std::span<const NodeId> outputs() const { return outputs_; }
  [[nodiscard]] const Layout &layout() const { return layout_; }
  [[nodiscard]] const Coloring &coloring() const { return coloring_; }
  // Which (function, symbol) cells the jacobian block holds, in its order.
  [[nodiscard]] const Sparsity &jacobian_pattern() const { return jacobian_; }
  [[nodiscard]] const std::vector<std::string> &symbols() const {
    return symbols_;
  }

  // How many input columns `xs` carries: symbols().size() is not the width.
  [[nodiscard]] std::size_t arity() const { return symbols_.size() + seeds_; }

  [[nodiscard]] bool live(NodeId v) const { return live_[v]; }

  // How many ids the freeze kept, which is what a compile reports having
  // emitted.
  [[nodiscard]] std::size_t live_count() const { return live_order_.size(); }

  // The ids left to compute, in topological order so a consumer emits them in
  // one pass.  A multiply every reader swallowed into an fma is computed by
  // nobody; contraction_at() decides that arithmetic, off the nodes.  The span
  // must not outlive this graph.
  [[nodiscard]] std::span<const NodeId> contracted_order() const {
    return contracted_order_;
  }

  // The fma at each node of contracted_order(), in step with it.  The freeze
  // already asks contraction_at() per node to decide the order; keeping the
  // answer is what stops every sweep re-deriving it per point.
  [[nodiscard]] std::span<const Contraction> contractions() const {
    return contractions_;
  }

  // Derived once, for codegen, the interpreter and the ABI size checks.
  struct Blocks {
    std::span<const NodeId> values;
    std::span<const NodeId> jacobian;
    std::span<const NodeId> hessian;
  };

  [[nodiscard]] Blocks output_blocks() const {
    const std::span<const NodeId> all{outputs_};
    return {.values = all.first(layout_.values),
            .jacobian = all.subspan(layout_.values, layout_.jacobian),
            .hessian = all.subspan(layout_.values + layout_.jacobian,
                                   layout_.hessian)};
  }

  [[nodiscard]] OpCode op_of(NodeId v) const { return properties_[v].op; }

  // Operands in slot order; arity is at most three, a select being the only op
  // that reaches it.
  [[nodiscard]] std::array<NodeId, 3> operands(NodeId v) const {
    std::array out{no_node, no_node, no_node};
    for (const auto &e : operand_edges(v)) {
      out[children_[e]] = static_cast<NodeId>(boost::target(e, children_));
    }
    return out;
  }

private:
  using adjacency_type = Adjacency;
  using vertex_type = Vertex;

  // Only the DOT writer wants these; everything else reads `operands`.
  friend class Dot<T>;
  [[nodiscard]] const adjacency_type &children() const { return children_; }

  [[nodiscard]] auto operand_edges(NodeId v) const {
    const auto [first, last] =
        boost::out_edges(static_cast<vertex_type>(v), children_);
    return std::ranges::subrange(first, last);
  }

  [[nodiscard]] auto live_nodes() const {
    return std::views::iota(NodeId{0}, static_cast<NodeId>(size())) |
           std::views::filter([this](NodeId v) { return live_[v]; });
  }

  // Nothing but the outputs and what they reach needs to be emitted.
  void mark_live() {
    live_ = detail::reachable(
        properties_.size(), outputs_, [this](NodeId v, auto &&mark) {
          for (const auto &edge : operand_edges(v)) {
            mark(static_cast<NodeId>(boost::target(edge, children_)));
          }
        });
    live_order_ = live_nodes() | impl::to<std::vector<NodeId>>();
  }

  // Liveness under contraction, which is all this freeze settles -- an add
  // reaches the multiply's operands rather than the multiply, so a multiply
  // nothing else reads drops out with the Neg of a subtraction behind it.
  void contract(bool on) {
    if (!on) {
      contracted_order_ = live_order_;
      contractions_ = detail::contraction_table(*this, contracted_order_, false);
      return;
    }
    const std::vector<bool> live = detail::reachable(
        properties_.size(), outputs_, [this](NodeId v, auto &&mark) {
          if (const Contraction c = contraction_at(*this, v)) {
            mark(c.x);
            mark(c.y);
            mark(c.z);
            return;
          }
          for (const auto &edge : operand_edges(v)) {
            mark(static_cast<NodeId>(boost::target(edge, children_)));
          }
        });
    contracted_order_ =
        live_order_ |
        std::views::filter([&live](NodeId v) { return live[v]; }) |
        impl::to<std::vector<NodeId>>();
    contractions_ = detail::contraction_table(*this, contracted_order_);
  }

  std::vector<Property> properties_;
  adjacency_type children_;
  std::vector<NodeId> outputs_;
  Layout layout_;
  Coloring coloring_;
  Sparsity jacobian_;
  std::vector<std::string> symbols_;
  std::size_t seeds_ = 0;
  std::vector<bool> live_;
  std::vector<NodeId> live_order_;
  std::vector<NodeId> contracted_order_;
  std::vector<Contraction> contractions_;
};

// Each step names one block of output columns; `finish` is the only thing that
// produces a Graph.
//
//   GraphBuilder{b}.value(f).build_jacobian().finish()
//   GraphBuilder{b}.values({f0, f1}).build_jacobian().finish()
//   GraphBuilder{b}.value(f).build_jacobian().build_hessian().finish()
//
// The class template argument is deduced from the builder.
template <impl::Numeric T = double> class GraphBuilder {
public:
  explicit constexpr GraphBuilder(Builder<T> &b) noexcept : builder_(&b) {}

  // The function the kernel computes.
  constexpr GraphBuilder &value(const RTExpression<T> &root) {
    return values({root});
  }

  // A system: m functions over the same symbols.
  constexpr GraphBuilder &values(std::initializer_list<RTExpression<T>> roots) {
    roots_ =
        roots |
        std::views::transform([&](const auto &e) { return e.id(*builder_); }) |
        impl::to<std::vector<NodeId>>();
    outputs_ = roots_;
    layout_.values = roots_.size();
    return *this;
  }

  // Nodes a caller already has: Equation builds its derivative in the
  // constructor and freezes only on a batch call.
  constexpr GraphBuilder &values_from(std::span<const NodeId> roots) {
    roots_.assign(roots.begin(), roots.end());
    outputs_.assign(roots.begin(), roots.end());
    layout_.values = roots_.size();
    return *this;
  }

  // The functions to differentiate, and not outputs: the graph then carries
  // no value block, and whatever only the value needs is not live.
  constexpr GraphBuilder &roots_from(std::span<const NodeId> roots) {
    roots_.assign(roots.begin(), roots.end());
    outputs_.clear();
    layout_.values = 0;
    return *this;
  }

  // The whole sweep rather than its nodes: the pattern is what makes the block
  // readable, so the two must not travel separately.
  constexpr GraphBuilder &jacobian_from(const Jacobian &j) {
    outputs_.insert(outputs_.end(), j.partial.begin(), j.partial.end());
    layout_.jacobian = j.partial.size();
    jacobian_ = j.pattern;
    return *this;
  }

  // Every partial, in symbol order.  One reverse sweep per function.
  constexpr GraphBuilder &build_jacobian() {
    return jacobian_from(rt::build_jacobian_impl(*builder_, roots_));
  }

  // Compressed by colour, not n x n: a handful of columns when banded.
  GraphBuilder &build_hessian() {
    const auto h = rt::build_hessian_impl(*builder_, roots_.front());
    outputs_.insert(outputs_.end(), h.compressed.begin(), h.compressed.end());
    layout_.hessian = h.compressed.size();
    coloring_ = h.coloring;
    return *this;
  }

  [[nodiscard]] Graph<T> finish(bool contract = true) const {
    return Graph<T>::freeze(*builder_, outputs_, layout_, coloring_, jacobian_,
                            contract);
  }

private:
  // The Hessian an Equation already holds, so the arena is not swept again:
  // rt::build_hessian_impl appends to the builder, and freeze() must not.
  template <typename... Ts> friend class impl::Equation;
  friend class py::PyEquation;
  constexpr GraphBuilder &hessian_from(const Hessian &h) {
    outputs_.insert(outputs_.end(), h.compressed.begin(), h.compressed.end());
    layout_.hessian = h.compressed.size();
    coloring_ = h.coloring;
    return *this;
  }

  // H v is a second-order block of one column, so the colouring stays empty
  // and coloring() is not always a way to read that block back.
  constexpr GraphBuilder &hessian_vector_from(const HessianVector &h) {
    outputs_.insert(outputs_.end(), h.product.begin(), h.product.end());
    layout_.hessian = h.product.size();
    return *this;
  }

  // J v is a first-order block of m columns, one per function.
  constexpr GraphBuilder &tangent_from(const Tangent &t) {
    outputs_.insert(outputs_.end(), t.product.begin(), t.product.end());
    layout_.jacobian = t.product.size();
    return *this;
  }

  // w'J is a first-order block of n columns and dense, so
  // jacobian_pattern() stays empty.
  constexpr GraphBuilder &vector_jacobian_from(const VectorJacobian &j) {
    outputs_.insert(outputs_.end(), j.product.begin(), j.product.end());
    layout_.jacobian = j.product.size();
    return *this;
  }

  Builder<T> *builder_;
  std::vector<NodeId> roots_;
  std::vector<NodeId> outputs_;
  typename Graph<T>::Layout layout_;
  Coloring coloring_;
  Sparsity jacobian_;
};

template <impl::Numeric T> GraphBuilder(Builder<T> &) -> GraphBuilder<T>;

} // namespace ddx::rt
