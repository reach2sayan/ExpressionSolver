#pragma once

#include "rt/builder.hpp"
#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/opcode.hpp"

#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/range/iterator_range_core.hpp>

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ddx::rt {

// The static graph: a builder frozen into CSR, which is the form codegen walks.
// Operand position rides along as an edge attribute, because a CSR row is a set
// and `a / b` is not `b / a`.
template <impl::Numeric T = double> class Graph {
public:
  using adjacency_type =
      boost::compressed_sparse_row_graph<boost::directedS, boost::no_property,
                                         std::uint32_t>;
  using vertex_type = boost::graph_traits<adjacency_type>::vertex_descriptor;

  using value_type = T;

  // What the output columns are, in the order they are stored: values first,
  // then Jacobian entries, then Hessian entries.  Codegen and the caller agree
  // on the meaning of a column from this, not from a positional convention.
  struct Layout {
    std::size_t values = 0;   // m
    std::size_t jacobian = 0; // m * n, row-major by function
    std::size_t hessian = 0;  // colours * n, compressed
  };

  struct Property {
    OpCode op = OpCode::Const;
    T value{};
    std::uint32_t slot = 0;
  };

  [[nodiscard]] static Graph freeze(const Builder<T> &b,
                                    std::span<const NodeId> outputs,
                                    Layout layout = {},
                                    Coloring coloring = {}) {
    Graph g;
    g.layout_ =
        layout.values == 0
            ? Layout{.values = outputs.size(), .jacobian = 0, .hessian = 0}
            : layout;
    g.coloring_ = std::move(coloring);
    g.symbols_.assign(b.symbols().begin(), b.symbols().end());
    g.outputs_.assign(outputs.begin(), outputs.end());
    g.properties_.reserve(b.size());

    std::vector<std::pair<std::size_t, std::size_t>> edges;
    std::vector<std::uint32_t> slots;
    edges.reserve(b.size() * 2);
    slots.reserve(b.size() * 2);

    for (const auto [v, n] : std::views::enumerate(b.nodes())) {
      g.properties_.push_back({.op = n.op, .value = n.value, .slot = n.slot});
      const auto id = static_cast<std::size_t>(v);
      if (arity_of(n.op) >= 1) {
        edges.emplace_back(id, n.a);
        slots.push_back(0);
      }
      if (arity_of(n.op) == 2) {
        edges.emplace_back(id, n.b);
        slots.push_back(1);
      }
    }

    g.children_ =
        adjacency_type(boost::edges_are_unsorted_multi_pass, edges.begin(),
                       edges.end(), slots.begin(), b.size());
    g.mark_live();
    return g;
  }

  [[nodiscard]] std::size_t size() const { return properties_.size(); }
  [[nodiscard]] const Property &operator[](NodeId v) const {
    return properties_[v];
  }
  [[nodiscard]] const adjacency_type &children() const { return children_; }
  [[nodiscard]] std::span<const NodeId> outputs() const { return outputs_; }
  [[nodiscard]] const Layout &layout() const { return layout_; }
  [[nodiscard]] const Coloring &coloring() const { return coloring_; }
  [[nodiscard]] const std::vector<std::string> &symbols() const {
    return symbols_;
  }
  [[nodiscard]] bool live(NodeId v) const { return live_[v]; }

  // A range, so the walks below are range-for and compose with the algorithms.
  [[nodiscard]] auto operand_edges(NodeId v) const {
    return boost::make_iterator_range(
        boost::out_edges(static_cast<vertex_type>(v), children_));
  }

  // The nodes some output depends on, in id order -- which is topological, so a
  // consumer emits them in one pass and never has to skip anything itself.
  // The view refers to this graph; it must not outlive it.
  [[nodiscard]] auto live_nodes() const {
    return std::views::iota(NodeId{0}, static_cast<NodeId>(size())) |
           std::views::filter([this](NodeId v) { return live_[v]; });
  }

  [[nodiscard]] std::size_t live_count() const {
    return static_cast<std::size_t>(std::ranges::distance(live_nodes()));
  }

  // The three column blocks of outputs(), in the order the layout names them.
  // Codegen, the interpreter and the ABI size checks all need the same split,
  // and deriving it here is what keeps them from each walking one flat list.
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

  // Properties in id order, for a caller that wants to walk them itself.
  [[nodiscard]] std::span<const Property> properties() const {
    return properties_;
  }

  // Operands in slot order.  Arity is at most two, so this is a fixed pair
  // rather than a vector.
  [[nodiscard]] std::array<NodeId, 2> operands(NodeId v) const {
    std::array out{no_node, no_node};
    const auto [first, last] =
        boost::out_edges(static_cast<vertex_type>(v), children_);
    std::ranges::for_each(
        std::ranges::subrange(first, last), [&](const auto &e) {
          out[children_[e]] = static_cast<NodeId>(boost::target(e, children_));
        });
    return out;
  }

private:
  // Nothing but the outputs and what they reach needs to be emitted.
  void mark_live() {
    live_ = detail::reachable(
        properties_.size(), outputs_, [this](NodeId v, auto &&mark) {
          for (const auto &edge : operand_edges(v)) {
            mark(static_cast<NodeId>(boost::target(edge, children_)));
          }
        });
  }

  std::vector<Property> properties_;
  adjacency_type children_;
  std::vector<NodeId> outputs_;
  Layout layout_;
  Coloring coloring_;
  std::vector<std::string> symbols_;
  std::vector<bool> live_;
};

// Assembling what a Graph is frozen with.  Each step names one block of output
// columns, and `build` is the only thing that produces a Graph.
//
//   GraphBuilder{b}.value(f).gradient().build()
//   GraphBuilder{b}.values({f0, f1}).jacobian().build()
//   GraphBuilder{b}.value(f).gradient().hessian().build()
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
        std::ranges::to<std::vector<NodeId>>();
    outputs_ = roots_;
    layout_.values = roots_.size();
    return *this;
  }

  // Reuse nodes a caller already has, rather than sweeping again: Equation
  // builds its derivative in the constructor and freezes only when a batch call
  // arrives.
  constexpr GraphBuilder &values_from(std::span<const NodeId> roots) {
    roots_.assign(roots.begin(), roots.end());
    outputs_.assign(roots.begin(), roots.end());
    layout_.values = roots_.size();
    return *this;
  }

  constexpr GraphBuilder &jacobian_from(std::span<const NodeId> partials) {
    outputs_.insert(outputs_.end(), partials.begin(), partials.end());
    layout_.jacobian = partials.size();
    return *this;
  }

  // Every partial, in symbol order.  One reverse sweep per function.
  constexpr GraphBuilder &jacobian() {
    const auto j = rt::jacobian(*builder_, roots_);
    outputs_.insert(outputs_.end(), j.partial.begin(), j.partial.end());
    layout_.jacobian = j.partial.size();
    return *this;
  }

  // The scalar spelling of the same thing.
  constexpr GraphBuilder &gradient() { return jacobian(); }

  // Compressed by colour, not n x n: for a banded coupling that is a handful of
  // columns rather than hundreds.
  GraphBuilder &hessian() {
    const auto h = rt::hessian(*builder_, roots_.front());
    outputs_.insert(outputs_.end(), h.compressed.begin(), h.compressed.end());
    layout_.hessian = h.compressed.size();
    coloring_ = h.coloring;
    return *this;
  }

  // Anything else worth a column of its own; counted as a value.
  constexpr GraphBuilder &output(const RTExpression<T> &e) {
    outputs_.push_back(e.id(*builder_));
    ++layout_.values;
    return *this;
  }

  [[nodiscard]] Graph<T> build() const {
    return Graph<T>::freeze(*builder_, outputs_, layout_, coloring_);
  }

private:
  Builder<T> *builder_;
  std::vector<NodeId> roots_;
  std::vector<NodeId> outputs_;
  typename Graph<T>::Layout layout_;
  Coloring coloring_;
};

template <impl::Numeric T> GraphBuilder(Builder<T> &) -> GraphBuilder<T>;

} // namespace ddx::rt
