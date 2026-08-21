#pragma once

#include "rt/graph.hpp"
#include "rt/opcode.hpp"
#include "util/fmt.hpp" // detail::fmt_put

#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/property_map/function_property_map.hpp>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <format>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

// The frozen graph as DOT, for looking at it:
//
//   std::cout << ddx::rt::Dot{g};          // or `... | dot -Tsvg`
//
// Its own header, not part of graph.hpp: boost/graph/graphviz.hpp carries the
// reader as well as the writer, and nothing that only evaluates a graph should
// pay for it.
namespace ddx::rt {

// What to draw.  Live is what codegen emits; All is for the times the question
// is what the freeze pruned, and draws the dead nodes dashed rather than
// dropping them.
enum class Scope : std::uint8_t { Live, All };

// Borrows both ends and renders on demand, the same shape as jit::Ir -- the
// graph has to outlive the handle, which the deleted overload makes the
// compiler check.
//
// Edges point from a node to its operands, the direction codegen walks, and
// carry the operand slot wherever the two are not interchangeable: a CSR row is
// a set, and `a / b` is not `b / a`.
template <impl::Numeric T = double> class Dot {
public:
  explicit Dot(const Graph<T> &g, Scope scope = Scope::Live) noexcept
      : graph_(g), scope_(scope) {}
  Dot(const Graph<T> &&, Scope = Scope::Live) = delete;

  [[nodiscard]] std::string str() const {
    std::ostringstream out;
    write(out);
    return out.str();
  }

  // Attributes go in through dynamic_properties rather than a hand-written
  // label writer, so the names are the ones graphviz knows and BGL does the
  // quoting.
  void write(std::ostream &out) const {
    const auto &adj = graph_.children();

    // One pass over the outputs, wrapped as a property map, rather than a
    // linear search per node: outputs are a list, and the writer asks about
    // every vertex.
    std::vector<char> is_output(boost::num_vertices(adj), 0);
    std::ranges::for_each(graph_.outputs(),
                          [&](NodeId o) { is_output[o] = 1; });
    const auto output_map = boost::make_iterator_property_map(
        is_output.begin(), boost::get(boost::vertex_index, adj));

    boost::dynamic_properties dp;
    dp.property("node_id", boost::get(boost::vertex_index, adj));
    dp.property("label", vertex_map([this](NodeId v) { return text(v); }));
    dp.property("shape", vertex_map([&](NodeId v) {
                  return shape(v, boost::get(output_map, v) != 0);
                }));
    dp.property("style", vertex_map([this](NodeId v) {
                  return std::string(graph_.live(v) ? "solid" : "dashed");
                }));
    dp.property("label", edge_map());

    if (scope_ == Scope::All) {
      boost::write_graphviz_dp(out, adj, dp);
    } else {
      boost::write_graphviz_dp(out, live_view(), dp);
    }
  }

  // The graph codegen actually emits.  An adaptor over the liveness bits the
  // freeze already computed, so it costs a predicate and no traversal, and the
  // vertices keep their original ids.  Not a Graph member: on Graph it would
  // put boost/graph/filtered_graph.hpp in the header every evaluator includes.
  [[nodiscard]] auto live_view() const {
    return boost::filtered_graph<typename Graph<T>::adjacency_type,
                                 boost::keep_all, IsLive>{
        graph_.children(), boost::keep_all{}, IsLive{&graph_}};
  }

private:
  using vertex_type = typename Graph<T>::vertex_type;

  // Only vertices are filtered: an edge out of a live node lands on a live
  // node, because reaching a node is what made its operands live.
  struct IsLive {
    const Graph<T> *graph = nullptr;
    bool operator()(vertex_type v) const {
      return graph->live(static_cast<NodeId>(v));
    }
  };

  using edge_type =
      boost::graph_traits<typename Graph<T>::adjacency_type>::edge_descriptor;

  template <std::invocable<NodeId> F> static auto vertex_map(F f) {
    return boost::make_function_property_map<vertex_type, std::string>(
        [f = std::move(f)](vertex_type v) {
          return f(static_cast<NodeId>(v));
        });
  }

  // The slot, but only where reading it back matters: on a commutative node the
  // number would say nothing, and on a unary one there is nothing to tell
  // apart.
  auto edge_map() const {
    return boost::make_function_property_map<edge_type, std::string>(
        [this, slot = boost::get(boost::edge_bundle, graph_.children())](
            const edge_type &e) -> std::string {
          const auto &adj = graph_.children();
          const auto op = graph_[static_cast<NodeId>(boost::source(e, adj))].op;
          if (arity_of(op) < 2 || is_commutative<T>(op)) {
            return {};
          }
          return std::to_string(boost::get(slot, e));
        });
  }

  // `id: what it is` -- the id because codegen, the interpreter and every error
  // message name nodes by it.
  [[nodiscard]] std::string text(NodeId v) const {
    const auto &p = graph_[v];
    const auto id = std::to_string(v) + ": ";
    switch (p.op) {
    case OpCode::Var:
      return id + graph_.symbols()[p.slot];
    case OpCode::Const:
      if constexpr (std::formattable<T, char>) {
        return id + std::format("{}", p.value);
      } else {
        return id + "const";
      }
    default:
      return id + std::string(label_of(p.op));
    }
  }

  [[nodiscard]] std::string shape(NodeId v, bool output) const {
    if (output) {
      return "doubleoctagon";
    }
    return is_leaf(graph_[v].op) ? "box" : "ellipse";
  }

  friend std::ostream &operator<<(std::ostream &out, const Dot &d) {
    d.write(out);
    return out;
  }

  const Graph<T> &graph_;
  Scope scope_;
};

template <impl::Numeric T> Dot(const Graph<T> &, Scope = Scope::Live) -> Dot<T>;

} // namespace ddx::rt

template <ddx::impl::Numeric T> struct std::formatter<ddx::rt::Dot<T>, char> {
  constexpr auto parse(std::format_parse_context &ctx) const {
    return ctx.begin();
  }
  auto format(const ddx::rt::Dot<T> &d, std::format_context &ctx) const {
    ddx::impl::detail::fmt_put(ctx, d.str());
    return ctx.out();
  }
};
