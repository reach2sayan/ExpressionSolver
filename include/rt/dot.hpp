#pragma once

#include "rt/graph.hpp"
#include "rt/opcode.hpp"
#include "util/export.hpp"
#include "util/fmt.hpp" // detail::fmt_put

#include <concepts>
#include <cstdint>
#include <format>
#include <ostream>
#include <span>
#include <string>
#include <vector>

// The frozen graph as DOT, for looking at it:
//
//   std::cout << ddx::rt::Dot{g};          // or `... | dot -Tsvg`
//
// Its own header, not part of graph.hpp: boost/graph/graphviz.hpp carries the
// reader as well as the writer, and the writer lives in src/rt/dot.cpp, so
// graphviz reaches that translation unit and no other.
namespace ddx::rt {

// Live is what codegen emits; All draws what the freeze pruned, dashed.
enum class Scope : std::uint8_t { Live, All };

// What the CSR does not say.  Every field is decided by the graph's scalar, so
// resolving them here is what leaves the writer carrying no `T`.
struct DotNode {
  std::string label;
  std::string shape;
  bool live = true;
  bool show_slots = false;
};

[[nodiscard]] DDX_API std::string
to_dot(const Adjacency &adj, std::span<const DotNode> nodes, Scope scope);

// Borrows and renders on demand, the same shape as jit::Ir; the deleted
// overload makes the compiler check that the graph outlives the handle.
//
// Edges point from a node to its operands, the direction codegen walks, and
// carry the operand slot where the two are not interchangeable.
template <impl::Numeric T = double> class Dot {
public:
  explicit Dot(const Graph<T> &g, Scope scope = Scope::Live) noexcept
      : graph_(g), scope_(scope) {}
  Dot(const Graph<T> &&, Scope = Scope::Live) = delete;

  [[nodiscard]] std::string str() const {
    return to_dot(graph_.children(), describe(), scope_);
  }

  void write(std::ostream &out) const { out << str(); }

private:
  [[nodiscard]] std::vector<DotNode> describe() const {
    std::vector<char> is_output(graph_.size(), 0);
    std::ranges::for_each(graph_.outputs(),
                          [&](NodeId o) { is_output[o] = 1; });

    return std::ranges::to<std::vector<DotNode>>(
        std::views::iota(NodeId{0}, static_cast<NodeId>(graph_.size())) |
        std::views::transform([&](NodeId v) {
          const auto op = graph_[v].op;
          return DotNode{.label = text(v),
                         .shape = is_output[v]  ? "doubleoctagon"
                                  : is_leaf(op) ? "box"
                                                : "ellipse",
                         .live = graph_.live(v),
                         // Only where reading it back matters: nothing to tell
                         // apart on a commutative or unary node.
                         .show_slots =
                             arity_of(op) == 2 && !is_commutative<T>(op)};
        }));
  }

  // `id: what it is` -- codegen, the interpreter and errors all name nodes by
  // the id.
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
