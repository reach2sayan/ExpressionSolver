#pragma once

#include "ops/algebra.hpp"
#include "rt/apply.hpp"
#include "rt/opcode.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ddx::rt {

using NodeId = std::uint32_t;
inline constexpr NodeId no_node = ~NodeId{0};

namespace detail {

// What `roots` reach, over a node set whose ids are topological, so one
// descending pass settles it.  Not a filtered view: the body marks entries the
// pass has not reached, which a lazy filter would read mid-write.  Operand
// access is the caller's -- a Builder has `a`/`b`, a frozen Graph a CSR row.
[[nodiscard]] inline std::vector<bool>
reachable(std::size_t n, std::span<const NodeId> roots, auto &&operands_of) {
  std::vector<bool> live(n, false);
  std::ranges::for_each(roots, [&](NodeId r) { live[r] = true; });
  for (NodeId v = static_cast<NodeId>(n); v-- > 0;) {
    if (live[v]) {
      operands_of(v, [&](NodeId u) { live[u] = true; });
    }
  }
  return live;
}

} // namespace detail

template <impl::Numeric T> struct Node {
  OpCode op = OpCode::Const;
  NodeId a = no_node;
  NodeId b = no_node;
  T value{};              // Const
  std::uint32_t slot = 0; // Var
};

// The mutable half of the graph: nodes are interned as they are formed, so an
// id *is* the identity of a subexpression.  The compile-time counterpart is the
// operator factories in expr/values.hpp, where identity is a type comparison.
template <impl::Numeric T = double> class Builder {
public:
  using value_type = T;
  using node_type = Node<T>;

  [[nodiscard]] constexpr NodeId constant(const T &v) {
    return intern({.op = OpCode::Const, .value = v});
  }

  [[nodiscard]] constexpr NodeId variable(std::string_view name) {
    const auto it = std::ranges::find(symbols_, name);
    // Index before growing: emplace_back invalidates `it`, and for a new symbol
    // the slot is the old size, which is what end() - begin() already is.
    const auto slot = static_cast<std::uint32_t>(it - symbols_.begin());
    if (it == symbols_.end()) {
      symbols_.emplace_back(name);
    }
    return intern({.op = OpCode::Var, .slot = slot});
  }

  [[nodiscard]] constexpr NodeId make(OpCode op, NodeId a, NodeId b = no_node) {
    if (const auto folded = fold(op, a, b)) {
      return *folded;
    }
    if (is_commutative<T>(op) && b != no_node && b < a) {
      std::swap(a, b);
    }
    return intern({.op = op, .a = a, .b = b});
  }

  [[nodiscard]] constexpr const Node<T> &operator[](NodeId id) const {
    return nodes_[id];
  }
  [[nodiscard]] constexpr std::size_t size() const { return nodes_.size(); }
  [[nodiscard]] constexpr std::span<const Node<T>> nodes() const {
    return nodes_;
  }
  [[nodiscard]] constexpr const std::vector<std::string> &symbols() const {
    return symbols_;
  }

  [[nodiscard]] constexpr bool is_constant(NodeId id, const T &v) const {
    if (nodes_[id].op != OpCode::Const) {
      return false;
    }
    if constexpr (std::equality_comparable<T>) {
      return nodes_[id].value == v;
    } else {
      return false;
    }
  }

private:
  // Open addressing in a plain vector rather than a hash map
  static constexpr std::uint64_t payload_of(const Node<T> &n) {
    if (n.op != OpCode::Const) {
      return std::uint64_t{n.slot};
    }
    // Anything wider than a word lands in one bucket and is separated by
    // `same` below -- a few comparisons among constants, nothing elsewhere.
    if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 8) {
      return std::bit_cast<std::uint64_t>(n.value);
    } else {
      return 0;
    }
  }

  static constexpr bool same(const Node<T> &l, const Node<T> &r) {
    if (l.op != r.op || l.a != r.a || l.b != r.b) {
      return false;
    } else if (l.op != OpCode::Const) {
      return l.slot == r.slot;
    }
    // CFieldLike promises arithmetic, not equality.
    if constexpr (std::equality_comparable<T>) {
      return l.value == r.value;
    } else {
      return false;
    }
  }

  static constexpr std::size_t hash_of(const Node<T> &n) {
    std::uint64_t h = static_cast<std::uint64_t>(n.op);
    const auto mix = [&h](std::uint64_t v) {
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      h *= 0xff51afd7ed558ccdULL;
      h ^= h >> 33;
    };
    mix(n.a);
    mix(n.b);
    mix(payload_of(n));
    return static_cast<std::size_t>(h);
  }

  constexpr void rehash() {
    table_.assign(table_.empty() ? 64 : table_.size() * 2, no_node);
    const std::size_t mask = table_.size() - 1;
    for (NodeId id = 0; id < nodes_.size(); ++id) {
      std::size_t i = hash_of(nodes_[id]) & mask;
      while (table_[i] != no_node) {
        i = (i + 1) & mask;
      }
      table_[i] = id;
    }
  }

  constexpr NodeId intern(const Node<T> &n) {
    // Keep the table at most half full; linear probing degrades sharply past
    // that.
    if ((nodes_.size() + 1) * 2 > table_.size()) {
      rehash();
    }
    const std::size_t mask = table_.size() - 1;
    std::size_t i = hash_of(n) & mask;
    while (table_[i] != no_node) {
      if (same(nodes_[table_[i]], n)) {
        return table_[i];
      }
      i = (i + 1) & mask;
    }
    const auto id = static_cast<NodeId>(nodes_.size());
    nodes_.push_back(n);
    table_[i] = id;
    return id;
  }

  // Is this node the literal k?  A scalar with no equality never matches, so
  // the identity rewrites in ops/algebra.hpp simply stop firing for it.
  constexpr bool holds(NodeId id, int k) const {
    if (id == no_node || nodes_[id].op != OpCode::Const) {
      return false;
    }
    if constexpr (std::equality_comparable<T>) {
      return nodes_[id].value == T(k);
    } else {
      return false;
    }
  }

  // Which RuleOp an opcode is, or none for one carrying no identities.
  static constexpr std::optional<impl::algebra::RuleOp>
  rule_op_of(OpCode op) noexcept {
    switch (op) {
    case OpCode::Add:
      return impl::algebra::RuleOp::Add;
    case OpCode::Mul:
      return impl::algebra::RuleOp::Mul;
    case OpCode::Div:
      return impl::algebra::RuleOp::Div;
    case OpCode::Pow:
      return impl::algebra::RuleOp::Pow;
    case OpCode::Neg:
      return impl::algebra::RuleOp::Neg;
    default:
      return std::nullopt;
    }
  }

  // The same predicates simplify.hpp answers with type traits, answered here
  // by id compare -- interning makes structural identity an id equality.
  constexpr bool holds_pred(impl::algebra::Pred p, NodeId a,
                            NodeId b) const noexcept {
    using impl::algebra::Pred;
    switch (p) {
    case Pred::ZeroA:
      return holds(a, 0);
    case Pred::ZeroB:
      return holds(b, 0);
    case Pred::OneA:
      return holds(a, 1);
    case Pred::OneB:
      return holds(b, 1);
    case Pred::Same:
      return a == b;
    case Pred::AOverB:
      return cancel_quotient(a, b).has_value();
    case Pred::BOverA:
      return cancel_quotient(b, a).has_value();
    case Pred::NegatedA:
      return a != no_node && nodes_[a].op == OpCode::Neg;
    }
    return false;
  }

  constexpr std::optional<NodeId> apply_rule(const impl::algebra::Rule &r,
                                             NodeId a, NodeId b) {
    using impl::algebra::Take;
    switch (r.then) {
    case Take::LitZero:
      return constant(T{0});
    case Take::LitOne:
      return constant(T{1});
    case Take::OperandA:
      return a;
    case Take::OperandB:
      return b;
    case Take::NumeratorOfA:
      return cancel_quotient(a, b);
    case Take::NumeratorOfB:
      return cancel_quotient(b, a);
    case Take::OperandOfA:
      return nodes_[a].a;
    }
    return std::nullopt;
  }

  constexpr std::optional<NodeId> fold(OpCode op, NodeId a, NodeId b) {
    const bool ca = a != no_node && nodes_[a].op == OpCode::Const;
    const bool cb = b != no_node && nodes_[b].op == OpCode::Const;
    const bool unary = arity_of(op) == 1;

    // Constant arithmetic first: it runs the same *_impl functor the
    // compile-time evaluator does, so a folded node and an evaluated one
    // cannot disagree.
    if (unary && ca) {
      return constant(apply<T>(op, nodes_[a].value));
    }
    if (!unary && ca && cb) {
      return constant(apply<T>(op, nodes_[a].value, nodes_[b].value));
    }

    const auto kind = rule_op_of(op);
    if (!kind) {
      return std::nullopt;
    }
    // First match wins, as ops/algebra.hpp is ordered.
    for (const impl::algebra::Rule &r : impl::algebra::kRules) {
      if (r.op != *kind || impl::algebra::is_unary(r) != unary) {
        continue;
      }
      if (r.needs_commutative_multiply && !impl::CCommutativeMultiply<T>) {
        continue;
      }
      if (holds_pred(r.when, a, b)) {
        return apply_rule(r, a, b);
      }
    }
    return std::nullopt;
  }

  // (n/d) * d -> n.  On a DAG the denominator match is an id compare.
  constexpr std::optional<NodeId> cancel_quotient(NodeId quotient,
                                                  NodeId x) const {
    const Node<T> &q = nodes_[quotient];
    return q.op == OpCode::Div && q.b == x ? std::optional{q.a} : std::nullopt;
  }

  std::vector<Node<T>> nodes_;
  std::vector<NodeId>
      table_; // power-of-two capacity; no_node marks a free slot
  std::vector<std::string> symbols_;
};

} // namespace ddx::rt
