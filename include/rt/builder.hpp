#pragma once

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

// What `roots` reach, over a node set whose ids are topological.  One
// descending pass settles it, and the pass is the subtle part: the body marks
// entries it has not visited yet, so this cannot be a filtered view -- a lazy
// filter would be reading the state the loop is still writing.
//
// The operand access is the caller's, because the two callers hold different
// things: a Builder has `a`/`b` on the node, a frozen Graph has a CSR row.
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
// id *is* the identity of a subexpression and a repeat costs a hash lookup.
// Everything here mirrors what the operator factories in expr/values.hpp do at
// compile time, one level cheaper -- structural identity is a uint32 compare
// rather than a type comparison.
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
    // Only a scalar the size of a word can be hashed by its bits.  Anything
    // else lands in one bucket and is separated by `same` below, which costs a
    // few comparisons among constants and nothing at all elsewhere.
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

  // The rewrites of expr/simplify.hpp.  x*0 -> 0, 0/x -> 0 and (n/d)*d -> n are
  // not IEEE-faithful; as there, they cancel arithmetic the derivative rules
  // manufactured rather than anything a caller wrote.
  // "is this node the constant k", for k of 0, 1 or -1.  T{0} and T{1} are what
  // CFieldLike promises; a scalar with no equality simply never matches, and
  // the identity rewrites quietly stop firing for it.
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

  constexpr std::optional<NodeId> fold(OpCode op, NodeId a, NodeId b) {
    const bool ca = a != no_node && nodes_[a].op == OpCode::Const;
    const bool cb = b != no_node && nodes_[b].op == OpCode::Const;

    if (arity_of(op) == 1) {
      if (ca) {
        return constant(apply<T>(op, nodes_[a].value));
      } else if (op == OpCode::Neg && nodes_[a].op == OpCode::Neg) {
        return nodes_[a].a;
      }
      return std::nullopt;
    }
    if (ca && cb) {
      return constant(apply<T>(op, nodes_[a].value, nodes_[b].value));
    }

    switch (op) {
    case OpCode::Add:
      if (holds(a, 0)) {
        return b;
      } else if (holds(b, 0)) {
        return a;
      }
      break;
    case OpCode::Mul:
      if (holds(a, 0) || holds(b, 0)) {
        return constant(T{0});
      }
      if (holds(a, 1)) {
        return b;
      }
      if (holds(b, 1)) {
        return a;
      }
      // (n/d) * d -> n holds for any T: `/` is right division, so d^-1 meets d
      // and cancels whichever side it came from.  d * (n/d) is d n d^-1, which
      // is n only when the factors commute -- the same split simplify.hpp
      // makes.
      if (const auto n = cancel_quotient(a, b).or_else([&] {
            return impl::CCommutativeMultiply<T> ? cancel_quotient(b, a)
                                                 : std::nullopt;
          })) {
        return n;
      }
      break;
    case OpCode::Div:
      if (a == b) {
        return constant(T{1});
      } else if (holds(a, 0)) {
        return constant(T{0});
      } else if (holds(b, 1)) {
        return a;
      }
      break;
    case OpCode::Pow:
      if (holds(b, 0)) {
        return constant(T{1});
      } else if (holds(b, 1)) {
        return a;
      }
      break;
    default:
      break;
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
