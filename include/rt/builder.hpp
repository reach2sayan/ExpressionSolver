#pragma once

#include "ops/algebra.hpp"
#include "rt/apply.hpp"
#include "rt/opcode.hpp"

#include <algorithm>
#include <array>
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

// A multiply and an add taken as one rounding: x * y + z, with x negated where
// the multiply reached the add through a Neg -- which is what a subtraction
// builds.  Named by the add; the multiply is a node like any other, and stops
// being emitted only where nothing else reads it.
struct Contraction {
  NodeId x = no_node;
  NodeId y = no_node;
  NodeId z = no_node;
  bool negated = false;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return x != no_node;
  }
};

namespace detail {

// Ids are topological, so one descending pass settles it.  Not a filtered view:
// the body marks entries the pass has not reached, which a lazy filter would
// read mid-write.  Operand access is the caller's.
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

// Nodes are interned as they are formed, so an id *is* the identity of a
// subexpression -- where the compile-time side compares types.
template <impl::Numeric T = double> class Builder {
public:
  using value_type = T;
  using node_type = Node<T>;

  [[nodiscard]] constexpr NodeId constant(const T &v) {
    return intern({.op = OpCode::Const, .value = v});
  }

  [[nodiscard]] constexpr NodeId variable(std::string_view name) {
    const auto it = std::ranges::find(symbols_, name);
    // Index before growing: emplace_back invalidates `it`.
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

  // What a walker reads that does not care which arity a node has, and the two
  // Graph answers as well -- contraction_at() is written once over both.
  [[nodiscard]] constexpr OpCode op_of(NodeId id) const {
    return nodes_[id].op;
  }
  [[nodiscard]] constexpr std::array<NodeId, 2> operands(NodeId id) const {
    return {nodes_[id].a, nodes_[id].b};
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
    // `same` below.
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
      constexpr std::uint64_t golden_ratio_64 = 0x9e3779b97f4a7c15ULL;
      constexpr std::uint64_t murmur_mix_64 = 0xff51afd7ed558ccdULL;
      h ^= v + golden_ratio_64 + (h << 6) + (h >> 2);
      h *= murmur_mix_64;
      h ^= h >> 33;
    };
    mix(n.a);
    mix(n.b);
    mix(payload_of(n));
    return static_cast<std::size_t>(h);
  }

  // The linear probe run from a hash, written once.  Unbounded because the
  // table is kept at most half full: every run reaches an empty slot.
  constexpr auto probe(std::size_t h) const {
    return std::views::iota(h) |
           std::views::transform(
               [mask = table_.size() - 1](std::size_t k) { return k & mask; });
  }

  constexpr void rehash() {
    // bit_ceil, not doubling: the size is the invariant the mask needs, and
    // half full is the load factor linear probing degrades sharply past.
    table_.assign(
        std::bit_ceil(std::max<std::size_t>(64, 2 * (nodes_.size() + 1))),
        no_node);
    for (const auto [id, n] : std::views::enumerate(nodes_)) {
      auto run = probe(hash_of(n));
      table_[*std::ranges::find_if(run, [this](std::size_t i) {
        return table_[i] == no_node;
      })] = static_cast<NodeId>(id);
    }
  }

  constexpr NodeId intern(const Node<T> &n) {
    if ((nodes_.size() + 1) * 2 > table_.size()) {
      rehash();
    }
    // The run stops at whichever comes first, the node itself or the empty
    // slot it would take.
    auto run = probe(hash_of(n));
    const std::size_t i = *std::ranges::find_if(run, [this, &n](std::size_t k) {
      return table_[k] == no_node || same(nodes_[table_[k]], n);
    });
    if (table_[i] != no_node) {
      return table_[i];
    }
    const auto id = static_cast<NodeId>(nodes_.size());
    nodes_.push_back(n);
    table_[i] = id;
    return id;
  }

  // A scalar with no equality never matches, so ops/algebra.hpp's identity
  // rewrites simply stop firing for it.
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

  // simplify.hpp's predicates, answered by id compare: interning makes
  // structural identity an id equality.
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

    // Constant arithmetic first, through the same *_impl functor the
    // compile-time evaluator uses.
    if (unary && ca) {
      return constant(apply<T>(op, nodes_[a].value));
    } else if (!unary && ca && cb) {
      return constant(apply<T>(op, nodes_[a].value, nodes_[b].value));
    }

    const auto kind = rule_op_of(op);
    if (!kind) {
      return std::nullopt;
    }
    // First match wins, as ops/algebra.hpp is ordered.
    const auto r = std::ranges::find_if(
        impl::algebra::kRules, [&](const impl::algebra::Rule &rule) {
          return rule.op == *kind && impl::algebra::is_unary(rule) == unary &&
                 (!rule.needs_commutative_multiply ||
                  impl::CCommutativeMultiply<T>) &&
                 holds_pred(rule.when, a, b);
        });
    return r == std::ranges::cend(impl::algebra::kRules) ? std::nullopt
                                                         : apply_rule(*r, a, b);
  }

  // (n/d) * d -> n.  On a DAG the denominator match is an id compare.
  constexpr std::optional<NodeId> cancel_quotient(NodeId quotient,
                                                  NodeId x) const {
    const Node<T> &q = nodes_[quotient];
    return q.op == OpCode::Div && q.b == x ? std::optional{q.a} : std::nullopt;
  }

  std::vector<Node<T>> nodes_;
  // Open-addressed by hand because interning runs inside constant evaluation
  // and no library hash container is constexpr -- neither
  // boost::unordered_flat_map nor std::unordered_map.
  std::vector<NodeId>
      table_; // power-of-two capacity; no_node marks a free slot
  std::vector<std::string> symbols_;
};

// fadd(fmul(x, y), z) -> fma(x, y, z): the multiply-add every backend forms
// under -ffp-contract=fast, formed here instead so that everything reading the
// graph forms the same ones.  Nothing else contracts -- no reassociation, and a
// division is left alone.
//
// The rule is structural: it asks the add what its operands are and nothing
// about how many readers they have.  That is what lets the arena walk, the
// frozen graph and the kernel agree to the bit, since only the first of the
// three knows nothing about which nodes the others kept.  A multiply two adds
// swallow is *not* computed twice -- an fma is the fmul and the fadd in one
// instruction, so the count is the same either way -- and where it has a reader
// of its own it is emitted as well, exactly as before.
//
// `nodes` is a Builder or a frozen Graph; both answer op_of() and operands().
// The lower-id operand wins where an add has two products, which is the slot
// order Builder::make already sorted commutative operands into.
template <typename Nodes>
[[nodiscard]] constexpr Contraction contraction_at(const Nodes &nodes,
                                                   NodeId v) {
  if (nodes.op_of(v) != OpCode::Add) {
    return {};
  }
  const auto ops = nodes.operands(v);
  for (const auto side : {0uz, 1uz}) {
    const NodeId u = ops[side];
    if (u == no_node) {
      continue;
    }
    const bool negated = nodes.op_of(u) == OpCode::Neg;
    const NodeId m = negated ? nodes.operands(u)[0] : u;
    if (m == no_node || nodes.op_of(m) != OpCode::Mul) {
      continue;
    }
    const auto factors = nodes.operands(m);
    return {.x = factors[0],
            .y = factors[1],
            .z = ops[1 - side],
            .negated = negated};
  }
  return {};
}

} // namespace ddx::rt
