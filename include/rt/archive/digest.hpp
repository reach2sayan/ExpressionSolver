#pragma once

#include "rt/archive/codec.hpp"
#include "rt/archive/snapshot.hpp" // the described member list this walks
#include "rt/graph.hpp"

#include <boost/describe.hpp>
#include <boost/hash2/fnv1a.hpp>
#include <boost/hash2/hash_append.hpp>
#include <boost/mp11/algorithm.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

// Two keys over a graph: what a file is stale against, and what a compiled
// lane was compiled from.  Both overloads live here together -- one name split
// across two headers is a silent-overload-set hazard.
namespace ddx::rt {

namespace detail {
// to_bits, never the member itself: hash2 appends a float as
// bit_cast<uint64_t>(v + 0), which maps -0.0 onto +0.0.  These digests key the
// object cache, and a graph holding -0.0 must not load +0.0's kernel.
constexpr void fold(boost::hash2::fnv1a_64 &h, const auto &v) {
  boost::hash2::hash_append(h, wire_flavor{}, to_bits(v));
}
} // namespace detail

// The model as the file keys on it: symbols, then the arena up to `upto`.
// Field by field, never a memcpy -- Node<double> has interior padding.
template <impl::Numeric T>
[[nodiscard]] std::uint64_t digest(std::span<const std::string> symbols,
                                   std::span<const Node<T>> nodes,
                                   std::size_t upto) {
  boost::hash2::fnv1a_64 h;
  // Length-framed by hash_append itself, for the symbols and each of them.
  boost::hash2::hash_append(h, detail::wire_flavor{}, symbols);
  detail::fold(h, upto);
  for (const auto &n : nodes.first(std::min(upto, nodes.size()))) {
    boost::mp11::mp_for_each<boost::describe::describe_members<
        Node<T>, boost::describe::mod_public>>(
        [&](auto D) { detail::fold(h, n.*D.pointer); });
  }
  return h.result();
}

// Keyed on what codegen reads of a lane, so a consumer sees a compile is done
// *before* emitting a module.  Ids are construction-ordered, making this a
// within-one-binary key, so a stale one must be a miss.
template <impl::Numeric T>
[[nodiscard]] std::uint64_t digest(const Graph<T> &g) {
  boost::hash2::fnv1a_64 h;
  const auto &layout = g.layout();
  detail::fold(h, g.arity());
  detail::fold(h, layout.values);
  detail::fold(h, layout.jacobian);
  detail::fold(h, layout.hessian);
  const auto order = g.contracted_order();
  detail::fold(h, order.size());
  for (const NodeId v : order) {
    const auto &p = g[v];
    const auto [a, b, c] = g.operands(v);
    detail::fold(h, v);
    detail::fold(h, p.op);
    detail::fold(h, p.slot);
    detail::fold(h, p.value);
    detail::fold(h, a);
    detail::fold(h, b);
    // The third operand too: a select differing only there is a different
    // graph, and this digest is what the object cache keys a kernel on.
    detail::fold(h, c);
  }
  for (const NodeId o : g.outputs()) {
    detail::fold(h, o);
  }
  return h.result();
}

} // namespace ddx::rt
