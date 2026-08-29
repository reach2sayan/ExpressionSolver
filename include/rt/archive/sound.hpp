#pragma once

#include "rt/archive/snapshot.hpp"
#include "rt/coupling.hpp"
#include "rt/opcode.hpp"
#include "util/error.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <vector>

// The trust boundary.  A loaded snapshot has cleared a checksum, which detects
// accidents and confers nothing: what the interpreter, the liveness walk and
// codegen rely on and none re-tests is checked here, and until a file nothing
// could build a graph that broke one.
namespace ddx::rt::detail {

template <impl::Numeric T> class Sound {
public:
  explicit constexpr Sound(const Snapshot<T> &s) noexcept : s_(s) {}

  // Every refusal is the same one: the file parsed, and does not hold.  The
  // order is load-bearing -- each check is what the next one indexes with.
  [[nodiscard]] result<void> operator()() const {
    return symbols() && nodes() && vars() && reachable() && jacobian() &&
                   hessians() && seeded()
               ? result<void>{}
               : fail(errc::archive_corrupt);
  }

private:
  [[nodiscard]] constexpr auto within() const noexcept {
    return [n = s_.nodes.size()](NodeId v) { return v < n; };
  }

  // var() keeps them sorted and unique.
  [[nodiscard]] bool symbols() const {
    return std::ranges::is_sorted(s_.symbols) &&
           std::ranges::adjacent_find(s_.symbols) == s_.symbols.end();
  }

  // Topological: every operand strictly below its reader, which is what the
  // runtime single-passes on.  Also what puts every Var's slot in range, which
  // vars() then indexes by.
  [[nodiscard]] bool nodes() const {
    const auto nsym = s_.symbols.size();
    // A seed slot is an xs column the kernel loads, and the widest block any
    // lane can carry is one per symbol or one per function.
    const auto nseed = std::max(nsym, s_.roots.size());
    return std::ranges::all_of(
        s_.nodes | std::views::enumerate, [nsym, nseed](const auto &at) {
          const auto &[i, node] = at;
          const auto arity = arity_of(node.op);
          return (arity >= 1 ? node.a < static_cast<NodeId>(i)
                             : node.a == no_node) &&
                 (arity >= 2 ? node.b < static_cast<NodeId>(i)
                             : node.b == no_node) &&
                 (arity == 3 ? node.c < static_cast<NodeId>(i)
                             : node.c == no_node) &&
                 (node.op != OpCode::Var || node.slot < nsym) &&
                 (node.op != OpCode::Seed || node.slot < nseed);
        });
  }

  // Exactly one Var per symbol: Builder::restore walks for them, so a symbol
  // named by none would leave a no_node for var() to hand back.  Interning is
  // what makes it exactly one rather than at least one.
  [[nodiscard]] bool vars() const {
    std::vector<std::uint32_t> named(s_.symbols.size(), 0);
    for (const auto &node : s_.nodes) {
      if (node.op == OpCode::Var) {
        ++named[node.slot]; // nodes() pinned the slot
      }
    }
    return std::ranges::all_of(named, [](std::uint32_t k) { return k == 1; });
  }

  [[nodiscard]] bool reachable() const {
    const auto in = within();
    return std::ranges::all_of(s_.roots, in) &&
           std::ranges::all_of(s_.jacobian.value, in) &&
           std::ranges::all_of(s_.jacobian.partial, in) && in(s_.jacobian.zero);
  }

  // The shape is pinned against the roots and the symbols first; `rowptr` is
  // then walked before any `row()` subspan is taken from it, and the columns
  // are checked last -- each is an unchecked payload scalar until it is.
  [[nodiscard]] bool jacobian() const {
    const auto &j = s_.jacobian;
    const auto &p = j.pattern;
    // Strictly ascending inside a row and inside the symbol count: at() binary
    // searches, so a duplicate or an unsorted row makes a cell unreachable.
    const auto sound_row = [&p](std::size_t i) {
      const auto row = p.row(i);
      return std::ranges::all_of(
                 row, [&p](std::uint32_t c) { return c < p.columns; }) &&
             std::ranges::is_sorted(row) &&
             std::ranges::adjacent_find(row) == row.end();
    };
    return p.rows == s_.roots.size() && p.columns == s_.symbols.size() &&
           p.rowptr.size() == p.rows + 1 && p.col.size() == j.partial.size() &&
           p.rowptr.front() == 0 && p.rowptr.back() == p.col.size() &&
           std::ranges::is_sorted(p.rowptr) &&
           std::ranges::all_of(std::views::iota(0uz, p.rows), sound_row);
  }

  // One per root, or none: a constant-evaluated equation never swept them.
  [[nodiscard]] bool hessians() const {
    return (s_.hessians.empty() || s_.hessians.size() == s_.roots.size()) &&
           std::ranges::all_of(
               s_.hessians, [this](const Hessian &h) { return coloured(h); });
  }

  // Each product is either whole or absent -- a constant-evaluated equation
  // swept none, and a half-written one is corrupt.
  [[nodiscard]] bool seeded() const {
    const auto in = within();
    const auto nsym = s_.symbols.size();
    const auto &h = s_.hvp;
    const auto &j = s_.vjp;
    const auto &t = s_.jvp;
    const bool hvp_absent =
        h.value == no_node && h.partial.empty() && h.product.empty();
    const bool vjp_absent = j.value.empty() && j.product.empty();
    const bool jvp_absent = t.value.empty() && t.product.empty();
    return (hvp_absent ||
            (in(h.value) && h.partial.size() == nsym &&
             h.product.size() == nsym && std::ranges::all_of(h.partial, in) &&
             std::ranges::all_of(h.product, in))) &&
           (vjp_absent ||
            (j.value.size() == s_.roots.size() && j.product.size() == nsym &&
             std::ranges::all_of(j.value, in) &&
             std::ranges::all_of(j.product, in))) &&
           (jvp_absent || (t.value.size() == s_.roots.size() &&
                           t.product.size() == s_.roots.size() &&
                           std::ranges::all_of(t.value, in) &&
                           std::ranges::all_of(t.product, in)));
  }

  [[nodiscard]] bool coloured(const Hessian &h) const {
    const auto &c = h.coloring;
    const auto nsym = s_.symbols.size();
    const auto in = within();
    // `count` comes straight off the payload, so `count * nsym` is checked by
    // division -- the product is where a forged count wraps into agreeing.
    // Colours over no symbols are no colours; nothing else pins `count`.
    const auto sized = [nsym, &c](const std::vector<std::size_t> &v) {
      return nsym == 0
                 ? v.empty() && c.count == 0
                 : c.count <= v.size() / nsym && v.size() == c.count * nsym;
    };
    // Every cell must name a slot inside the compressed block: two sharing
    // one would sum two second derivatives into a single column.  `cells` is
    // pinned against the cells actually owned before anything trusts it.
    const auto owned = [&c] {
      return static_cast<std::size_t>(std::ranges::count_if(
          c.cell, [](std::size_t k) { return k != no_column; }));
    };
    return in(h.value) && in(h.zero) && std::ranges::all_of(h.partial, in) &&
           std::ranges::all_of(h.compressed, in) && c.color.size() == nsym &&
           sized(c.scatter) && sized(c.cell) && h.partial.size() == nsym &&
           std::ranges::all_of(c.color,
                               [&c](std::size_t k) { return k < c.count; }) &&
           c.cells == owned() && h.compressed.size() == c.cells &&
           std::ranges::all_of(
               c.cell,
               [&c](std::size_t k) { return k == no_column || k < c.cells; }) &&
           std::ranges::all_of(c.scatter, [nsym](std::size_t k) {
             return k == no_column || k < nsym;
           });
  }

  const Snapshot<T> &s_;
};

} // namespace ddx::rt::detail
