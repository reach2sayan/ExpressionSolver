#pragma once

#include "jit/kernel.hpp" // jit::Options -- a header type in every build
#include "rt/builder.hpp"
#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/graph.hpp"

#include <boost/describe.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// What a built equation *is*, apart from how it is stored: the arena and the
// sweeps, not a frozen Graph, which freeze() rebuilds in one pass.  archive.hpp
// puts one of these on disk and takes it back; nothing here knows about bytes.
namespace ddx::rt {

// One lane's machine code, and what must agree before it runs; a mismatch means
// recompile, never an error.
struct Object {
  std::uint8_t want = 0; // Equation's Want: values, jacobian, hessian
  std::string symbol;    // underivable, so adopt() is handed it back verbatim
  // jit::Compiler::host_identity(): triple, CPU, folded features, LLVM version.
  std::string host;
  std::uint64_t digest = 0; // digest() of the lane's frozen graph
  jit::Options options;
  std::vector<std::byte> code;
};

// The stored kernel for a lane, if graph, host and codegen options all agree:
// adopt() cannot see that an object came from another graph, so nothing
// reaches it unchecked.
[[nodiscard]] inline const Object *find_object(std::span<const Object> objects,
                                               Want want, std::uint64_t digest,
                                               std::string_view host,
                                               const jit::Options &options) {
  const auto it = std::ranges::find_if(objects, [&](const Object &o) {
    return o.want == std::to_underlying(want) && o.digest == digest &&
           o.host == host && jit::same_codegen(o.options, options) &&
           !o.code.empty();
  });
  return it == objects.end() ? nullptr : &*it;
}

// All the C++ facade and PyEquation share, and so the only thing serialised.
template <impl::Numeric T> struct Snapshot {
  std::vector<std::string> symbols;
  std::vector<Node<T>> nodes;
  std::vector<NodeId> roots;
  Jacobian jacobian;
  std::vector<Hessian> hessians;
  // Serialised rather than re-swept: Equation(Snapshot&&) calls no build_*, so
  // the file describes every block its lanes can freeze.
  HessianVector hvp;
  VectorJacobian vjp;
  Tangent jvp;
  jit::Options options;
  // The staleness key: rebuilding the model reproduces this prefix.
  std::uint32_t model_nodes = 0;
  std::vector<Object> objects;
  BOOST_DESCRIBE_CLASS(Snapshot, (),
                       (symbols, nodes, roots, jacobian, hessians, hvp, vjp,
                        jvp, options, model_nodes, objects),
                       (), ())
};

} // namespace ddx::rt

namespace ddx::rt {

BOOST_DESCRIBE_STRUCT(Sparsity, (), (rowptr, col, rows, columns))
BOOST_DESCRIBE_STRUCT(Jacobian, (), (value, partial, pattern, zero))
BOOST_DESCRIBE_STRUCT(Hessian, (), (value, partial, compressed, coloring, zero))
BOOST_DESCRIBE_STRUCT(HessianVector, (), (value, partial, product))
BOOST_DESCRIBE_STRUCT(VectorJacobian, (), (value, product))
BOOST_DESCRIBE_STRUCT(Tangent, (), (value, product))
BOOST_DESCRIBE_STRUCT(Coloring, (), (color, count, scatter, cell, cells))
BOOST_DESCRIBE_STRUCT(Object, (), (want, symbol, host, digest, options, code))

namespace detail {

// Builder's side of the loader.  A struct, not a friended function template, so
// builder.hpp forward-declares an empty type and gains no include.
struct Restore {
  template <impl::Numeric T>
  static void into(Builder<T> &b, std::vector<Node<T>> nodes,
                   std::vector<std::string> symbols) {
    b.restore(std::move(nodes), std::move(symbols));
  }
};

} // namespace detail

// Owning, and it consumes the snapshot: the node array *is* the arena's.
template <impl::Numeric T>
[[nodiscard]] std::unique_ptr<Builder<T>> rebuild(Snapshot<T> &snap) {
  auto arena = std::make_unique<Builder<T>>();
  detail::Restore::into(*arena, std::move(snap.nodes), std::move(snap.symbols));
  return arena;
}

} // namespace ddx::rt
