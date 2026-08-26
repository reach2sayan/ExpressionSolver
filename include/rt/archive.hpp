#pragma once

#include "jit/kernel.hpp" // jit::Options -- a header type in every build
#include "rt/builder.hpp"
#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/graph.hpp"
#include "rt/opcode.hpp"
#include "util/error.hpp"
#include "util/export.hpp"

#include <boost/describe.hpp>
#include <boost/describe/detail/bases.hpp>
#include <boost/describe/detail/members.hpp>
#include <boost/endian/conversion.hpp>
#include <boost/mp11/algorithm.hpp>

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// A built equation on disk: the arena and the sweeps, not a frozen Graph, which
// freeze() rebuilds in one pass.  The colouring is in the file because
// coupling_pattern reads a Builder.
//
// Not Boost.Serialization: a compiled library whose only error channel is
// throwing, which aborts here.  Nothing below hands a byte to a container
// before the checksum and the invariants pass.
namespace ddx::rt {

// One lane's machine code, and what must agree before it runs; a mismatch means
// recompile, never an error.
struct Object {
  std::uint8_t want = 0; // Equation's Want: values, jacobian, hessian
  std::string symbol; // underivable, so adopt() is handed it back verbatim
  // jit::Compiler::host_identity(): triple, CPU, folded features, LLVM version.
  std::string host;
  std::uint64_t digest = 0; // digest() of the lane's frozen graph
  jit::Options options;
  std::vector<std::byte> code;
};

// All the C++ facade and PyEquation share, and so the only thing serialised.
template <impl::Numeric T> struct Snapshot {
  std::vector<std::string> symbols;
  std::vector<Node<T>> nodes;
  std::vector<NodeId> roots;
  Jacobian jacobian;
  std::vector<Hessian> hessians;
  jit::Options options;
  // The staleness key: rebuilding the model reproduces this prefix.
  std::uint32_t model_nodes = 0;
  std::vector<Object> objects;
};

} // namespace ddx::rt

// --- the described field lists ----------------------------------------------
//
// Here rather than beside each struct, so builder.hpp and its neighbours need
// nothing but ddx.  Folded into the schema stamp, so an unlisted field refuses
// old files.  Class templates cannot take BOOST_DESCRIBE_STRUCT.
namespace ddx::jit {
// Identity, not policy: two compiles differing only in `retain_object` or
// `cache_dir` emit the same machine code.
BOOST_DESCRIBE_STRUCT(Options, (),
                      (backend, points, lanes, opt_level, codegen_level, slp,
                       loop_vectorize, veclib, contract, time_passes))
} // namespace ddx::jit

namespace ddx::rt {

template <impl::Numeric T> BOOST_DESCRIBE_BASES(Node<T>, )
template <impl::Numeric T>
BOOST_DESCRIBE_PUBLIC_MEMBERS(Node<T>, op, a, b, value, slot)
template <impl::Numeric T> BOOST_DESCRIBE_PROTECTED_MEMBERS(Node<T>)
template <impl::Numeric T> BOOST_DESCRIBE_PRIVATE_MEMBERS(Node<T>)

template <impl::Numeric T> BOOST_DESCRIBE_BASES(Snapshot<T>, )
template <impl::Numeric T>
BOOST_DESCRIBE_PUBLIC_MEMBERS(Snapshot<T>, symbols, nodes, roots, jacobian,
                              hessians, options, model_nodes, objects)
template <impl::Numeric T> BOOST_DESCRIBE_PROTECTED_MEMBERS(Snapshot<T>)
template <impl::Numeric T> BOOST_DESCRIBE_PRIVATE_MEMBERS(Snapshot<T>)

BOOST_DESCRIBE_STRUCT(Jacobian, (), (value, partial, rows, columns))
BOOST_DESCRIBE_STRUCT(Hessian, (), (value, partial, compressed, coloring, zero))
BOOST_DESCRIBE_STRUCT(Coloring, (), (color, count, scatter, cell, cells))
BOOST_DESCRIBE_STRUCT(Object, (),
                      (want, symbol, host, digest, options, code))

// The described fields only: jit::Options::operator== compares the policy ones
// too, and a stored kernel must not be refused for those.
[[nodiscard]] inline bool same_codegen(const jit::Options &a,
                                       const jit::Options &b) {
  bool same = true;
  boost::mp11::mp_for_each<boost::describe::describe_members<
      jit::Options, boost::describe::mod_public>>(
      [&](auto D) { same = same && a.*D.pointer == b.*D.pointer; });
  return same;
}

// --- the file prologue ------------------------------------------------------

inline constexpr std::string_view graph_magic = "ddxgraph";
inline constexpr std::size_t header_bytes = 56;

// Little-endian and fixed-width throughout, so the graph half is portable;
// size_t is the one host type it leans on.
static_assert(sizeof(std::size_t) == 8);

// Public because the JIT's object cache writes behind the same prologue, with
// its own magic and `format`.
struct FileHeader {
  std::string_view magic = graph_magic;
  std::uint32_t format = 0;
  std::uint32_t schema = 0;
  std::uint16_t opcodes = 0; // label-table entries that follow the header
  std::uint8_t scalar_size = 0;
  std::uint8_t scalar_kind = 0; // 1 floating point, 0 otherwise
  std::uint32_t model_nodes = 0;
  std::uint64_t model_digest = 0;
  std::uint64_t payload_bytes = 0;
  std::uint32_t payload_crc = 0;
};

// Bumped by hand for what the schema stamp cannot see: a reordered section, a
// new prologue field, a change in coverage.  2 put the opcode table under the
// checksum; at 1 a flipped label remapped '+' to '/'.
inline constexpr std::uint32_t graph_format = 2;

DDX_API void put_header(const FileHeader &h, std::span<std::byte> into);
DDX_API result<FileHeader> get_header(std::span<const std::byte> bytes,
                                      std::string_view magic);
DDX_API std::uint32_t checksum(std::span<const std::byte> bytes);
DDX_API result<std::vector<std::byte>> read_file(const std::filesystem::path &);
// Staged beside the target and renamed over it: a reader never sees a
// half-written file, and a failed save leaves the old one standing.
DDX_API result<void> write_file(const std::filesystem::path &,
                                std::span<const std::byte>);

// Byte values are table-order, so appending a transcendental shifts every
// enumerator above it; the file names them by label and remaps on load.
DDX_API std::vector<std::string> opcode_labels();
DDX_API std::optional<OpCode> opcode_of(std::string_view label);

namespace detail {

// --- wire scalars -----------------------------------------------------------

// By size, not by name: uint32_t and unsigned are the same type on one platform
// and different on another.
template <std::size_t N>
using wire_uint_t = std::conditional_t<
    N == 1, std::uint8_t,
    std::conditional_t<
        N == 2, std::uint16_t,
        std::conditional_t<N == 4, std::uint32_t, std::uint64_t>>>;

template <typename U>
concept CWireScalar =
    std::integral<std::remove_cvref_t<U>> ||
    std::floating_point<std::remove_cvref_t<U>> ||
    std::is_enum_v<std::remove_cvref_t<U>>;

// The unsigned image of one scalar: an integer as itself, a float by its bits,
// an enum by its underlying type.
template <typename U>
[[nodiscard]] constexpr auto to_bits(const U &v) noexcept {
  using S = std::remove_cvref_t<U>;
  using W = wire_uint_t<sizeof(S)>;
  if constexpr (std::is_enum_v<S>) {
    return static_cast<W>(static_cast<std::underlying_type_t<S>>(v));
  } else if constexpr (std::floating_point<S>) {
    return std::bit_cast<W>(v);
  } else {
    return static_cast<W>(v);
  }
}

template <typename U>
[[nodiscard]] constexpr U from_bits(wire_uint_t<sizeof(U)> w) noexcept {
  if constexpr (std::is_enum_v<U>) {
    return static_cast<U>(static_cast<std::underlying_type_t<U>>(w));
  } else if constexpr (std::floating_point<U>) {
    return std::bit_cast<U>(w);
  } else {
    return static_cast<U>(w);
  }
}

// --- the two sinks ----------------------------------------------------------

class Writer {
public:
  explicit Writer(std::vector<std::byte> &into) noexcept : out_(&into) {}

  template <CWireScalar U> void scalar(const U &v) {
    const auto w = boost::endian::native_to_little(to_bits(v));
    const auto *const p = reinterpret_cast<const std::byte *>(&w);
    out_->insert(out_->end(), p, p + sizeof w);
  }

  void length(std::size_t n) { scalar(static_cast<std::uint64_t>(n)); }

  void text(const std::string &s) {
    length(s.size());
    const auto *const p = reinterpret_cast<const std::byte *>(s.data());
    out_->insert(out_->end(), p, p + s.size());
  }

  void blob(const std::vector<std::byte> &b) {
    length(b.size());
    out_->insert(out_->end(), b.begin(), b.end());
  }

  template <typename E> void array(const std::vector<E> &v) {
    length(v.size());
    const auto *const p = reinterpret_cast<const std::byte *>(v.data());
    out_->insert(out_->end(), p, p + v.size() * sizeof(E));
  }

  // A Writer cannot run short; it answers so one traversal drives both sinks.
  [[nodiscard]] static constexpr bool ok() noexcept { return true; }

private:
  std::vector<std::byte> *out_;
};

class Reader {
public:
  static constexpr std::uint8_t no_op = 0xFF;

  // `remap` turns the file's opcode byte into this build's, or no_op where the
  // file names one this build does not have.
  Reader(std::span<const std::byte> from,
         std::span<const std::uint8_t> remap) noexcept
      : in_(from), remap_(remap) {}

  void scalar(CWireScalar auto &v) {
    using S = std::remove_cvref_t<decltype(v)>;
    wire_uint_t<sizeof(S)> w{};
    if (!take(&w, sizeof w)) {
      return;
    }
    boost::endian::little_to_native_inplace(w);
    if constexpr (std::same_as<S, OpCode>) {
      const auto i = static_cast<std::size_t>(w);
      const std::uint8_t mapped = i < remap_.size() ? remap_[i] : no_op;
      if (mapped == no_op) {
        bad_ = true;
        return;
      }
      v = static_cast<OpCode>(mapped);
    } else {
      v = from_bits<S>(w);
    }
  }

  [[nodiscard]] std::size_t length() {
    std::uint64_t n = 0;
    scalar(n);
    return static_cast<std::size_t>(n);
  }

  void text(std::string &s) {
    const auto n = length();
    if (!ok() || !credible(n, 1)) {
      bad_ = true;
      return;
    }
    s.resize(n);
    (void)take(s.data(), n);
  }

  void blob(std::vector<std::byte> &b) {
    const auto n = length();
    if (!ok() || !credible(n, 1)) {
      bad_ = true;
      return;
    }
    b.resize(n);
    (void)take(b.data(), n);
  }

  template <typename E> void array(std::vector<E> &v) {
    const auto n = length();
    if (!ok() || !credible(n, sizeof(E))) {
      bad_ = true;
      return;
    }
    v.resize(n);
    (void)take(v.data(), n * sizeof(E)); // credible() bounded the product
  }

  // Before every resize: an enormous length allocates wildly, and there is no
  // bad_alloc to catch.
  template <typename V> void resize(V &v, std::size_t n) {
    if (!credible(n, element_floor<typename V::value_type>())) {
      bad_ = true;
      return;
    }
    v.resize(n);
  }

  [[nodiscard]] bool ok() const noexcept { return !bad_; }
  [[nodiscard]] bool spent() const noexcept { return in_.empty(); }
  [[nodiscard]] std::size_t remaining() const noexcept { return in_.size(); }

private:
  // The fewest bytes an element can occupy: a string still costs its length.
  template <typename U> [[nodiscard]] static constexpr std::size_t
  element_floor() noexcept {
    if constexpr (CWireScalar<U>) {
      return sizeof(wire_uint_t<sizeof(U)>);
    } else {
      return sizeof(std::uint64_t);
    }
  }

  [[nodiscard]] bool credible(std::size_t n, std::size_t element) const {
    return element == 0 || n <= in_.size() / element;
  }

  bool take(void *into, std::size_t n) {
    if (bad_ || in_.size() < n) {
      bad_ = true;
      return false;
    }
    std::memcpy(into, in_.data(), n);
    in_ = in_.subspan(n);
    return true;
  }

  std::span<const std::byte> in_;
  std::span<const std::uint8_t> remap_;
  bool bad_ = false;
};

template <typename V> constexpr bool is_vector_v = false;
template <typename U, typename A>
constexpr bool is_vector_v<std::vector<U, A>> = true;

// Whether a vector moves in one memcpy, which the colouring's colours * n tables
// are what make worth having.  Same bytes either way, so a big-endian host falls
// through to the loop and reads the same file.
template <typename E>
constexpr bool bulk_v =
    CWireScalar<E> && !std::same_as<E, OpCode> && !std::same_as<E, bool> &&
    std::is_trivially_copyable_v<E> && sizeof(E) == sizeof(wire_uint_t<sizeof(E)>) &&
    std::endian::native == std::endian::little;

// --- the one traversal ------------------------------------------------------

// A described aggregate is its fields in order, a vector its length then its
// elements, a string and a blob their own length, everything else a scalar.
// Both sinks answer the same calls, so there is no second implementation to
// drift from; `V` is const on the writing pass only.
template <typename Sink, typename V> void wire(Sink &s, V &v) {
  using U = std::remove_const_t<V>;
  if constexpr (std::same_as<U, std::string> ||
                std::same_as<U, std::vector<std::byte>>) {
    if constexpr (std::same_as<U, std::string>) {
      s.text(v);
    } else {
      s.blob(v);
    }
  } else if constexpr (CWireScalar<U>) {
    s.scalar(v);
  } else if constexpr (is_vector_v<U>) {
    // Nested, not one `&&`: the bulk test names U::value_type and both
    // arguments would be instantiated.
    if constexpr (bulk_v<typename U::value_type>) {
      s.array(v);
    } else {
      if constexpr (std::is_const_v<V>) {
        s.length(v.size());
      } else {
        s.resize(v, s.length());
        if (!s.ok()) {
          return;
        }
      }
      for (auto &e : v) {
        wire(s, e);
        if (!s.ok()) {
          return;
        }
      }
    }
  } else {
    static_assert(boost::describe::has_describe_members<U>::value,
                  "rt::archive: no described field list for this type");
    boost::mp11::mp_for_each<
        boost::describe::describe_members<U, boost::describe::mod_public>>(
        [&](auto D) {
          if (s.ok()) {
            wire(s, v.*D.pointer);
          }
        });
  }
}

// --- the schema stamp -------------------------------------------------------

constexpr void fold32(std::uint32_t &h, std::string_view s) noexcept {
  for (const char c : s) {
    h = (h ^ static_cast<unsigned char>(c)) * 16777619u;
  }
}

// Field names and leaf widths folded in traversal order and carried in the
// prologue, so a moved or retyped field refuses old files.
template <typename V> constexpr void fold_schema(std::uint32_t &h) {
  using U = std::remove_const_t<V>;
  if constexpr (std::same_as<U, std::string>) {
    fold32(h, "str");
  } else if constexpr (std::same_as<U, std::vector<std::byte>>) {
    fold32(h, "blob");
  } else if constexpr (CWireScalar<U>) {
    fold32(h, std::floating_point<U> ? "f" : "u");
    h = (h ^ static_cast<std::uint32_t>(sizeof(wire_uint_t<sizeof(U)>))) *
        16777619u;
  } else if constexpr (is_vector_v<U>) {
    fold32(h, "vec<");
    fold_schema<typename U::value_type>(h);
    fold32(h, ">");
  } else {
    static_assert(boost::describe::has_describe_members<U>::value);
    fold32(h, "{");
    boost::mp11::mp_for_each<
        boost::describe::describe_members<U, boost::describe::mod_public>>(
        [&](auto D) {
          fold32(h, D.name);
          fold32(h, ":");
          fold_schema<std::remove_reference_t<
              decltype(std::declval<U &>().*D.pointer)>>(h);
          fold32(h, ";");
        });
    fold32(h, "}");
  }
}

template <typename V> [[nodiscard]] constexpr std::uint32_t schema_of() {
  std::uint32_t h = 2166136261u;
  fold_schema<V>(h);
  return h;
}

// --- what a loaded snapshot has to satisfy ----------------------------------

// What the interpreter, the liveness walk and codegen rely on and none re-tests:
// until a file, nothing could build a graph that broke one.
template <impl::Numeric T>
[[nodiscard]] result<void> sound(const Snapshot<T> &s) {
  const auto n = s.nodes.size();
  const auto nsym = s.symbols.size();
  const auto in_range = [n](NodeId v) { return v < n; };

  if (!std::ranges::is_sorted(s.symbols) ||
      std::ranges::adjacent_find(s.symbols) != s.symbols.end()) {
    return fail(errc::archive_corrupt); // var() keeps them sorted and unique
  }

  for (const auto [i, node] : s.nodes | std::views::enumerate) {
    const auto arity = arity_of(node.op);
    // Topological: every operand strictly below its reader, which is what the
    // runtime single-passes on.
    const bool a_ok = arity >= 1 ? node.a < static_cast<NodeId>(i)
                                 : node.a == no_node;
    const bool b_ok = arity == 2 ? node.b < static_cast<NodeId>(i)
                                 : node.b == no_node;
    if (!a_ok || !b_ok) {
      return fail(errc::archive_corrupt);
    }
    if (node.op == OpCode::Var && node.slot >= nsym) {
      return fail(errc::archive_corrupt);
    }
  }

  // Builder::restore walks for them, so a symbol named by no Var leaves a
  // no_node for var() to hand back.
  std::vector<std::uint32_t> named(nsym, 0);
  for (const auto &node : s.nodes) {
    if (node.op == OpCode::Var) {
      ++named[node.slot];
    }
  }
  if (!std::ranges::all_of(named, [](std::uint32_t k) { return k == 1; })) {
    return fail(errc::archive_corrupt);
  }

  if (!std::ranges::all_of(s.roots, in_range) ||
      !std::ranges::all_of(s.jacobian.value, in_range) ||
      !std::ranges::all_of(s.jacobian.partial, in_range)) {
    return fail(errc::archive_corrupt);
  }
  // Load-bearing order: `||` short-circuits, so `rows` and `columns` are pinned
  // before anything multiplies them, the product otherwise being two unchecked
  // payload scalars that can wrap into agreeing.
  if (s.jacobian.rows != s.roots.size() || s.jacobian.columns != nsym ||
      s.jacobian.partial.size() != s.jacobian.rows * s.jacobian.columns) {
    return fail(errc::archive_corrupt);
  }
  // One per root, or none: a constant-evaluated equation never swept them.
  if (!s.hessians.empty() && s.hessians.size() != s.roots.size()) {
    return fail(errc::archive_corrupt);
  }

  for (const auto &h : s.hessians) {
    const auto &c = h.coloring;
    if (!in_range(h.value) || !in_range(h.zero) ||
        !std::ranges::all_of(h.partial, in_range) ||
        !std::ranges::all_of(h.compressed, in_range)) {
      return fail(errc::archive_corrupt);
    }
    // `count` comes straight off the payload, so `count * nsym` is checked by
    // division -- the product is where a forged count wraps into agreeing.  A
    // checksum detects accidents, it does not confer trust.
    const auto sized = [nsym, &c](const std::vector<std::size_t> &v) {
      // Colours over no symbols are no colours; nothing else pins `count`.
      return nsym == 0 ? v.empty() && c.count == 0
                       : c.count <= v.size() / nsym && v.size() == c.count * nsym;
    };
    if (c.color.size() != nsym || !sized(c.scatter) || !sized(c.cell) ||
        h.partial.size() != nsym) {
      return fail(errc::archive_corrupt);
    }
    if (!std::ranges::all_of(c.color, [&c](std::size_t k) {
          return k < c.count;
        })) {
      return fail(errc::archive_corrupt);
    }
    // Every cell must name a slot inside the compressed block: two sharing one
    // would sum two second derivatives into a single column.  Pinned first.
    const auto owned = std::ranges::count_if(
        c.cell, [](std::size_t k) { return k != no_column; });
    if (c.cells != static_cast<std::size_t>(owned) ||
        h.compressed.size() != c.cells ||
        !std::ranges::all_of(c.cell,
                             [&c](std::size_t k) {
                               return k == no_column || k < c.cells;
                             }) ||
        !std::ranges::all_of(c.scatter, [nsym](std::size_t k) {
          return k == no_column || k < nsym;
        })) {
      return fail(errc::archive_corrupt);
    }
  }
  return {};
}

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

// --- digests ----------------------------------------------------------------

inline constexpr std::uint64_t fnv64_seed = 1469598103934665603ull;

constexpr void fold64(std::uint64_t &h, std::uint64_t w) noexcept {
  for (int i = 0; i < 8; ++i) {
    h = (h ^ ((w >> (i * 8)) & 0xFF)) * 1099511628211ull;
  }
}

// The model as the file keys on it: symbols, then the arena up to `upto`.
// Field by field, never a memcpy -- Node<double> has interior padding.
template <impl::Numeric T>
[[nodiscard]] std::uint64_t digest(std::span<const std::string> symbols,
                                   std::span<const Node<T>> nodes,
                                   std::size_t upto) {
  std::uint64_t h = fnv64_seed;
  fold64(h, symbols.size());
  for (const auto &s : symbols) {
    fold64(h, s.size());
    for (const char c : s) {
      fold64(h, static_cast<unsigned char>(c));
    }
  }
  fold64(h, upto);
  for (const auto &n : nodes.first(std::min(upto, nodes.size()))) {
    fold64(h, static_cast<std::uint64_t>(n.op));
    fold64(h, n.a);
    fold64(h, n.b);
    fold64(h, detail::to_bits(n.value));
    fold64(h, n.slot);
  }
  return h;
}

// Keyed on what codegen reads of a lane, so a consumer sees a compile is done
// *before* emitting a module.  Ids are construction-ordered, making this a
// within-one-binary key, so a stale one must be a miss.
template <impl::Numeric T>
[[nodiscard]] std::uint64_t digest(const Graph<T> &g) {
  std::uint64_t h = fnv64_seed;
  const auto &layout = g.layout();
  fold64(h, g.symbols().size());
  fold64(h, layout.values);
  fold64(h, layout.jacobian);
  fold64(h, layout.hessian);
  const auto order = g.contracted_order();
  fold64(h, order.size());
  for (const NodeId v : order) {
    const auto &p = g[v];
    const auto [a, b] = g.operands(v);
    fold64(h, v);
    fold64(h, static_cast<std::uint64_t>(p.op));
    fold64(h, p.slot);
    fold64(h, detail::to_bits(p.value));
    fold64(h, a);
    fold64(h, b);
  }
  for (const NodeId o : g.outputs()) {
    fold64(h, o);
  }
  return h;
}

// --- save / load / verify ---------------------------------------------------

template <std::floating_point T>
[[nodiscard]] result<void> save(const Snapshot<T> &snap,
                                const std::filesystem::path &path) {
  std::vector<std::byte> payload;
  payload.reserve(snap.nodes.size() * sizeof(Node<T>) + 1024);
  detail::Writer w{payload};
  detail::wire(w, std::as_const(snap));

  const auto labels = opcode_labels();
  std::vector<std::byte> table;
  detail::Writer tw{table};
  std::ranges::for_each(labels, [&tw](const std::string &l) { tw.text(l); });

  // Together: the table is what every opcode byte *means*, so outside the
  // checksum one flipped bit reinterprets the graph.
  std::vector<std::byte> body = table;
  body.insert(body.end(), payload.begin(), payload.end());

  const FileHeader h{
      .magic = graph_magic,
      .format = graph_format,
      .schema = detail::schema_of<Snapshot<T>>(),
      .opcodes = static_cast<std::uint16_t>(labels.size()),
      .scalar_size = sizeof(T),
      .scalar_kind = 1,
      .model_nodes = snap.model_nodes,
      .model_digest =
          digest<T>(snap.symbols, snap.nodes, snap.model_nodes),
      .payload_bytes = payload.size(),
      .payload_crc = checksum(body)};

  std::vector<std::byte> file(header_bytes);
  put_header(h, file);
  file.insert(file.end(), body.begin(), body.end());
  return write_file(path, file);
}

// The prologue this build would write, against the one the file carries.
template <std::floating_point T>
[[nodiscard]] result<FileHeader> compatible(std::span<const std::byte> bytes) {
  auto head = get_header(bytes, graph_magic);
  if (!head) {
    return head;
  }
  if (head->format != graph_format ||
      head->schema != detail::schema_of<Snapshot<T>>() ||
      head->scalar_size != sizeof(T) || head->scalar_kind != 1) {
    return fail(errc::bad_archive);
  }
  return head;
}

// Not `load`: that is rt::load<T, Outputs>(), which answers with an Equation.
template <std::floating_point T = double>
[[nodiscard]] result<Snapshot<T>>
load_snapshot(const std::filesystem::path &path) {
  auto bytes = read_file(path);
  if (!bytes) {
    return std::unexpected{bytes.error()};
  }
  auto head = compatible<T>(*bytes);
  if (!head) {
    return std::unexpected{head.error()};
  }

  // Before a byte of either is parsed: corruption in the table changes the
  // graph's meaning rather than breaking it.
  std::span<const std::byte> body = std::span{*bytes}.subspan(header_bytes);
  if (checksum(body) != head->payload_crc) {
    return fail(errc::archive_corrupt);
  }

  // The label table, then its inverse against this build's enum.
  std::span<const std::byte> rest = body;
  std::vector<std::uint8_t> remap(head->opcodes, detail::Reader::no_op);
  {
    detail::Reader tr{rest, {}};
    for (auto &slot : remap) {
      std::string label;
      tr.text(label);
      if (!tr.ok()) {
        return fail(errc::archive_corrupt);
      }
      if (const auto op = opcode_of(label)) {
        slot = static_cast<std::uint8_t>(*op);
      }
    }
    rest = rest.last(tr.remaining()); // the payload is what the table left
  }

  if (rest.size() != head->payload_bytes) {
    return fail(errc::archive_corrupt);
  }

  Snapshot<T> snap;
  detail::Reader r{rest, remap};
  detail::wire(r, snap);
  if (!r.ok() || !r.spent()) {
    return fail(errc::archive_corrupt);
  }

  // Only now is any of it trusted.
  if (const auto why = detail::sound(snap); !why) {
    return std::unexpected{why.error()};
  }
  // The one region a checksum cannot cover, so every field is verified and none
  // is written that is not.  `model_nodes` is duplicated under the checksum and
  // would otherwise be written, never read, and one edit from being trusted.
  if (head->model_nodes != snap.model_nodes ||
      snap.model_nodes > snap.nodes.size() ||
      digest<T>(snap.symbols, snap.nodes, snap.model_nodes) !=
          head->model_digest) {
    return fail(errc::archive_corrupt);
  }
  return snap;
}

// Whether this build can read the file at all: prologue, checksum, structural
// invariants.  *Which* equation it holds is Equation::verify().
template <std::floating_point T = double>
[[nodiscard]] result<void> verify(const std::filesystem::path &path) {
  return load_snapshot<T>(path).transform([](const Snapshot<T> &) {});
}

} // namespace ddx::rt
