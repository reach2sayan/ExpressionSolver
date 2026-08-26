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

// A built equation, on disk.  What is saved is the *arena and the sweeps*, not
// a frozen Graph: rt::evaluate_* all take a Builder, the interpreter's tape is
// sized by the arena, and a Graph's CSR, liveness and contracted order are all
// derived -- freeze() rebuilds them in one linear pass.  What cannot be
// recovered from a Graph is the Hessian colouring, since coupling_pattern reads
// a Builder, so that is in the file.
//
// Boost.Serialization is not used, and cannot be: it is a compiled library
// where every other Boost here is header-only, and its sole error channel is
// throwing, which src/rt/boost_no_exceptions.cpp turns into abort().  Nothing
// below hands a byte to a container before the checksum and the structural
// invariants have passed, because there is no throw to catch.
namespace ddx::rt {

// Machine code for one lane, and everything that has to agree before it may be
// run.  Best-effort throughout: a mismatch means recompile, never an error.
struct Object {
  std::uint8_t want = 0; // Equation's Want: values, jacobian, hessian
  // The name the code is linked under.  Chosen by the compile and derivable
  // from nothing, so adopt() has to be handed it back verbatim.
  std::string symbol;
  // jit::Compiler::host_identity() -- triple, CPU, folded feature set and LLVM
  // version, as one string so a miss can be explained rather than merely
  // detected.
  std::string host;
  std::uint64_t digest = 0; // digest() of the lane's frozen graph
  jit::Options options;
  std::vector<std::byte> code;
};

// The whole of what two equations -- the C++ facade and PyEquation -- have in
// common, and therefore the only thing this file serialises.  Both fill one in
// to save and consume one to load; the serialiser is written once, here.
template <impl::Numeric T> struct Snapshot {
  std::vector<std::string> symbols;
  std::vector<Node<T>> nodes;
  std::vector<NodeId> roots;
  Jacobian jacobian;
  std::vector<Hessian> hessians;
  jit::Options options;
  // The arena as it stood before the sweeps appended to it.  The staleness
  // key: a caller who rebuilds the model gets the same prefix back, and the
  // sweeps are exactly what it does not want to redo.
  std::uint32_t model_nodes = 0;
  std::vector<Object> objects;
};

} // namespace ddx::rt

// --- the described field lists ----------------------------------------------
//
// Here rather than beside each struct, so builder.hpp, derivative.hpp and
// coupling.hpp keep needing nothing but ddx itself -- the layering
// include/rt/CMakeLists.txt states.  A field added to one of them and not
// listed here is safe rather than silent: these names are folded into the
// schema stamp, so a changed struct refuses old files instead of misreading
// them.  Class templates cannot take BOOST_DESCRIBE_STRUCT, hence the three
// member macros spelled out.
namespace ddx::jit {
// Identity, not policy.  `retain_object` and `cache_dir` are deliberately
// absent: two compiles differing only in whether the bytes were kept, or in
// where a cache lives, produce the same machine code, so neither may enter a
// key that decides whether stored code may be run.  Everything that does
// change the code is here.
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

// Equal in everything that decides what machine code a compile emits, which is
// the described fields above -- and so deliberately not `retain_object` or
// `cache_dir`.  jit::Options::operator== compares those two as well, and a
// stored kernel must not be refused because the caller has since asked for the
// bytes to be kept, or pointed a cache somewhere else.  The describe list is
// the definition of identity here, and this is what reads it.
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

// Every integer on the wire is little-endian and fixed-width, so the graph half
// of a file is portable.  size_t is the one host type the format leans on, and
// the tree is -march=x86-64-v3 or better, so it is pinned rather than encoded.
static_assert(sizeof(std::size_t) == 8);

// The fixed prologue of a ddx on-disk artefact.  Public because the JIT's
// object cache writes its own entries behind the same prologue with its own
// magic and its own `format` counter -- one convention, two producers.
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

// Bumped by hand when the layout changes in a way the schema stamp cannot see
// -- a reordered section, a new prologue field, or a change in what the
// checksum covers.  2: the checksum covers the opcode table as well as the
// payload.  At 1 it did not, and a single-bit flip in a label silently remapped
// an opcode: '+' is 0x2B and '/' is 0x2F, one bit apart and the same arity, so
// every Add in a file became a Div, `sound()` saw nothing wrong, and a gradient
// came back quietly wrong.
inline constexpr std::uint32_t graph_format = 2;

DDX_API void put_header(const FileHeader &h, std::span<std::byte> into);
DDX_API result<FileHeader> get_header(std::span<const std::byte> bytes,
                                      std::string_view magic);
DDX_API std::uint32_t checksum(std::span<const std::byte> bytes);
DDX_API result<std::vector<std::byte>> read_file(const std::filesystem::path &);
// Written beside the target and renamed over it, so a reader never sees a
// half-written file and a failed save leaves the old one standing.
DDX_API result<void> write_file(const std::filesystem::path &,
                                std::span<const std::byte>);

// This build's opcode labels, and the inverse a loader needs.  OpCode's byte
// values are table-order (rt/opcode.hpp), so appending a transcendental shifts
// every enumerator above it: the file stores labels and remaps on load, which
// is what keeps old files readable across exactly that change.
DDX_API std::vector<std::string> opcode_labels();
DDX_API std::optional<OpCode> opcode_of(std::string_view label);

namespace detail {

// --- wire scalars -----------------------------------------------------------

// Fixed width by size, not by name: uint32_t and unsigned are the same type on
// one platform and different on another, so naming them would be a redefinition
// here and a gap there.
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

  // A Writer cannot run short.  It answers anyway, because that is what lets
  // one traversal drive both sinks.
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

  template <CWireScalar U> void scalar(U &v) {
    using S = std::remove_cvref_t<U>;
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

  // A length the remaining bytes could not possibly back.  Checked before every
  // resize: a plausible-but-enormous length is the one corruption that would
  // otherwise allocate wildly, and there is no bad_alloc to catch here.
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
  // The fewest bytes one element can occupy on the wire.  A string or a vector
  // still costs its own length field, so nothing is free.
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

// Whether a whole vector of these is the same bytes as the elements one at a
// time -- in which case it moves in one memcpy rather than one per element.
// The colouring is why this is worth having: `scatter` and `cell` are each
// colours * n of them, so a graph over 256 symbols carries a megabyte that
// would otherwise cross the boundary eight bytes at a time.
//
// The file does not change: this is the same little-endian, fixed-width layout
// the element loop writes, which is why a big-endian host may simply fall
// through to that loop and still read the same file.
template <typename E>
constexpr bool bulk_v =
    CWireScalar<E> && !std::same_as<E, OpCode> && !std::same_as<E, bool> &&
    std::is_trivially_copyable_v<E> && sizeof(E) == sizeof(wire_uint_t<sizeof(E)>) &&
    std::endian::native == std::endian::little;

// --- the one traversal ------------------------------------------------------

// A described aggregate is its fields in order, a vector is its length then its
// elements, a string and a blob carry their own length, and everything else is
// a fixed-width scalar.  Both sinks answer the same calls, so this is written
// once and read both ways -- which is the whole reason the format has no second
// implementation to drift from.
//
// `V` carries const on the writing pass and not on the reading one; the two
// differ only in whether a vector's length is written or resized to.
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
    // Nested rather than one `&&`: the bulk test names U::value_type, which a
    // non-vector U does not have, and a template argument is instantiated
    // whether or not the left of the && already settled it.
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

// Every described field name and every leaf's wire width, folded in traversal
// order.  A renamed, reordered, retyped or added field changes this, and the
// prologue carries it, so a file written before the change is refused rather
// than read as though nothing had moved.
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

// Every check here is an invariant that the interpreter, the liveness walk and
// codegen all rely on and none of them re-tests, because until now nothing
// could build a graph that broke one.  tests/rt/tests_rt_graph.cpp asserts the
// same set on a freshly frozen graph.  A template rather than an out-of-line
// DDX_API function because it needs the node array and drags in no Boost.Graph
// -- which is the only reason coupling.cpp and dot.cpp are compiled at all.
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
    // Ids are topological -- every operand strictly below its reader.  This is
    // the one invariant the whole runtime single-passes on.
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

  // Interning leaves exactly one Var node per symbol, and Builder::restore
  // rebuilds its slot-addressed vars_ by walking for them -- a symbol named by
  // none would leave a no_node there for var() to hand back later.
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
  // The order here is load-bearing, not stylistic: `||` short-circuits, so
  // `rows` and `columns` are pinned to lengths this file actually has before
  // anything multiplies them.  Reversed, the product would be two unchecked
  // payload scalars and could wrap into agreeing with `partial.size()` -- which
  // is the hole the colouring's `count` had.  Leave them first.
  if (s.jacobian.rows != s.roots.size() || s.jacobian.columns != nsym ||
      s.jacobian.partial.size() != s.jacobian.rows * s.jacobian.columns) {
    return fail(errc::archive_corrupt);
  }
  // A Hessian per root, or none at all: an equation constant-evaluated its way
  // past the sweeps, and a loaded one may not have been asked for them.
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
    // `count` is read straight from the payload and is bounded by nothing yet,
    // so `count * nsym` is checked by division first: the product is exactly
    // where a forged count wraps and agrees with a length it has no business
    // agreeing with.  Reachable only by forgery -- the checksum stops every
    // damaged file long before this -- which is the point.  A checksum detects
    // accidents; it does not make the payload trustworthy, and everything the
    // payload is then indexed by has to be checked as though it were hostile.
    const auto sized = [nsym, &c](const std::vector<std::size_t> &v) {
      // Colours over no symbols are no colours; the tables are empty either
      // way, so nothing would otherwise pin `count` at all.
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
    // `cells` is the compressed block's width, and every cell a column owns
    // must name a slot inside it -- two cells sharing one would sum two
    // second derivatives into a single output column.
    // Same order dependency: `cells` is pinned against a count over `cell`
    // before anything indexes with it.
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

// Builder's side of the loader.  A struct rather than a friended function
// template, so builder.hpp forward-declares an empty type and gains no include
// for it.
struct Restore {
  template <impl::Numeric T>
  static void into(Builder<T> &b, std::vector<Node<T>> nodes,
                   std::vector<std::string> symbols) {
    b.restore(std::move(nodes), std::move(symbols));
  }
};

} // namespace detail

// The arena a snapshot describes.  Owning, because a loaded equation has no
// caller's arena to borrow and nothing else refers to this one.  The snapshot
// is consumed: its node array *is* the arena's.
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

// The model as the file keys on it: the symbols, then the arena up to `upto`.
// Field by field, never a memcpy of the node -- Node<double> is 24 bytes with
// interior padding, so a struct blit is not reproducible.
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

// One frozen lane, keyed on exactly what codegen reads of it -- so a consumer
// can tell that a compile is already done *before* emitting a module, rather
// than after.  Node ids are assigned in construction order, so this is a
// within-one-binary key: gcc and clang number the same graph differently, and
// so would two ddx builds that construct it differently.  A stale key must
// therefore be a miss, never a reinterpretation.
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

  // The table and the payload are checksummed together.  The table is not
  // decoration: it is what every opcode byte in the payload *means*, so leaving
  // it outside the checksum let one flipped bit reinterpret the whole graph.
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

// Not spelled `load`: rt::load<T, Outputs>() is the one a caller reaches for,
// and it answers with an Equation.  This is the layer under it.
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

  // The checksum first, over the table and the payload together, and before a
  // single byte of either is parsed.  The old order read the table out of
  // unverified bytes -- and since the table decides what every opcode in the
  // payload means, that was the one section where corruption changed the
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
  // The prologue is the one region a checksum cannot cover, because the
  // checksum lives in it -- so every field it carries is verified here, field
  // by field, and no field is written that is not.  `model_nodes` is duplicated
  // in the payload under the checksum and would otherwise be inert: written,
  // never read, and one edit away from being read and trusted.  Inert but
  // unprotected is the state a field is in immediately before it is protected
  // and wrong.
  if (head->model_nodes != snap.model_nodes ||
      snap.model_nodes > snap.nodes.size() ||
      digest<T>(snap.symbols, snap.nodes, snap.model_nodes) !=
          head->model_digest) {
    return fail(errc::archive_corrupt);
  }
  return snap;
}

// Whether this file is one this build can read at all: the prologue, the
// checksum and the structural invariants.  It says nothing about *which*
// equation it holds -- Equation::verify() answers that.
template <std::floating_point T = double>
[[nodiscard]] result<void> verify(const std::filesystem::path &path) {
  return load_snapshot<T>(path).transform([](const Snapshot<T> &) {});
}

} // namespace ddx::rt
