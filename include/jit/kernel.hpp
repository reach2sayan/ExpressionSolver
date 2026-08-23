#pragma once

#include "symbolic/expressions.hpp" // ddx::impl::Numeric
#include "util/error.hpp"
#include "util/export.hpp"
#include "util/pinned.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ddx::rt {
// The JIT emits machine types, so only a graph over a machine scalar compiles.
// The constraint has to match the definition, hence the one ddx include.
template <impl::Numeric T> class Graph;
} // namespace ddx::rt

namespace ddx::jit {

// Which vector math library the loop vectoriser may call.  Off by default:
// glibc's vector routines are documented to ~4 ULP where the scalar ones are
// ~0.5, and a kernel is checked against the interpreter, which calls the scalar
// ones.  Asking for one is asking for that trade.
enum class VecLib : std::uint8_t { None, Auto, Libmvec };

// Lean drops the two vectorisers.  On a large body the loop vectoriser is 95%
// of the pass pipeline and then declines to vectorise -- dropping it took one
// measured graph from 17.6s to 0.3s at no cost to throughput.  On a small one
// it succeeds and is worth 1.5-2.2x, so Auto keeps it there and errs high:
// choosing Default wrongly costs compile time, the reverse costs kernel speed.
enum class Pipeline : std::uint8_t { Auto, Default, Lean };

// Carries text where the library's other errors do not: LLVM's are not one of
// a fixed set.
struct error {
  errc code;
  std::string detail;
};

template <typename T> using result = std::expected<T, error>;

#ifndef DDX_JIT_DEFAULT_OPT
#define DDX_JIT_DEFAULT_OPT 2
#endif
#ifndef DDX_JIT_DEFAULT_CONTRACT
#define DDX_JIT_DEFAULT_CONTRACT 1
#endif

inline constexpr unsigned default_opt_level = DDX_JIT_DEFAULT_OPT;
inline constexpr bool default_contract = DDX_JIT_DEFAULT_CONTRACT != 0;

struct Options {
  // Never 0: that disables the loop vectoriser
  unsigned opt_level = default_opt_level;
  VecLib veclib = VecLib::None;
  bool contract = default_contract; // Follows DDX_FP_FLAGS
  // Per-pass timing to stderr, through LLVM's own TimePassesHandler.  Off by
  // default: it is a diagnostic for whoever is asking why a compile is slow,
  // and it prints where nothing else in the library does.
  bool time_passes = false;
  Pipeline pipeline = Pipeline::Auto;
  // Where Auto turns over, in IR instructions after emission.
  std::size_t lean_above = 2000;

  // Emit as slabs above this many live nodes; 0 never splits.  Each is
  // allocated and compiled on its own, at the price of crossing values going
  // through memory.  Off by default: it measured as a loss at every size
  // reachable in reasonable time.
  std::size_t split_above = 0;
  // Slab size as a multiple of the peak wavefront, which bounds traffic per
  // node at 2 / slab_factor whatever the graph looks like.
  std::size_t slab_factor = 10;
  std::size_t min_slab = 4096;
};

// Where a compile spent its time.  `codegen` brackets the symbol lookup too:
// LLJIT compiles lazily, so the lookup is what forces the backend to run.
struct CompileReport {
  std::size_t nodes = 0;        // live nodes in the graph
  std::size_t instructions = 0; // IR instructions after emission
  std::chrono::nanoseconds emit{};
  std::chrono::nanoseconds optimize{};
  std::chrono::nanoseconds codegen{};
  std::size_t slabs = 1;         // modules emitted; more than one when split
  std::size_t scratch_slots = 0; // values crossing a slab boundary
};

// One compiled graph.  A copy is one atomic increment: a Kernel keeps the JIT
// that owns its code alive, and dropping the last one frees the code.
class Kernel {
public:
  using function_type = void (*)(const double *const *, double *const *,
                                 double *const *, double *const *, std::size_t);
  // The same, plus the scratch a slab reads its inputs from and leaves its
  // outputs in.
  using slab_type = void (*)(const double *const *, double *const *,
                             double *const *, double *const *, double *,
                             std::size_t, std::size_t);

  // Scratch is slots * block, so the block is what keeps it in cache --
  // whole-batch scratch on a large graph is tens of megabytes.
  static constexpr std::size_t kScratchBudget = 1u << 18; // 256 KiB

  Kernel() = default;
  Kernel(function_type fn, std::size_t arity, std::size_t values,
         std::size_t jacobian, std::size_t hessian,
         std::shared_ptr<void> code) noexcept
      : fn_(fn), code_(std::move(code)), arity_(arity), values_(values),
        jacobian_(jacobian), hessian_(hessian) {}

  Kernel(std::vector<slab_type> slabs, std::size_t slots, std::size_t arity,
         std::size_t values, std::size_t jacobian, std::size_t hessian,
         std::shared_ptr<void> code) noexcept
      : code_(std::move(code)), slabs_(std::move(slabs)), slots_(slots),
        block_(slots == 0
                   ? std::size_t{256}
                   : std::clamp(kScratchBudget / (slots * sizeof(double)),
                                std::size_t{8}, std::size_t{256})),
        arity_(arity), values_(values), jacobian_(jacobian), hessian_(hessian) {
  }

  // xs[j] is the column for symbol j, g[j] the partial in it, each of length n;
  // a block that was not requested is `{}`.  noexcept because codegen marks the
  // kernel and every libm declaration it calls nounwind.
  void operator()(std::span<const double *const> xs, std::span<double *const> f,
                  std::span<double *const> g, std::span<double *const> h,
                  std::size_t n) const noexcept {
    if (slabs_.empty()) {
      fn_(xs.data(), f.data(), g.data(), h.data(), n);
      return;
    }
    // One buffer per thread, so a split kernel stays as callable as an unsplit
    // one; a caller who cannot allocate here passes its own below.
    static thread_local std::vector<double> scratch;
    if (scratch.size() < scratch_size(n)) {
      scratch.resize(scratch_size(n));
    }
    (*this)(xs, f, g, h, n, scratch);
  }

  // Every slab over one block, then the next: crossings stay hot in scratch,
  // and a slab's code is touched once per block rather than once per point.
  void operator()(std::span<const double *const> xs, std::span<double *const> f,
                  std::span<double *const> g, std::span<double *const> h,
                  std::size_t n, std::span<double> scratch) const noexcept {
    if (slabs_.empty()) {
      fn_(xs.data(), f.data(), g.data(), h.data(), n);
      return;
    }
    for (std::size_t base = 0; base < n; base += block_) {
      const std::size_t len = std::min(block_, n - base);
      for (const slab_type slab : slabs_) {
        slab(xs.data(), f.data(), g.data(), h.data(), scratch.data(), base,
             len);
      }
    }
  }

  // What the overload above needs for a batch of n, in doubles; zero unsplit.
  [[nodiscard]] std::size_t scratch_size(std::size_t n) const noexcept {
    return slots_ * std::min(block_, n);
  }
  [[nodiscard]] std::size_t block() const noexcept { return block_; }
  [[nodiscard]] std::size_t slabs() const noexcept { return slabs_.size(); }

  [[nodiscard]] explicit operator bool() const noexcept {
    return fn_ != nullptr || !slabs_.empty();
  }
  [[nodiscard]] std::size_t arity() const noexcept { return arity_; }
  [[nodiscard]] std::size_t values() const noexcept { return values_; }
  [[nodiscard]] std::size_t jacobian_columns() const noexcept {
    return jacobian_;
  }
  [[nodiscard]] std::size_t hessian_columns() const noexcept {
    return hessian_;
  }
  [[nodiscard]] std::size_t outputs() const noexcept {
    return values_ + jacobian_ + hessian_;
  }

private:
  function_type fn_ = nullptr;
  std::shared_ptr<void> code_; // Held, never read; operator() calls through fn_
  std::vector<slab_type> slabs_;
  std::size_t slots_ = 0;
  std::size_t block_ = 1;
  std::size_t arity_ = 0;
  std::size_t values_ = 0;
  std::size_t jacobian_ = 0;
  std::size_t hessian_ = 0;
};

// A Compiler *is* the LLJIT, and every Kernel shares it: one going out of scope
// frees nothing still callable, and reclaims nothing either.
class Compiler : private impl::noncopyable {
public:
  // A factory: bring-up fails for reasons that are not a caller's mistake,
  // and then the graph interprets.
  [[nodiscard]] static DDX_JIT_API result<Compiler> create();

  DDX_JIT_API ~Compiler();
  DDX_JIT_API Compiler(Compiler &&) noexcept;
  DDX_JIT_API Compiler &operator=(Compiler &&) noexcept;

  // The report is filled where one is given; nothing else observes it, so a
  // caller that does not measure pays nothing.
  [[nodiscard]] DDX_JIT_API result<Kernel>
  compile(const rt::Graph<double> &g, const Options &opt = {},
          CompileReport *report = nullptr);

private:
  struct Impl;
  DDX_JIT_API explicit Compiler(std::shared_ptr<Impl> impl) noexcept;

  // Private, but Ir::str() calls it from a consumer's translation unit, so it
  // is exported like the public ones.
  friend class Ir;
  [[nodiscard]] DDX_JIT_API result<std::string>
  render_ir(const rt::Graph<double> &g, const Options &opt) const;

  std::shared_ptr<Impl> impl_;
};

// The optimised IR a graph would compile to.  Borrowing, so the pipeline runs
// only if something reads it; the deleted overloads check both ends outlive
// the handle.
class Ir {
public:
  Ir(const Compiler &c, const rt::Graph<double> &g, Options opt = {}) noexcept
      : compiler_(c), graph_(g), options_(opt) {}
  Ir(const Compiler &&, const rt::Graph<double> &, Options = {}) = delete;
  Ir(const Compiler &, const rt::Graph<double> &&, Options = {}) = delete;

  [[nodiscard]] result<std::string> str() const {
    return compiler_.render_ir(graph_, options_);
  }

private:
  const Compiler &compiler_;
  const rt::Graph<double> &graph_;
  Options options_;
};

} // namespace ddx::jit
