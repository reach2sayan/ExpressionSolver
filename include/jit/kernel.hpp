#pragma once

#include "symbolic/expressions.hpp" // ddx::impl::Numeric
#include "util/error.hpp"
#include "util/export.hpp"
#include "util/pinned.hpp"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <future>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ddx::rt {
// The JIT emits machine types, so only a machine scalar compiles.  The
// constraint has to match the definition, hence the one ddx include.
template <impl::Numeric T> class Graph;
} // namespace ddx::rt

namespace ddx::jit {

// Which vector math library a transcendental over lanes may call.  Off: glibc's
// vector routines are documented to ~4 ULP where the scalar ones are ~0.5.
enum class VecLib : std::uint8_t { None, Auto, Libmvec };

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

// Whether an equation compiles its graph.
//
//   Interpret   the default.  No compiler is asked, so a program that never
//               says otherwise never loads LLVM.  Not a fallback: the block
//               sweep runs within ~1.4x of a kernel.
//   Compile     Equation::options() starts the compile there and then; calls
//               before it lands are swept, and switch over when it arrives.
//
// Compile is a *ladder*: a codegen-0 kernel answers first and one at the stated
// codegen_level replaces it.  Nothing blocks -- wait_for_kernel() is how a
// caller asks to -- and every level agrees to the bit, so the swap is
// invisible; kernel_level() says which rung is answering.
enum class Backend : std::uint8_t { Interpret, Compile };

struct Options {
  // Read by Equation, never by Compiler::compile(), which is asked outright.
  Backend backend = Backend::Interpret;
  // The batch a caller intends to hand to one call.  Stated, not inferred: the
  // kernel is built when a backend is chosen, before any call exists to read an
  // `n` from.  It decides the lane width and nothing else.
  std::size_t points = 1;
  // The body is emitted over <lanes x double>; 0 derives it from `points`, 1 is
  // scalar.  Every width gives the same bits, lanes being independent.
  unsigned lanes = 0;
  // LLVM's optimisation level for the IR pipeline, 0 to 3.
  unsigned opt_level = default_opt_level;
  // 0 to 3, and under Backend::Compile the *top* rung, so 0 is the one value
  // asking for a single one.  Selection and register allocation are ~95% of a
  // compile, so this trades kernel speed for compile time.  All four agree to
  // the bit.
  unsigned codegen_level = 1;
  // Packs independent subexpressions *within* one point, the only axis open to
  // a caller holding a single one.  Model-dependent, hence off.
  bool slp = false;
  // Off: the loop is already emitted `lanes` wide, and past ~19 columns the
  // alias check it needs between every pair exceeds its own budget.
  bool loop_vectorize = false;
  VecLib veclib = VecLib::None;
  bool contract = default_contract; // Follows DDX_FP_FLAGS
  // Per-pass timing to stderr.  Off: it prints where nothing else here does.
  bool time_passes = false;
  // For Kernel::object().  On: the object is the one part of a saved equation
  // nothing can reconstruct, so a caller that never saves opts out rather than
  // finding out too late.
  bool retain_object = true;
  // Where objects are kept between runs, empty for nowhere.  Read before a
  // compile and written after, keyed by graph, options and host, so a second run
  // links instead of compiling.  Not part of what a kernel *is*, hence outside
  // the key and absent from rt::Object.
  std::string cache_dir{};

  // What Equation::options() compares before discarding a compiled kernel.
  friend bool operator==(const Options &, const Options &) = default;
};

// `codegen` brackets the symbol lookup: LLJIT compiles lazily, so the lookup is
// what forces the backend to run.
struct CompileReport {
  std::size_t nodes = 0;        // live nodes in the graph
  std::size_t instructions = 0; // IR instructions after emission
  std::chrono::nanoseconds emit{};
  std::chrono::nanoseconds optimize{};
  std::chrono::nanoseconds codegen{};
};

// One compiled graph.  A copy is one atomic increment: a Kernel keeps the JIT
// that owns its code alive, and dropping the last frees the code.
class Kernel {
public:
  using function_type = void (*)(const double *const *, double *const *,
                                 double *const *, double *const *, std::size_t);

  using object_type = std::shared_ptr<const std::vector<std::byte>>;
  using symbol_type = std::shared_ptr<const std::string>;

  Kernel() = default;
  Kernel(function_type fn, std::size_t arity, std::size_t values,
         std::size_t jacobian, std::size_t hessian, std::shared_ptr<void> code,
         symbol_type symbol = {}, object_type object = {}) noexcept
      : fn_(fn), code_(std::move(code)), symbol_(std::move(symbol)),
        object_(std::move(object)), arity_(arity), values_(values),
        jacobian_(jacobian), hessian_(hessian) {}

  // xs[j] is the column for symbol j, g[j] the partial in it, each n long; an
  // unrequested block is `{}`.  noexcept because codegen marks the kernel and
  // every libm declaration it calls nounwind.
  void operator()(std::span<const double *const> xs, std::span<double *const> f,
                  std::span<double *const> g, std::span<double *const> h,
                  std::size_t n) const noexcept {
    // The ABI takes bare pointers, so the spans are what carry the count; a
    // mismatch past here is silent memory corruption.
    assert(xs.size() == arity_ && f.size() == values_ &&
           g.size() == jacobian_ && h.size() == hessian_);
    fn_(xs.data(), f.data(), g.data(), h.data(), n);
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return fn_ != nullptr;
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

  // What the code was linked from, to store and hand back to adopt().  Empty
  // only where Options::retain_object was turned off.
  [[nodiscard]] std::span<const std::byte> object() const noexcept {
    return object_ ? std::span<const std::byte>{*object_}
                   : std::span<const std::byte>{};
  }

  // What adopt() has to be told to find the code again: chosen by the compile,
  // so whoever stores the object stores this beside it.
  [[nodiscard]] std::string_view symbol() const noexcept {
    return symbol_ ? std::string_view{*symbol_} : std::string_view{};
  }

private:
  function_type fn_ = nullptr;
  std::shared_ptr<void> code_; // Held, never read; operator() calls through fn_
  symbol_type symbol_;         // The name `fn_` was looked up under
  object_type object_;         // The bytes `code_` was linked from, or none
  std::size_t arity_ = 0;
  std::size_t values_ = 0;
  std::size_t jacobian_ = 0;
  std::size_t hessian_ = 0;
};

// A Compiler *is* the LLJIT and every Kernel shares it, so one going out of
// scope frees nothing still callable.
class Compiler : private impl::noncopyable {
public:
  // A factory: bring-up fails for reasons that are not a caller's mistake, and
  // then the graph interprets.
  [[nodiscard]] static DDX_JIT_API result<Compiler> create();

  // What an object file must agree with, as one line --
  // `x86_64-pc-linux-gnu/znver3/+avx2,+fma/llvm-20.1`.  A string, not a folded
  // key, because this answers *why* a stored kernel was passed over.
  [[nodiscard]] DDX_JIT_API std::string_view host_identity() const noexcept;

  DDX_JIT_API ~Compiler();
  DDX_JIT_API Compiler(Compiler &&) noexcept;
  DDX_JIT_API Compiler &operator=(Compiler &&) noexcept;

  // Filled where one is given, so a caller that does not measure pays nothing.
  [[nodiscard]] DDX_JIT_API result<Kernel>
  compile(const rt::Graph<double> &g, const Options &opt = {},
          CompileReport *report = nullptr);

  // Off the calling thread, so the graph is shared rather than borrowed.  The
  // future's destructor does not join: dropping it abandons the result.
  [[nodiscard]] DDX_JIT_API std::shared_future<result<Kernel>>
  compile_async(std::shared_ptr<const rt::Graph<double>> g, Options opt = {});

  // Link an object file compiled earlier and hand back the kernel in it,
  // skipping emission, the pass pipeline and codegen alike.
  //
  // Nothing is verified beyond the link: an object from another target or graph
  // is a caller's mistake this cannot see, which is why rt::Object carries the
  // host, the Options and the graph digest.  The column counts are stated
  // because the object has no graph to read them off.  A corrupt one errors
  // rather than aborting -- an entry that will not link is a miss.
  [[nodiscard]] DDX_JIT_API result<Kernel>
  adopt(std::span<const std::byte> object, std::string_view symbol,
        std::size_t arity, std::size_t values, std::size_t jacobian,
        std::size_t hessian);

private:
  struct Impl;
  DDX_JIT_API explicit Compiler(std::shared_ptr<Impl> impl) noexcept;

  // Private, but Ir::str() calls it from a consumer's TU, so it is exported.
  friend class Ir;
  [[nodiscard]] DDX_JIT_API result<std::string>
  render_ir(const rt::Graph<double> &g, const Options &opt) const;

  std::shared_ptr<Impl> impl_;
};

// The optimised IR a graph would compile to.  Borrowing, so the pipeline runs
// only if something reads it; the deleted overloads check both ends outlive it.
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
