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
// The JIT emits machine types, so only a graph over a machine scalar compiles.
// The constraint has to match the definition, hence the one ddx include.
template <impl::Numeric T> class Graph;
} // namespace ddx::rt

namespace ddx::jit {

// Which vector math library a transcendental over a vector of lanes may call.
// Off by default: glibc's vector routines are documented to ~4 ULP where the
// scalar ones are ~0.5, and a kernel is checked against the interpreter, which
// calls the scalar ones.  Asking for one is asking for that trade.
enum class VecLib : std::uint8_t { None, Auto, Libmvec };

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

// Whether an equation compiles its graph, and what a call does while it is
// still compiling.  Not named Auto: nothing decides, and deciding would need
// the one thing an equation does not know -- how many times the kernel will be
// called, which is what a compile has to repay.
//
//   Interpret   the default.  No compiler is asked, so a program that never
//               says otherwise never loads LLVM.  Not a fallback: the block
//               sweep runs within ~1.4x of a kernel.
//   Compile     Equation::options() starts the compile there and then; calls
//               before it lands are swept, and switch over when it arrives.
//
// Compile is a *ladder*, not a compile.  It asks for two kernels: one at
// codegen 0, which is what answers first, and one at the stated codegen_level,
// which replaces it when it lands.  At 128 variables the cheap rung compiles
// 2.8x to 5.1x faster for a kernel 1.2x to 2.3x slower, so a caller stops
// sweeping that much sooner and keeps the better kernel for the rest of the
// run.  Both rungs go to the same background pool and neither is waited for.
// The swap is invisible: every codegen level agrees to the bit, so a loop that
// runs across one sees no movement.  Equation::kernel_level() is the only way
// to see which rung is answering.
//
// It does not block, and the first call does not wait.  The compile starts at
// the moment it is *asked for*, not at the first call, so it overlaps whatever
// the caller does next; a caller who would rather have a kernel than an
// answer now asks outright, with Equation::wait_for_kernel() -- which waits for
// the first rung, not the best one.  There is one compiling backend rather than
// a blocking and a non-blocking one because waiting is a question a caller asks
// per call, not a property of the build.
//
// A result does not move when the kernel arrives.  The two paths contract the
// same multiply-adds into the same fma, because the contraction is decided in
// the graph -- rt::contraction_at() reads it off the nodes, the sweep calls
// std::fma and the kernel emits llvm.fma -- so a loop running across the
// switchover sees the same bits before and after it.
//
// A compile still in flight when the process exits is joined then, not
// abandoned: dropping the equation costs nothing, exiting waits for LLVM.
enum class Backend : std::uint8_t { Interpret, Compile };

struct Options {
  // Read by Equation, never by Compiler::compile(), which is asked outright.
  Backend backend = Backend::Interpret;
  // The batch a caller intends to hand to one jacobian() or hessian() call.
  // Stated rather than inferred: the kernel is built when a backend is chosen,
  // which is before any call exists to read an `n` from.  It decides the lane
  // width and nothing else -- a call carrying some other number of points is
  // answered correctly, just not by the kernel that number would have built.
  // 1 is the minimisation case: a gradient per step, at one point.
  std::size_t points = 1;
  // Points per loop iteration: the body is emitted over <lanes x double>.
  // 0 is the host's widest vector register in doubles -- 4 on AVX2, 8 on
  // AVX-512 -- and 1 is scalar.  Any width compiles; the backend splits a
  // vector the host cannot hold.  Every width gives the same bits: the lanes
  // are independent IEEE operations, and a transcendental is the same scalar
  // libm call per lane.
  //
  // 0 derives it from `points`, which is what a caller who has said what their
  // batch is should leave it at.  A kernel `lanes` wide computes a register's
  // worth of points and stores the ones asked for, which is the right trade for
  // a batch and the wrong one for a single point: measured at 16 variables, one
  // point per call, a scalar kernel is 1.2x to 4.1x faster than the host's.
  unsigned lanes = 0;
  // LLVM's optimisation level for the IR pipeline, 0 to 3.
  unsigned opt_level = default_opt_level;
  // LLVM's codegen level, 0 to 3 -- under Backend::Compile, the *top* rung of
  // the ladder rather than the level: a compile at codegen 0 always runs
  // underneath it, and 0 is therefore the one value that asks for a single
  // rung.  Instruction selection and register allocation are ~95% of a
  // compile, so this is the knob that trades kernel speed for compile time.
  //
  // 1 by default, not 2: measured over the four thermodynamic gradients at 16,
  // 32 and 64 variables, at both lane widths, it compiles 11-15% faster and
  // its kernels are within the noise either way -- several are nominally
  // quicker.  3 is 2's compile time for 2's kernel.  All four agree to the bit:
  // 1, 2 and 3 share an instruction selector and a register allocator, and 0
  // forms no FMAs of its own but is handed llvm.fma outright.  0 is still the
  // wide trade -- the linear-time allocator and FastISel.  Against 1, which is
  // what the ladder puts it under, that is 1.8x to 2.7x off the compile at 16
  // to 32 variables and 2.8x to 5.1x at 128, for 1.2x to 2.3x onto the kernel;
  // the 8-9x figure is against 2 and is not the comparison the default makes.
  unsigned codegen_level = 1;
  // PipelineTuningOptions::SLPVectorization.  Packs independent
  // subexpressions *within* one point -- a different axis from `lanes`, which
  // packs one operation across several points, and the only one of the two
  // available to a caller holding a single point.  Measured at one lane and 64
  // variables: 14% off uniquac and 5% off mse for 0-66% of compile, nothing on
  // rss, marginally negative on pr.  Model-dependent, hence off.
  bool slp = false;
  // PipelineTuningOptions::LoopVectorization.  Off because the loop is already
  // emitted `lanes` wide, so there is nothing left to find -- and because it
  // cannot find it anyway past about nineteen columns, where the runtime alias
  // check it needs between every pair of them exceeds its own budget and it
  // gives up having paid for the analysis.  Measured at one lane: it multiplies
  // the pass pipeline by up to 683x and returns the same kernel or a worse one.
  bool loop_vectorize = false;
  VecLib veclib = VecLib::None;
  bool contract = default_contract; // Follows DDX_FP_FLAGS
  // Per-pass timing to stderr, through LLVM's own TimePassesHandler.  Off by
  // default: it is a diagnostic for whoever is asking why a compile is slow,
  // and it prints where nothing else in the library does.
  bool time_passes = false;
  // Keep the object file the backend produced, so Kernel::object() can hand it
  // back for storing and adopt()ing later.  Off by default: it is machine code
  // held for the life of the kernel, and a caller who is not going to save it
  // should not carry it.
  bool retain_object = false;
  // Where compiled objects are kept between runs, or empty for nowhere, which
  // is the default: writing to a caller's disk is not something a library does
  // because it would be faster.  A directory named here is consulted before a
  // compile and written after one, keyed by the graph, the options and the
  // host -- so a second run of the same equation links an object instead of
  // compiling it, skipping emission, the pass pipeline and codegen alike.
  //
  // Not part of what a kernel *is*: two compiles differing only in this
  // produce the same code, which is why it is excluded from the key and why
  // rt::Object does not carry it.
  std::string cache_dir{};

  // What Equation::options() compares before discarding a compiled kernel.
  friend bool operator==(const Options &, const Options &) = default;
};

// Where a compile spent its time.  `codegen` brackets the symbol lookup too:
// LLJIT compiles lazily, so the lookup is what forces the backend to run.
struct CompileReport {
  std::size_t nodes = 0;        // live nodes in the graph
  std::size_t instructions = 0; // IR instructions after emission
  std::chrono::nanoseconds emit{};
  std::chrono::nanoseconds optimize{};
  std::chrono::nanoseconds codegen{};
};

// One compiled graph.  A copy is one atomic increment: a Kernel keeps the JIT
// that owns its code alive, and dropping the last one frees the code.
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

  // xs[j] is the column for symbol j, g[j] the partial in it, each of length n;
  // a block that was not requested is `{}`.  noexcept because codegen marks the
  // kernel and every libm declaration it calls nounwind.
  void operator()(std::span<const double *const> xs, std::span<double *const> f,
                  std::span<double *const> g, std::span<double *const> h,
                  std::size_t n) const noexcept {
    // The ABI takes bare pointers; the spans are what say the column count was
    // right.  A mismatch past here is silent memory corruption.
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

  // The object file this kernel's code was linked from, for a caller who
  // intends to store it and hand it back to adopt() later.  An adopted kernel
  // always has one; a compiled one has it only where the compile asked, because
  // keeping a megabyte of machine code alive for every kernel that will never
  // be saved is not a cost a compile should pay by default.
  // Options::retain_object is what turns it on.
  //
  // Shared, so a copy stays one atomic increment: a Kernel is meant to be
  // cheap to pass around and this must not change that.
  [[nodiscard]] std::span<const std::byte> object() const noexcept {
    return object_ ? std::span<const std::byte>{*object_}
                   : std::span<const std::byte>{};
  }

  // The name the code is linked under, which is what adopt() has to be told to
  // find it again.  A caller cannot derive it -- it is chosen by the compile --
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

// A Compiler *is* the LLJIT, and every Kernel shares it: one going out of scope
// frees nothing still callable, and reclaims nothing either.
class Compiler : private impl::noncopyable {
public:
  // A factory: bring-up fails for reasons that are not a caller's mistake,
  // and then the graph interprets.
  [[nodiscard]] static DDX_JIT_API result<Compiler> create();

  // Everything an object file has to agree with before it may be run here:
  // the triple, the CPU, the feature string and the LLVM that produced it, as
  // one line -- `x86_64-pc-linux-gnu/znver3/+avx2,+fma/llvm-20.1`.
  //
  // A string rather than a folded integer because this is the field that
  // answers *why* a stored kernel was passed over, and "the key differed" is
  // not an answer anyone can act on.  Compared once when an object is adopted,
  // never in a loop.
  [[nodiscard]] DDX_JIT_API std::string_view host_identity() const noexcept;

  DDX_JIT_API ~Compiler();
  DDX_JIT_API Compiler(Compiler &&) noexcept;
  DDX_JIT_API Compiler &operator=(Compiler &&) noexcept;

  // The report is filled where one is given; nothing else observes it, so a
  // caller that does not measure pays nothing.
  [[nodiscard]] DDX_JIT_API result<Kernel>
  compile(const rt::Graph<double> &g, const Options &opt = {},
          CompileReport *report = nullptr);

  // The same compile, off the calling thread.  The graph is shared rather than
  // borrowed because the compile outlives this call, and the future is one
  // whose destructor does not join: dropping it abandons the result, it does
  // not wait for it.
  [[nodiscard]] DDX_JIT_API std::shared_future<result<Kernel>>
  compile_async(std::shared_ptr<const rt::Graph<double>> g, Options opt = {});

  // Link an object file compiled earlier -- by this process or a previous one
  // -- and hand back the kernel in it, skipping emission, the pass pipeline and
  // codegen alike.  This is the whole of what a compile is worth: at 128
  // variables the top rung costs seconds and does not repay its own CPU until
  // millions of points, so not doing it twice is worth more than any lever
  // inside it.
  //
  // Nothing is verified beyond the link: an object built for another target, or
  // from another graph, is a caller's mistake and this cannot see it.  Whoever
  // stored the bytes stores what they have to agree with -- rt::Object carries
  // the triple, the CPU, the features, the Options and the graph digest for
  // exactly that reason -- and a mismatch is answered by compiling, never by
  // running the wrong code.  The column counts are stated because the object no
  // longer has a graph to read them off.
  //
  // Errors rather than aborts on a truncated or corrupt object: a cache entry
  // that will not link is a miss.
  [[nodiscard]] DDX_JIT_API result<Kernel>
  adopt(std::span<const std::byte> object, std::string_view symbol,
        std::size_t arity, std::size_t values, std::size_t jacobian,
        std::size_t hessian);

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
