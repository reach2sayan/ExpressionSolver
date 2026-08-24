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
//   Background  Equation::options() starts the compile there and then; calls
//               before it lands are swept, and switch over when it arrives.
//   Compile     the same launch, and the first call waits for it, so results
//               are the kernel's from the start.
//
// Both compiling backends start at the moment they are *asked for*, not at the
// first call, so the compile overlaps whatever the caller does next.
//
// Background alone moves results in the last bits at the moment the kernel
// arrives: the kernel contracts a multiply and an add into an FMA where the
// sweep evaluates them separately, so the two agree to rounding rather than to
// the bit.  A loop running across that point sees a ULP or two of movement.
//
// A compile still in flight when the process exits is joined then, not
// abandoned: dropping the equation costs nothing, exiting waits for LLVM.
enum class Backend : std::uint8_t { Interpret, Background, Compile };

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
  // LLVM's codegen level, 0 to 3.  Instruction selection and register
  // allocation are ~95% of a compile, so this is the knob that trades kernel
  // speed for compile time.
  //
  // 1 by default, not 2: measured over the four thermodynamic gradients at 16,
  // 32 and 64 variables, at both lane widths, it compiles 11-15% faster and
  // its kernels are within the noise either way -- several are nominally
  // quicker.  3 is 2's compile time for 2's kernel.  All three select the same
  // instruction selector and the same register allocator, so they agree to the
  // bit; only 0 differs, and widely -- it takes the linear-time allocator and
  // FastISel, which forms no FMAs, for 8.9x off the compile and 1.6x onto the
  // kernel.
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

  Kernel() = default;
  Kernel(function_type fn, std::size_t arity, std::size_t values,
         std::size_t jacobian, std::size_t hessian,
         std::shared_ptr<void> code) noexcept
      : fn_(fn), code_(std::move(code)), arity_(arity), values_(values),
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

private:
  function_type fn_ = nullptr;
  std::shared_ptr<void> code_; // Held, never read; operator() calls through fn_
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

  // The same compile, off the calling thread.  The graph is shared rather than
  // borrowed because the compile outlives this call, and the future is one
  // whose destructor does not join: dropping it abandons the result, it does
  // not wait for it.
  [[nodiscard]] DDX_JIT_API std::shared_future<result<Kernel>>
  compile_async(std::shared_ptr<const rt::Graph<double>> g, Options opt = {});

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
