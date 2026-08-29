#pragma once

#include "symbolic/expressions.hpp"
#include "util/error.hpp"
#include "util/export.hpp"
#include "util/pinned.hpp"

#include <boost/describe.hpp>
#include <boost/mp11/algorithm.hpp>

#include <cassert>
#include <chrono>
#include <concepts>
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
template <impl::Numeric T> class Graph;
} // namespace ddx::rt

namespace ddx::jit {

// Off: glibc's vector routines are ~4 ULP where the scalar ones are ~0.5.
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

// Compile and Adapt are a ladder: a codegen-0 kernel answers first and one at
// codegen_level replaces it, bit-identically; calls before the first rung lands
// are swept.  Adapt asks for a rung once the batch has paid for it.
enum class Backend : std::uint8_t { Interpret, Compile, Adapt };

struct Options {
  // Read by Equation, never by Compiler::compile().
  Backend backend = Backend::Interpret;
  // The batch one call will carry.  Stated, not inferred: the kernel is built
  // before any call exists to read an `n` from.  Sets the lane width only.
  std::size_t points = 1;
  // 0 derives it from `points`, 1 is scalar; every width gives the same bits.
  unsigned lanes = 0;
  // 0 to 3.
  unsigned opt_level = default_opt_level;
  // 0 to 3; under Backend::Compile the top rung, so 0 asks for a single one.
  // Selection and register allocation are ~95% of a compile.
  unsigned codegen_level = 1;
  // Within one point; model-dependent, hence off.
  bool slp = false;
  // Off: the body is already `lanes` wide, and past ~19 columns the alias
  // checks exceed their own budget.
  bool loop_vectorize = false;
  // Points, not points x nodes: a compile costs about what a node costs and a
  // swept point saves about what a node saves, so the size cancels out.
  std::size_t warm_points = 1uz << 16;
  std::size_t hot_points = 1uz << 20;
  // With one, a derived `lanes` is the widest width the library serves -- four
  // doubles for libmvec -- whatever the host's registers hold.
  VecLib veclib = VecLib::None;
  bool contract = default_contract; // Follows DDX_FP_FLAGS
  bool time_passes = false;         // Per-pass timing to stderr
  // On: the object is the one part of a saved equation nothing can reconstruct.
  bool retain_object = true;
  // Object cache directory, empty for none; keyed by graph, options and host.
  std::string cache_dir{};

  friend bool operator==(const Options &, const Options &) = default;
};

// Identity, not policy: `retain_object` and `cache_dir` change no machine code.
// Also the field list the archive serialises Options by.
BOOST_DESCRIBE_STRUCT(Options, (),
                      (backend, points, lanes, opt_level, codegen_level, slp,
                       loop_vectorize, veclib, contract, time_passes))

// `backend` is described because a saved equation restores it, but says whether
// to compile, never what is emitted.
[[nodiscard]] inline bool same_codegen(const Options &a, const Options &b) {
  bool same = true;
  boost::mp11::mp_for_each<
      boost::describe::describe_members<Options, boost::describe::mod_public>>(
      [&](auto D) {
        if constexpr (!std::same_as<std::remove_cvref_t<decltype(a.*D.pointer)>,
                                    Backend>) {
          same = same && a.*D.pointer == b.*D.pointer;
        }
      });
  return same;
}

// `codegen` brackets the symbol lookup: LLJIT compiles lazily.
struct CompileReport {
  std::size_t nodes = 0;        // live nodes in the graph
  std::size_t instructions = 0; // IR instructions after emission
  std::chrono::nanoseconds emit{};
  std::chrono::nanoseconds optimize{};
  std::chrono::nanoseconds codegen{};
};

// A copy is one atomic increment; the last Kernel to go frees the code.
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
  // unrequested block is `{}`.  noexcept: codegen marks the kernel nounwind.
  void operator()(std::span<const double *const> xs, std::span<double *const> f,
                  std::span<double *const> g, std::span<double *const> h,
                  std::size_t n) const noexcept {
    // A mismatch past here is silent memory corruption.
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

  // Empty only where Options::retain_object was turned off.
  [[nodiscard]] std::span<const std::byte> object() const noexcept {
    return object_ ? std::span<const std::byte>{*object_}
                   : std::span<const std::byte>{};
  }

  // Chosen by the compile: whoever stores the object stores this beside it.
  [[nodiscard]] std::string_view symbol() const noexcept {
    return symbol_ ? std::string_view{*symbol_} : std::string_view{};
  }

private:
  function_type fn_ = nullptr;
  std::shared_ptr<void> code_; // Held, never read
  symbol_type symbol_;
  object_type object_;
  std::size_t arity_ = 0;
  std::size_t values_ = 0;
  std::size_t jacobian_ = 0;
  std::size_t hessian_ = 0;
};

// The LLJIT itself, shared by every Kernel it made.
class Compiler : private impl::noncopyable {
public:
  [[nodiscard]] static DDX_JIT_API result<Compiler> create();

  // `x86_64-pc-linux-gnu/znver3/+avx2,+fma/llvm-20.1`: a string, not a folded
  // key, because this answers *why* a stored kernel was passed over.
  [[nodiscard]] DDX_JIT_API std::string_view host_identity() const noexcept;

  DDX_JIT_API ~Compiler();
  DDX_JIT_API Compiler(Compiler &&) noexcept;
  DDX_JIT_API Compiler &operator=(Compiler &&) noexcept;

  [[nodiscard]] DDX_JIT_API result<Kernel>
  compile(const rt::Graph<double> &g, const Options &opt = {},
          CompileReport *report = nullptr);

  // The future's destructor does not join: dropping it abandons the result.
  [[nodiscard]] DDX_JIT_API std::shared_future<result<Kernel>>
  compile_async(std::shared_ptr<const rt::Graph<double>> g, Options opt = {});

  // Links an object compiled earlier.  Nothing beyond the link is verified --
  // rt::Object carries the host, Options and graph digest for that -- and a
  // corrupt object is an error, not an abort.  The counts are stated because
  // the object has no graph to read them off.
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

// Borrowing, so the pipeline runs only if something reads it; the deleted
// overloads refuse temporaries.
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
