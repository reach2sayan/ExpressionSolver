#pragma once

#include "symbolic/expressions.hpp" // ddx::impl::Numeric
#include "util/error.hpp"
#include "util/export.hpp"
#include "util/pinned.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>

namespace ddx::rt {
// The JIT emits machine types, so only a graph over a machine scalar compiles.
// The constraint has to match the definition, hence the one ddx include.
template <impl::Numeric T> class Graph;
} // namespace ddx::rt

namespace ddx::jit {

// Which vector math library the loop vectoriser may call.
enum class VecLib : std::uint8_t { None, Auto, Libmvec };

// Carries text where the rest of the library's errors do not: LLVM's are not
// one of a fixed set, and there is no allocation-free path through here.
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
  VecLib veclib = VecLib::Auto;
  bool contract = default_contract; // Follows DDX_FP_FLAGS
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

// A Compiler *is* the LLJIT, and every Kernel it hands out shares it: one going
// out of scope frees nothing still callable, and reclaims nothing either.
class Compiler : private impl::noncopyable {
public:
  // A factory, because bring-up fails for reasons that are not a caller's
  // mistake -- no native target on this host.  Then the graph interprets.
  [[nodiscard]] static DDX_JIT_API result<Compiler> create();

  DDX_JIT_API ~Compiler();
  DDX_JIT_API Compiler(Compiler &&) noexcept;
  DDX_JIT_API Compiler &operator=(Compiler &&) noexcept;

  [[nodiscard]] DDX_JIT_API result<Kernel> compile(const rt::Graph<double> &g,
                                                   const Options &opt = {});

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

// The optimised IR a graph would compile to.  Borrowing rather than a string,
// so the pipeline runs only if something reads it; the deleted overloads make
// the compiler check that both ends outlive the handle.
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
