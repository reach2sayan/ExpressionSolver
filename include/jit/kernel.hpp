#pragma once

#include "expr/expressions.hpp" // ddx::impl::Numeric
#include "jit/export.hpp"
#include "util/error.hpp"
#include "util/pinned.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>

// The JIT's public surface.
namespace ddx::rt {
// The JIT emits machine types, so it compiles graphs over a machine scalar.
// A graph over anything else -- Numeric admits matrices and quaternions -- is
// interpretable but not compilable, which is why everything below names
// Graph<double> rather than the template.  The constraint has to match the
// definition, hence the one ddx include; it carries no dependency of its own.
template <impl::Numeric T> class Graph;
} // namespace ddx::rt

namespace ddx::jit {

// Which vector math library the loop vectoriser may call.
enum class VecLib : std::uint8_t { None, Auto, Libmvec };

// A JIT failure carries text where the rest of the library's does not: LLVM's
// own errors are not one of a fixed set, and the reason a target will not come
// up on this host is the whole of what a caller can act on.  There is no
// numeric path through here to keep allocation-free.
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

// One compiled graph.  Cheap to copy: a copy is one atomic increment, because a
// Kernel keeps the JIT that owns its code alive rather than pointing into
// something another object may free.  Dropping the last Kernel frees the code.
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

  // Spans on the C++ surface, raw pointers only in function_type, which is the
  // JIT's actual ABI: a block that was not requested is `{}` here rather than a
  // null pointer whose length the callee has to infer.  xs[j] is the column for
  // symbol j, g[j] the column for the partial in that symbol, all of length n.
  // noexcept because the emitted function is nounwind: codegen marks both the
  // kernel and every libm declaration it calls, so there is no unwind edge to
  // cross here.
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
  // Never read: it is here to be held.  operator() calls through fn_ alone.
  std::shared_ptr<void> code_;
  std::size_t arity_ = 0;
  std::size_t values_ = 0;
  std::size_t jacobian_ = 0;
  std::size_t hessian_ = 0;
};

// A Compiler *is* the LLJIT, and the code a Kernel calls lives in it. Move-only
// on the surface, so there is one Compiler to hand on or hold in a static;
// underneath, every Kernel it hands out shares that LLJIT.  A Compiler going
// out of scope therefore frees nothing still callable -- and reclaims nothing
// either: one surviving Kernel holds the target machine, the symbol generators
// and every module compiled through it.
class Compiler : private impl::noncopyable {
public:
  // Bringing up the LLJIT is the one thing here that can fail for reasons that
  // are not a caller's mistake -- no native target for this host, no way to
  // build a target machine -- so it happens in a factory rather than a
  // constructor.  A caller that cannot get one is not stuck: the runtime graph
  // interprets instead.
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
  // crosses the boundary like the public ones and is exported like them.
  friend class Ir;
  [[nodiscard]] DDX_JIT_API result<std::string>
  render_ir(const rt::Graph<double> &g, const Options &opt) const;

  std::shared_ptr<Impl> impl_;
};

// The optimised IR a graph would compile to.  A borrowing handle rather than a
// string, so the IR prints through the same formatter as everything else and
// the pipeline only runs if something reads it.  Both ends have to outlive the
// handle, which the deleted overloads make the compiler check.
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
