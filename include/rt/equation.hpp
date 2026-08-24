#pragma once

#include "ops/numeric.hpp" // compile_time_factorial
// The univariate sweep runs on the truncated-polynomial scalar forward mode
// supplies.
#include "dual/taylor_dual.hpp"
#include "md/md.hpp"
#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/expressions.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "symbolic/equation.hpp"
#include "util/ranges.hpp" // to<C>(), piped by us -- see the header for why

// Optional: without it the batch calls run the interpreter, under the same
// signatures.
#ifdef DDX_HAS_JIT
#include "jit/kernel.hpp"

#include <future>
#endif

#include <mutex> // unique_lock, which <shared_mutex> does not carry
#include <shared_mutex>

#include <algorithm>
#include <array>
#include <concepts>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <vector>

// Keyed on the RTExpression<T> *pattern*: a `requires` over <TFirst, TRest...>
// would be ambiguous with the compile-time specialisation, since the two
// concepts do not subsume one another.
namespace ddx::impl {

namespace rt_detail {

template <typename A, typename T>
concept CPointArg =
    Numeric<std::remove_cvref_t<A>> || CEntry<A> ||
    (std::ranges::input_range<A> &&
     Numeric<std::remove_cvref_t<std::ranges::range_value_t<A>>>);

// CEvalArg admits any input_range, so without this a span of columns would try
// to become a scalar.
template <typename R, typename Ptr>
concept CColumns = std::ranges::contiguous_range<R> &&
                   std::convertible_to<std::ranges::range_value_t<R>, Ptr>;

#ifdef DDX_HAS_JIT
// One LLJIT for the process, not one per Equation instantiation: a static in a
// class template is per specialisation, so a program holding a scalar equation
// and a two-output system used to bring up two of them.  Sharing also gives the
// kernel names one counter rather than one each, so no two compiles can mint
// the same name.  Concurrent compiles through it are what
// ConcurrentIRCompiler's per-module TargetMachine exists for.
//
// A function-local static in an inline function, so it is one object across
// translation units -- and reached only from a compiling backend, which is what
// keeps an interpret-only program from ever bringing LLVM up.  A Kernel holds a
// share of what it came from, so nothing here has to outlive anything.
[[nodiscard]] inline jit::Compiler *shared_compiler() {
  static jit::result<jit::Compiler> instance = jit::Compiler::create();
  return instance ? &*instance : nullptr;
}
#endif

} // namespace rt_detail

template <Numeric T, typename... Rest>
  requires(std::same_as<Rest, rt::RTExpression<T>> && ...)
class Equation<rt::RTExpression<T>, Rest...> {
public:
  using value_type = T;
  static constexpr std::size_t output_dim = 1 + sizeof...(Rest);

  // A constructor cannot answer with an error, and an expression can name no
  // graph two ways: a bare literal, or a symbol named with no arena current.
  // Rather than a result<Equation> at every call site, the refusal rides on the
  // Equation, where the poison RTExpression's journey ends.  The constructors
  // take the graph as a precondition; create() establishes it.
  [[nodiscard]] static constexpr Equation create(rt::RTExpression<T> first,
                                                 Rest... rest) {
    if (const auto bad = why_not(first, rest...)) {
      return Equation{*bad};
    }
    return Equation{first, rest...};
  }

  // Owning: the Builder is heap-allocated, so the nodes survive the move.  A
  // refusal drops `owned` here, which is the arena nobody will ever use.
  [[nodiscard]] static Equation create(std::unique_ptr<rt::Builder<T>> owned,
                                       rt::RTExpression<T> first,
                                       Rest... rest) {
    if (const auto bad = why_not(first, rest...)) {
      return Equation{*bad};
    }
    return Equation{std::move(owned), first, rest...};
  }

  // Why the equation could not be built, or nullopt.  poisoned() is the same
  // question without the reason.
  [[nodiscard]] constexpr bool poisoned() const noexcept {
    return bad_.has_value();
  }
  [[nodiscard]] constexpr std::optional<error> status() const noexcept {
    return bad_;
  }

  // Optional rather than a degenerate 0: a literal-only graph legitimately has
  // no symbols, so a count could not tell "none" from "no equation".  A loop
  // written over arity() stops compiling rather than running zero times.
  [[nodiscard]] constexpr std::optional<std::size_t> arity() const {
    return poisoned() ? std::nullopt : std::optional{symbol_count()};
  }
  [[nodiscard]] constexpr std::optional<std::span<const std::string>>
  symbols() const {
    if (poisoned()) {
      return std::nullopt;
    }
    // Not std::optional{...}: Builder::symbols() answers with a const vector&,
    // so CTAD would deduce optional<vector> and span the copy as it died.
    return std::span<const std::string>{arena_->symbols()};
  }

  // The symbol list exists only at run time, so every spelling, positional
  // included, answers with result<T>.
  template <rt_detail::CPointArg<T>... Args>
  [[nodiscard]] constexpr result<std::vector<T>>
  point(const Args &...args) const {
    if (bad_) {
      return std::unexpected{*bad_};
    }
    std::vector<T> at(symbol_count(), T{});
    result<void> ok{};
    if constexpr ((CEntry<Args> && ...) && sizeof...(Args) > 0) {
      ((ok = ok.and_then([&] { return assign_named(at, args); })), ...);
    } else if constexpr (sizeof...(Args) == 1 &&
                         (std::ranges::input_range<Args> && ...)) {
      ok = assign_range(at, args...);
    } else {
      const std::array<T, sizeof...(Args)> positional{static_cast<T>(args)...};
      ok = assign_range(at, positional);
    }
    return ok.transform([&at] { return std::move(at); });
  }

  [[nodiscard]] constexpr auto
  evaluate(const rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      const auto values = rt::evaluate_all(*arena_, at);
      if constexpr (output_dim == 1) {
        return values[roots_[0]];
      } else {
        return roots_ |
               std::views::transform([&](rt::NodeId r) { return values[r]; }) |
               to<std::vector<T>>();
      }
    });
  }

  // Row-major m x n, matching Equation::jacobian -- which drops the leading
  // axis at m == 1, where this is simply n long.
  [[nodiscard]] constexpr result<std::vector<T>>
  jacobian(const rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform(
        [this](const auto &at) { return harvest(derivative_.partial, at); });
  }

  // Dense row-major m x n x n; the graph holds it compressed by colour.
  [[nodiscard]] result<std::vector<T>>
  hessian(const rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      const auto &blocks = hessians_;
      const auto values = rt::evaluate_all(*arena_, at);
      const std::size_t n = symbol_count();

      std::vector<T> out(output_dim * n * n);
      const impl::md::mdspan dense{
          out.data(), impl::md::dextents<std::size_t, 3>{output_dim, n, n}};
      for (const auto [k, block] : std::views::enumerate(blocks)) {
        for (const auto [i, j] : std::views::cartesian_product(
                 std::views::iota(0uz, n), std::views::iota(0uz, n))) {
          dense[static_cast<std::size_t>(k), i, j] = values[block.at(i, j)];
        }
      }
      return out;
    });
  }

#ifdef DDX_HAS_JIT
  // The whole of a consumer's setup: whether to compile at all, and how.
  // Discards anything already compiled, so the choice holds however late.
  // The one non-const member: concurrent const calls are safe, a call
  // overlapping this one is the caller's race.
  Equation &options(const jit::Options &opt) {
    if (opt == options_) {
      return *this; // Asking for what is already running relaunches nothing.
    }
    options_ = opt;
    // A fresh cache rather than a cleared one: a reader still holding a
    // snapshot from the old configuration finishes against it, and the compile
    // the old one had in flight is abandoned rather than waited for.
    cache_ = std::make_unique<Cache>();
    // Choosing a backend is what starts the build, so that it overlaps whatever
    // the caller does before their first call.  The values-and-Jacobian lane
    // only: a caller who never asks for a Hessian should not compile one, and
    // that lane prepares itself on first use as this one used to.
    if (opt.backend != jit::Backend::Interpret && !poisoned()) {
      const std::unique_lock fill{cache_->plain.mutex};
      prepare(cache_->plain, false);
    }
    return *this;
  }
  [[nodiscard]] const jit::Options &options() const noexcept {
    return options_;
  }
#endif

  // Whether a batch call goes through compiled code: false with no backend,
  // on a refused compile, and under Interpret.
  [[nodiscard]] bool uses_kernel() const {
#ifdef DDX_HAS_JIT
    return !poisoned() && static_cast<bool>(snapshot(false)->kernel);
#else
    return false;
#endif
  }

#ifdef DDX_HAS_JIT
  // Block until a Background compile has landed, and answer uses_kernel().
  // Nothing else waits: this is for a caller who would rather have the kernel
  // than the sweep, and for a test that needs the answer to be settled.
  [[nodiscard]] bool wait_for_kernel() const {
    if (poisoned()) {
      return false;
    }
    (void)snapshot(false); // launches it, if nothing has yet
    {
      const std::shared_lock read{cache_->plain.mutex};
      if (cache_->plain.pending.valid()) {
        cache_->plain.pending.wait();
      }
    }
    return static_cast<bool>(snapshot(false)->kernel);
  }
#endif

  [[nodiscard]] std::optional<std::size_t> hessian_colors() const
    requires(output_dim == 1)
  {
    return poisoned() ? std::nullopt
                      : std::optional{hessians_.front().colors()};
  }

  // One Taylor sweep: seed c[0] = x0, c[1] = 1, then un-normalise c[Order].
  template <std::size_t Order>
  [[nodiscard]] result<T> univariate_derivative(T x0) const
    requires(output_dim == 1 && Order > 0)
  {
    if (bad_) {
      return std::unexpected{*bad_};
    }
    if (symbol_count() != 1) {
      return fail(errc::not_univariate);
    }
    using Taylor = impl::TaylorDual<T, Order>;
    Taylor seed;
    seed.c[0] = x0;
    seed.c[1] = T{1};

    const std::array<Taylor, 1> at{seed};
    const auto values = rt::evaluate_all(*arena_, at);
    return values[roots_[0]].c[Order] *
           static_cast<T>(impl::detail::compile_time_factorial(Order));
  }

  // --- batch ---------------------------------------------------------------
  // `xs[j]` is the column for symbol j; each output span holds one pointer per
  // output column, every column n long.  An unchecked column-count mismatch is
  // silent memory corruption.

  [[nodiscard]] result<void>
  jacobian(const rt_detail::CColumns<const T *> auto &xs,
           const rt_detail::CColumns<T *> auto &f,
           const rt_detail::CColumns<T *> auto &g, std::size_t n) const {
    return dispatch(*snapshot(false), as_columns(xs), as_columns(f),
                    as_columns(g), {}, n);
  }

  [[nodiscard]] result<void>
  hessian(const rt_detail::CColumns<const T *> auto &xs,
          const rt_detail::CColumns<T *> auto &f,
          const rt_detail::CColumns<T *> auto &g,
          const rt_detail::CColumns<T *> auto &h, std::size_t n) const
    requires(output_dim == 1)
  {
    return dispatch(*snapshot(true), as_columns(xs), as_columns(f),
                    as_columns(g), as_columns(h), n);
  }

  // What a caller sizes its buffers by.  Read off what the constructor already
  // swept rather than off a frozen graph: these are the three counts
  // GraphBuilder would compute the layout from, so they answer the same, and
  // asking must not build -- sizing a Hessian buffer used to compile one.
  [[nodiscard]] std::optional<std::size_t> value_columns() const {
    return poisoned() ? std::nullopt : std::optional{roots_.size()};
  }
  [[nodiscard]] std::optional<std::size_t> jacobian_columns() const {
    return poisoned() ? std::nullopt
                      : std::optional{derivative_.partial.size()};
  }
  [[nodiscard]] std::optional<std::size_t> hessian_columns() const
    requires(output_dim == 1)
  {
    return poisoned() ? std::nullopt
                      : std::optional{hessians_.front().compressed.size()};
  }

private:
  // Past the poison guard, so the arena is known good; arity() is the checked
  // spelling.
  [[nodiscard]] constexpr std::size_t symbol_count() const {
    return arena_->symbols().size();
  }

  // Both take a graph as a precondition, which create() establishes.  The third
  // construction is none at all: arena_ null, roots_ empty, derivative_
  // default, and every accessor short-circuits before reading them.
  constexpr explicit Equation(error why) : bad_(why) {
    if !consteval {
      cache_ = std::make_unique<Cache>();
    }
  }

  constexpr explicit Equation(rt::RTExpression<T> first, Rest... rest)
      : arena_(first.builder(), borrow) {
    roots_.reserve(output_dim);
    roots_.push_back(first.id(*arena_));
    (roots_.push_back(rest.id(*arena_)), ...);
    derivative_ = rt::jacobian(*arena_, roots_);
    // The colouring rt::hessian needs runs through Boost.Graph, and a Cache
    // holds a mutex; neither is available while constant-evaluating, and a
    // constant-evaluated equation reaches neither.
    if !consteval {
      hessians_ = roots_ | std::views::transform([&](rt::NodeId r) {
                    return rt::hessian(*arena_, r);
                  }) |
                  to<std::vector<rt::Hessian>>();
      cache_ = std::make_unique<Cache>();
    }
  }

  Equation(std::unique_ptr<rt::Builder<T>> owned, rt::RTExpression<T> first,
           Rest... rest)
      : Equation(first, rest...) {
    arena_ = ArenaPtr{owned.release(), reclaim};
  }

  // The graph an expression names, or why it names none.  Poison first: no
  // arena current is a different mistake from a literal that reached no graph.
  [[nodiscard]] static constexpr std::optional<error>
  why_not(const rt::RTExpression<T> &first, const Rest &...rest) noexcept {
    if (first.poisoned() || (rest.poisoned() || ...)) {
      return error{errc::no_arena};
    }
    if (first.builder() == nullptr) {
      return error{errc::no_graph};
    }
    return std::nullopt;
  }

  template <CEntry V>
  [[nodiscard]] constexpr result<void> assign_named(std::vector<T> &at,
                                                    const V &nv) const {
    const auto name = std::remove_cvref_t<V>::symbol.view();
    const auto names = arena_->symbols();
    const auto it = std::ranges::find(names, name);
    if (it == names.end()) {
      return fail(errc::unknown_symbol);
    }
    at[static_cast<std::size_t>(it - names.begin())] = static_cast<T>(nv.value);
    return {};
  }

  [[nodiscard]] constexpr result<void>
  assign_range(std::vector<T> &at,
               const std::ranges::input_range auto &r) const {
    if (std::ranges::size(r) != at.size()) {
      return fail(errc::wrong_arity);
    }
    std::ranges::transform(r, at.begin(),
                           [](const auto &v) { return static_cast<T>(v); });
    return {};
  }

  [[nodiscard]] constexpr std::vector<T>
  harvest(const std::vector<rt::NodeId> &nodes,
          const std::vector<T> &at) const {
    const auto values = rt::evaluate_all(*arena_, at);
    return nodes |
           std::views::transform([&](rt::NodeId n) { return values[n]; }) |
           to<std::vector<T>>();
  }

  // Published, never written: a kernel arrives as a *new* Compiled, so nothing
  // a reader can see is ever modified under it.
  struct Compiled {
    // Shared, not held: a background compile keeps its own share, so the graph
    // outlives an Equation that went away mid-compile.
    std::shared_ptr<const rt::Graph<T>> graph;
#ifdef DDX_HAS_JIT
    // Empty where T is not the JIT's scalar, and under Background until the
    // compile lands.
    jit::Kernel kernel{};
#endif
  };

  // One shape of graph, filled once and never invalidated -- which is what the
  // single slot it replaces could not promise, since wanting a Hessian used to
  // refill it and dangle whatever reference another call was holding.
  struct Lane {
    mutable std::shared_mutex mutex;
    std::shared_ptr<const Compiled> ready;
#ifdef DDX_HAS_JIT
    std::shared_future<jit::result<jit::Kernel>> pending; // set once, then read
#endif
  };

  struct Cache {
    Lane plain;        // values and Jacobian
    Lane with_hessian; // and the compressed Hessian columns
  };

  // Ownership, not a reference: a reader holds its share for the whole call, so
  // publishing cannot pull the graph out from under a dispatch() in flight.
  [[nodiscard]] std::shared_ptr<const Compiled>
  snapshot(bool want_hessian) const {
    Lane &lane = want_hessian ? cache_->with_hessian : cache_->plain;
    {
      const std::shared_lock read{lane.mutex};
      if (lane.ready && !settling(lane)) {
        return lane.ready;
      }
    }
    const std::unique_lock fill{lane.mutex}; // another thread may have won
    if (!lane.ready) {
      prepare(lane, want_hessian);
    }
#ifdef DDX_HAS_JIT
    // Compile is the promise that a call answers from the kernel, so this is
    // where it is kept -- once, since adopting clears `pending`.
    if (options_.backend == jit::Backend::Compile && lane.pending.valid()) {
      lane.pending.wait();
    }
#endif
    if (arrived(lane)) {
      adopt(lane);
    }
    return lane.ready;
  }

  // Whether the lane still owes the reader work before its snapshot is the
  // best one available: a landed kernel to publish, or a compile to wait for.
  [[nodiscard]] bool settling([[maybe_unused]] const Lane &lane) const {
#ifdef DDX_HAS_JIT
    return arrived(lane) || (options_.backend == jit::Backend::Compile &&
                             lane.pending.valid());
#else
    return false;
#endif
  }

  // Republish rather than write into what a reader may be holding.
  void adopt([[maybe_unused]] Lane &lane) const {
#ifdef DDX_HAS_JIT
    // A refused compile leaves the kernel empty, and the sweep stays.
    const auto &landed = lane.pending.get();
    lane.ready = std::make_shared<const Compiled>(Compiled{
        .graph = lane.ready->graph, .kernel = landed ? *landed : jit::Kernel{}});
    lane.pending = {};
#endif
  }

  // Whether a background compile has finished but not yet been published.
  // Polling a shared_future from many threads is well-defined, and `pending` is
  // written once under the write lock and never reassigned.
  [[nodiscard]] static bool arrived([[maybe_unused]] const Lane &lane) {
#ifdef DDX_HAS_JIT
    using namespace std::chrono_literals;
    return lane.pending.valid() &&
           lane.pending.wait_for(0s) == std::future_status::ready;
#else
    return false;
#endif
  }

  // Freeze this lane's graph and, under a compiling backend, put the compile in
  // flight.  Every derivative it needs was swept in the constructor, so this
  // only reads the arena -- Graph::freeze takes it by const reference.  Called
  // under the lane's write lock, from options() and from snapshot().
  void prepare(Lane &lane, bool want_hessian) const {
    // No arena to build from.  An empty Graph reports zero columns of any kind.
    if (bad_) {
      lane.ready = std::make_shared<const Compiled>(
          Compiled{.graph = std::make_shared<const rt::Graph<T>>()});
      return;
    }
    rt::GraphBuilder<T> gb{*arena_};
    gb.values_from(roots_);
    gb.jacobian_from(derivative_.partial);
    if (want_hessian) {
      gb.hessian_from(hessians_.front());
    }
    Compiled out{.graph = std::make_shared<const rt::Graph<T>>(gb.build())};
#ifdef DDX_HAS_JIT
    if constexpr (std::same_as<T, double>) {
      // Always asynchronous, Compile included: what separates the two is where
      // the result is waited for, not where the work starts.  Not an error a
      // caller handles -- run() falls back to interpret().
      auto *const c =
          options_.backend != jit::Backend::Interpret ? compiler() : nullptr;
      if (c != nullptr) {
        lane.pending = c->compile_async(out.graph, effective_options());
      }
    }
#endif
    lane.ready = std::make_shared<const Compiled>(std::move(out));
  }

#ifdef DDX_HAS_JIT
  // How wide to emit, from the batch the caller said they have.  A kernel `w`
  // lanes wide computes `w` points and stores the ones asked for, so on a batch
  // of one it does a register's worth of work to answer for a single point --
  // measured at 16 variables, 1.2x to 4.1x of the scalar kernel.  Below kLanes
  // the sweep goes scalar for the same reason, so the two agree on where a
  // batch stops being a batch.  A stated `lanes` is honoured as it stands.
  [[nodiscard]] jit::Options effective_options() const noexcept {
    jit::Options opt = options_;
    if (opt.lanes == 0 && opt.points < kLanes) {
      opt.lanes = 1;
    }
    return opt;
  }

  // A Kernel does not own its code, so the compiler outlives every kernel.
  // Null on a host with no JIT, asked once.
  static jit::Compiler *compiler() { return rt_detail::shared_compiler(); }
#endif

  // Points per block sweep.  Two AVX2 registers of doubles: wide enough that
  // the per-node switch is amortised, narrow enough that the tape stays a
  // handful of nodes' worth per lane.
  static constexpr std::size_t kLanes = 8;

  void interpret(const Compiled &c, std::span<const T *const> xs,
                 std::span<T *const> f, std::span<T *const> g,
                 std::span<T *const> h, std::size_t n) const {
    const auto blocks = c.graph->output_blocks();
    const auto order = c.graph->live_order();
    const std::size_t symbols = symbol_count();

    // Tapes are sized by the arena rather than the graph: a borrowed builder
    // may have grown since the freeze, and live ids index below that.
    //
    // A batch shorter than a block sweeps one point at a time rather than
    // padding out to kLanes.  Padding is nearly free on the tail of a long
    // batch and is the whole cost on a short one -- a minimisation routine
    // asking for one gradient per step would otherwise pay for eight.
    if (n < kLanes) {
      std::vector<T> at(symbols);
      std::vector<T> tape(arena_->size());
      for (const std::size_t i : std::views::iota(0uz, n)) {
        for (const auto [j, column] : std::views::enumerate(xs)) {
          at[static_cast<std::size_t>(j)] = column[i];
        }
        rt::evaluate_into(*arena_, at, order, std::span<T>{tape});
        for (const auto [columns, block] : std::views::zip(
                 std::array{f, g, h},
                 std::array{blocks.values, blocks.jacobian, blocks.hessian})) {
          for (const auto [column, o] : std::views::zip(columns, block)) {
            column[i] = tape[o];
          }
        }
      }
      return;
    }

    // kLanes points per sweep: the switch is paid once per node per block, and
    // each operation becomes a lane loop wide enough to vectorise.  A short
    // final block repeats its last point; the repeated lanes are never read
    // back.
    std::vector<T> lanes(symbols * kLanes);
    std::vector<T> tape(arena_->size() * kLanes);

    for (std::size_t base = 0; base < n; base += kLanes) {
      const std::size_t width = std::min(kLanes, n - base);
      for (const auto [j, column] : std::views::enumerate(xs)) {
        T *const dst = lanes.data() + static_cast<std::size_t>(j) * kLanes;
        for (const std::size_t l : std::views::iota(0uz, kLanes)) {
          dst[l] = column[base + std::min(l, width - 1)];
        }
      }
      rt::evaluate_block<kLanes>(*arena_, std::span<const T>{lanes}, order,
                                 std::span<T>{tape});

      for (const auto [columns, block] : std::views::zip(
               std::array{f, g, h},
               std::array{blocks.values, blocks.jacobian, blocks.hessian})) {
        for (const auto [column, o] : std::views::zip(columns, block)) {
          const T *const src = tape.data() + std::size_t{o} * kLanes;
          for (const std::size_t l : std::views::iota(0uz, width)) {
            column[base + l] = src[l];
          }
        }
      }
    }
  }

  template <std::ranges::contiguous_range R>
  static auto as_columns(const R &r) {
    using Ptr = std::ranges::range_value_t<R>;
    return std::span<Ptr const>{std::ranges::data(r), std::ranges::size(r)};
  }

  [[nodiscard]] result<void>
  dispatch(const Compiled &c, std::span<const T *const> xs,
           std::span<T *const> f, std::span<T *const> g, std::span<T *const> h,
           std::size_t n) const {
    if (bad_) {
      return std::unexpected{*bad_};
    }
    const auto blocks = c.graph->output_blocks();
    if (xs.size() != symbol_count() || f.size() != blocks.values.size() ||
        g.size() != blocks.jacobian.size() ||
        h.size() != blocks.hessian.size()) {
      return fail(errc::wrong_column_count);
    }
    run(c, xs, f, g, h, n);
    return {};
  }

  void run(const Compiled &c, std::span<const T *const> xs,
           std::span<T *const> f, std::span<T *const> g, std::span<T *const> h,
           std::size_t n) const {
#ifdef DDX_HAS_JIT
    if constexpr (std::same_as<T, double>) {
      if (c.kernel) {
        c.kernel(xs, f, g, h, n);
        return;
      }
    }
#endif
    interpret(c, xs, f, g, h, n);
  }

  // The deleter is the only difference between an arena the caller keeps and
  // one this Equation was handed.
  using ArenaPtr = std::unique_ptr<rt::Builder<T>, void (*)(rt::Builder<T> *)>;
  static constexpr void borrow(rt::Builder<T> *) noexcept {}
  static constexpr void reclaim(rt::Builder<T> *b) noexcept { delete b; }

  ArenaPtr arena_{nullptr, borrow};
  std::vector<rt::NodeId> roots_;

#ifdef DDX_HAS_JIT
  jit::Options options_{};
#endif
  // Eager: one reverse sweep is microseconds, and it keeps every per-point
  // accessor const and constexpr.
  rt::Jacobian derivative_;
  // Swept once in the constructor and never touched again, which is the whole
  // of what makes the const members safe to call concurrently: rt::hessian
  // *appends to the arena*, and an arena a caller lent us may be lent to
  // another Equation as well.  ~1.2M nodes and 350 ms for a 50-species mixture,
  // against a graph build that would otherwise sweep it a second time.
  std::vector<rt::Hessian> hessians_;
  // Owned outright -- nothing outside an Equation holds it -- which is also
  // what keeps a constant-evaluated Equation destructible: unique_ptr's
  // destructor is constexpr where shared_ptr's is not.  Null only there.
  std::unique_ptr<Cache> cache_;
  // Set exactly when why_not() refused, so poisoned() <=> arena_ == nullptr.
  std::optional<error> bad_{};
};

} // namespace ddx::impl

namespace ddx::rt {

#ifdef DDX_HAS_JIT
// Named here as well as in jit, since choosing it is a caller's business.
using jit::Backend;
#endif

// Over expressions already built in a caller's own arena.  Partial
// specialisations contribute no deduction guides, so this is the whole of CTAD.
template <impl::Numeric T, typename... Ts>
  requires(std::same_as<Ts, RTExpression<T>> && ...)
[[nodiscard]] constexpr auto equation(RTExpression<T> first, Ts... rest) {
  return impl::Equation<RTExpression<T>, Ts...>::create(first, rest...);
}

// Build an Equation without naming an arena:
//
//   const auto eq = ddx::rt::equation([] {
//     const auto x = var("x");
//     const auto y = var("y");
//     return exp(x) * sin(y);
//   });
//
// The arena is moved into the Equation, so it can be returned and stored where
// one over a caller's Builder cannot.  equation() makes an arena current for
// the callback's duration.  A range of expressions is a system.
template <impl::Numeric T = double, std::invocable Assemble>
[[nodiscard]] auto equation(Assemble &&assemble) {
  auto arena = std::make_unique<Builder<T>>();
  auto built = [&] {
    const auto scope = detail::scoped_arena(*arena);
    return std::forward<Assemble>(assemble)();
  }();

  using Built = std::remove_cvref_t<decltype(built)>;
  if constexpr (std::same_as<Built, RTExpression<T>>) {
    return impl::Equation<RTExpression<T>>::create(std::move(arena), built);
  } else {
    // output_dim lives in the type, so the size must too -- array, not vector.
    constexpr std::size_t outputs = std::tuple_size_v<Built>;
    static_assert(outputs > 0,
                  "equation: a system needs at least one function");
    return impl::index_apply<outputs - 1>([&]<std::size_t... Rest>() {
      return impl::Equation<RTExpression<T>, detail::Repeat<Rest, T>...>::
          create(std::move(arena), built[0], built[Rest + 1]...);
    });
  }
}

} // namespace ddx::rt
