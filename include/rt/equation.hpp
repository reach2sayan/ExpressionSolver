#pragma once

#include "ops/numeric.hpp" // compile_time_factorial
#include "dual/taylor_dual.hpp" // the univariate sweep's scalar
#include "md/md.hpp"
#include "rt/archive.hpp"
#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/expressions.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "rt/rebalance.hpp"
#include "symbolic/equation.hpp"
#include "util/ranges.hpp" // to<C>() and append(), ours

#include <boost/container/small_vector.hpp>

// Optional: without it the batch calls interpret, under the same signatures.
#ifdef DDX_HAS_JIT
#include "jit/kernel.hpp"

#include <future>
#endif

#include <filesystem>
#include <mutex> // unique_lock, which <shared_mutex> does not carry
#include <shared_mutex>

#include <algorithm>
#include <array>
#include <atomic>
#include <concepts>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <vector>

// Keyed on the RTExpression<T> *pattern*: a `requires` over <TFirst, TRest...>
// would be ambiguous with the compile-time specialisation.
namespace ddx::impl {

namespace rt_detail {

template <typename A, typename T>
concept CPointArg =
    Numeric<std::remove_cvref_t<A>> || CEntry<A> ||
    (std::ranges::input_range<A> &&
     Numeric<std::remove_cvref_t<std::ranges::range_value_t<A>>>);

// CPointArg admits any input_range, so without this a span of columns would try
// to become a scalar.
template <typename R, typename Ptr>
concept CColumns = std::ranges::contiguous_range<R> &&
                   std::convertible_to<std::ranges::range_value_t<R>, Ptr>;

#ifdef DDX_HAS_JIT
// One LLJIT for the process -- a static in a class template would be per
// specialisation -- reached only from a compiling backend, so an interpret-only
// program never brings LLVM up.
[[nodiscard]] inline jit::Compiler *shared_compiler() {
  static jit::result<jit::Compiler> instance = jit::Compiler::create();
  return instance ? &*instance : nullptr;
}
#endif

} // namespace rt_detail

template <Numeric T, typename... Rest>
  requires(std::same_as<Rest, rt::RTExpression<T>> && ...)
class Equation<rt::RTExpression<T>, Rest...> {
  // What a lane's graph carries; each level is the one below plus a block.
  enum class Want : std::uint8_t { Values, Jacobian, Hessian };

public:
  using value_type = T;
  static constexpr std::size_t output_dim = 1 + sizeof...(Rest);

  // A refusal rides on the Equation rather than a result<Equation> at every
  // call site; see poisoned().
  [[nodiscard]] static constexpr Equation create(rt::RTExpression<T> first,
                                                 Rest... rest) {
    if (const auto bad = why_not(first, rest...)) {
      return Equation{*bad};
    }
    first.builder()->seal();
    return Equation{first, rest...};
  }

  // Owning: the Builder is heap-allocated, so the nodes survive the move.
  [[nodiscard]] static Equation create(std::unique_ptr<rt::Builder<T>> owned,
                                       rt::RTExpression<T> first,
                                       Rest... rest) {
    if (const auto bad = why_not(first, rest...)) {
      return Equation{*bad};
    }
    first.builder()->seal();
    return Equation{std::move(owned), first, rest...};
  }

  // Build the model, then take the sweeps off disk if the file still describes
  // it.  An absent or stale file rebuilds rather than refusing; loaded() tells.
  [[nodiscard]] static Equation cached(const std::filesystem::path &path,
                                       std::unique_ptr<rt::Builder<T>> owned,
                                       rt::RTExpression<T> first, Rest... rest)
    requires std::floating_point<T>
  {
    if (const auto why = why_not(first, rest...)) {
      return Equation{*why};
    }
    std::vector<rt::NodeId> roots;
    roots.reserve(output_dim);
    roots.push_back(first.id(*owned));
    (roots.push_back(rest.id(*owned)), ...);

    if (auto snap = rt::load_snapshot<T>(path);
        snap && snap->roots.size() == output_dim &&
        std::ranges::equal(snap->roots, roots) &&
        rt::digest<T>(snap->symbols, snap->nodes, snap->model_nodes) ==
            rt::digest<T>(owned->symbols(), owned->nodes(), owned->size())) {
      return Equation{std::move(*snap)};
    }
    Equation eq = create(std::move(owned), first, rest...);
    // A cache that cannot be written is still a working equation.
    (void)eq.save(path);
    return eq;
  }

  // Why the equation could not be built, or nullopt.
  [[nodiscard]] constexpr bool poisoned() const noexcept {
    return bad_.has_value();
  }
  [[nodiscard]] constexpr std::optional<error> status() const noexcept {
    return bad_;
  }

  // Optional, not a degenerate 0: a literal-only graph legitimately has none.
  [[nodiscard]] constexpr std::optional<std::size_t> arity() const {
    return poisoned() ? std::nullopt : std::optional{symbol_count()};
  }
  [[nodiscard]] constexpr std::optional<std::span<const std::string>>
  symbols() const {
    if (poisoned()) {
      return std::nullopt;
    }
    // Not std::optional{...}: symbols() answers with a const vector&, so CTAD
    // would deduce optional<vector> and span the copy as it died.
    return std::span<const std::string>{arena_->symbols()};
  }

  // The symbol list exists only at run time, so every spelling can refuse.
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
      std::array<T, output_dim> f{};
      gather(Want::Values, at, std::span<T>{f});
      if constexpr (output_dim == 1) {
        return f[0];
      } else {
        return std::vector<T>(f.begin(), f.end());
      }
    });
  }

  // Row-major m x n; at m == 1 the leading axis goes and this is n long.
  [[nodiscard]] constexpr result<std::vector<T>>
  jacobian(const rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      std::array<T, output_dim> f{};
      std::vector<T> g(derivative_.partial.size());
      gather(Want::Jacobian, at, std::span<T>{f}, g);
      return g;
    });
  }

  // Dense row-major m x n x n; the graph holds it compressed by colour.
  [[nodiscard]] result<std::vector<T>>
  hessian(const rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      const std::size_t n = symbol_count();
      std::vector<T> out(output_dim * n * n);
      const impl::md::mdspan dense{
          out.data(), impl::md::dextents<std::size_t, 3>{output_dim, n, n}};
      const auto dims = std::views::iota(0uz, n);

      // Only one root's Hessian is frozen into a lane; a system walks the arena.
      if constexpr (output_dim == 1) {
        const rt::Coloring &c = hessians_.front().coloring;
        std::array<T, 1> f{};
        std::vector<T> g(derivative_.partial.size());
        std::vector<T> h(hessians_.front().compressed.size());
        gather(Want::Hessian, at, std::span<T>{f}, g, h);
        // Every other column of that colour is structurally zero.
        for (const auto [i, j] : std::views::cartesian_product(dims, dims)) {
          const std::size_t col = c.color[j];
          dense[0uz, i, j] = c.target(col, i) == j ? h[c.column(col, i)] : T{0};
        }
      } else {
        // Hessian::at reads only the compressed cells and `zero`.
        auto wanted = hessians_ |
                      std::views::transform(&rt::Hessian::compressed) |
                      std::views::join | impl::to<std::vector<rt::NodeId>>();
        impl::append(wanted,
                     hessians_ | std::views::transform(&rt::Hessian::zero));
        const auto values = rt::evaluate_reachable(*arena_, wanted, at);
        for (const auto [k, block] : hessians_ | std::views::enumerate) {
          for (const auto [i, j] : std::views::cartesian_product(dims, dims)) {
            dense[static_cast<std::size_t>(k), i, j] = values[block.at(i, j)];
          }
        }
      }
      return out;
    });
  }

#ifdef DDX_HAS_JIT
  // Whether to compile at all, and how; discards anything already compiled.
  // The one non-const member: a call overlapping it is the caller's race.
  Equation &options(const jit::Options &opt) {
    if (opt == options_) {
      return *this;
    }
    options_ = opt;
    cache_ = std::make_unique<Cache>();
    // Choosing a backend starts the build, so it overlaps whatever the caller
    // does before their first call.
    if (opt.backend != jit::Backend::Interpret && !poisoned()) {
      const std::unique_lock fill{cache_->plain.mutex};
      prepare(cache_->plain, Want::Jacobian);
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
    return !poisoned() && static_cast<bool>(snapshot(Want::Jacobian)->kernel);
#else
    return false;
#endif
  }

#ifdef DDX_HAS_JIT
  // Which rung is answering, as its codegen level, and nothing where the sweep
  // is.  Every rung gives the same bits, so this is the only way to see it
  // climb.  Does not wait.
  [[nodiscard]] std::optional<unsigned> kernel_level() const {
    if (poisoned()) {
      return std::nullopt;
    }
    const auto snap = snapshot(Want::Jacobian);
    return snap->kernel ? std::optional{snap->level} : std::nullopt;
  }

  // Block until the *first* rung lands -- rungs[0], the cheap compile -- and
  // answer uses_kernel().  Nothing else waits.
  //
  // Under Adapt it waits for a rung already bought and buys none: a call that
  // quietly overrode the counter would make every measurement of the policy a
  // lie.  A caller who wants a kernel now asks for Backend::Compile.
  [[nodiscard]] bool wait_for_kernel() const {
    if (poisoned()) {
      return false;
    }
    (void)snapshot(Want::Jacobian); // launches it, if nothing has yet
    {
      const std::shared_lock read{cache_->plain.mutex};
      if (const auto &first = cache_->plain.rungs.front(); first.pending.valid()) {
        first.pending.wait();
      }
    }
    return static_cast<bool>(snapshot(Want::Jacobian)->kernel);
  }

  // How far the Jacobian lane is toward its next rung, and nothing under any
  // other backend or once there is none left.  Without it a lane that is still
  // counting and one whose compile was refused both read as no kernel.
  struct Warmup {
    std::size_t points;    // batch points this lane has been asked for
    std::size_t threshold; // where the next rung is bought
  };
  [[nodiscard]] std::optional<Warmup> warming() const {
    if (poisoned() || options_.backend != jit::Backend::Adapt) {
      return std::nullopt;
    }
    const Lane &lane = cache_->plain;
    const unsigned asked = lane.asked.load(std::memory_order_relaxed);
    return asked >= 2
               ? std::nullopt
               : std::optional{Warmup{
                     lane.points.load(std::memory_order_relaxed),
                     rung_at(asked)}};
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
    // One root out of an arena holding a gradient and a Hessian too, and a
    // Taylor node costs (Order + 1) coefficients.
    const auto values = rt::evaluate_reachable(
        *arena_, std::span<const rt::NodeId>{roots_}.first(1), at);
    return values[roots_[0]].c[Order] *
           static_cast<T>(impl::detail::compile_time_factorial(Order));
  }

  // --- batch ---------------------------------------------------------------
  // `xs[j]` is the column for symbol j; each output span holds one pointer per
  // output column, every column n long.

  // Its own lane, not jacobian()'s with the partials dropped: on a coupled
  // model the gradient is most of the graph.
  [[nodiscard]] result<void>
  evaluate(const rt_detail::CColumns<const T *> auto &xs,
           const rt_detail::CColumns<T *> auto &f, std::size_t n) const {
    return dispatch(*snapshot(Want::Values, n), as_columns(xs), as_columns(f),
                    {}, {}, n);
  }

  [[nodiscard]] result<void>
  jacobian(const rt_detail::CColumns<const T *> auto &xs,
           const rt_detail::CColumns<T *> auto &f,
           const rt_detail::CColumns<T *> auto &g, std::size_t n) const {
    return dispatch(*snapshot(Want::Jacobian, n), as_columns(xs),
                    as_columns(f), as_columns(g), {}, n);
  }

  [[nodiscard]] result<void>
  hessian(const rt_detail::CColumns<const T *> auto &xs,
          const rt_detail::CColumns<T *> auto &f,
          const rt_detail::CColumns<T *> auto &g,
          const rt_detail::CColumns<T *> auto &h, std::size_t n) const
    requires(output_dim == 1)
  {
    return dispatch(*snapshot(Want::Hessian, n), as_columns(xs), as_columns(f),
                    as_columns(g), as_columns(h), n);
  }

  // What a caller sizes buffers by, read off the constructor's sweep rather
  // than a frozen graph: asking must not build a lane.
  [[nodiscard]] constexpr std::optional<std::size_t> value_columns() const {
    return poisoned() ? std::nullopt : std::optional{roots_.size()};
  }
  [[nodiscard]] constexpr std::optional<std::size_t> jacobian_columns() const {
    return poisoned() ? std::nullopt
                      : std::optional{derivative_.partial.size()};
  }
  [[nodiscard]] std::optional<std::size_t> hessian_columns() const
    requires(output_dim == 1)
  {
    return poisoned() ? std::nullopt
                      : std::optional{hessians_.front().compressed.size()};
  }

  // --- the equation on disk --------------------------------------------------
  // The arena and the sweeps, which is what the constructor spends its time on.
  // No lane's Graph: freezing one is a linear pass over an arena the file has.

  [[nodiscard]] result<void> save(const std::filesystem::path &path) const
    requires std::floating_point<T>
  {
    return poisoned() ? std::unexpected{*bad_} : rt::save(snapshot(), path);
  }

  // Whether `path` holds *this* equation.  rt::verify() answers the other
  // question, whether a file is readable by this build at all.
  [[nodiscard]] result<void> verify(const std::filesystem::path &path) const
    requires std::floating_point<T>
  {
    if (poisoned()) {
      return std::unexpected{*bad_};
    }
    const auto snap = rt::load_snapshot<T>(path);
    if (!snap) {
      return std::unexpected{snap.error()};
    }
    return describes(*snap) ? result<void>{} : fail(errc::archive_mismatch);
  }

  // Nothing built and nothing swept; the model need not exist in this program.
  // The output count is in the type, so another number is refused.
  [[nodiscard]] static result<Equation> load(const std::filesystem::path &path)
    requires std::floating_point<T>
  {
    auto snap = rt::load_snapshot<T>(path);
    if (!snap) {
      return std::unexpected{snap.error()};
    }
    if (snap->roots.size() != output_dim) {
      return fail(errc::archive_mismatch);
    }
    return Equation{std::move(*snap)};
  }

  // The only way to tell which path rt::equation(path, model) took.
  [[nodiscard]] constexpr bool loaded() const noexcept { return loaded_; }

private:
  // Past the poison guard, so the arena is known good; arity() is the checked
  // spelling.
  [[nodiscard]] constexpr std::size_t symbol_count() const {
    return arena_->symbols().size();
  }

  // PyEquation fills the same struct, so one serialiser serves both.
  [[nodiscard]] rt::Snapshot<T> snapshot() const {
    rt::Snapshot<T> snap;
    snap.symbols = arena_->symbols();
    snap.nodes.assign(arena_->nodes().begin(), arena_->nodes().end());
    snap.roots = roots_;
    snap.jacobian = derivative_;
    snap.hessians = hessians_;
    snap.model_nodes = model_nodes_;
#ifdef DDX_HAS_JIT
    snap.options = options_;
    snap.objects = objects();
#endif
    return snap;
  }

#ifdef DDX_HAS_JIT
  // What the lanes are holding, for the file to carry.  Only kernels that kept
  // their bytes, so without retain_object an equation saves its graph and no
  // code.
  [[nodiscard]] std::vector<rt::Object> objects() const {
    if (poisoned()) {
      return {};
    }
    auto *const c = compiler();
    if (c == nullptr) {
      return {};
    }
    std::vector<rt::Object> out;
    for (const Want want :
         {Want::Values, Want::Jacobian, Want::Hessian}) {
      Lane &lane = lane_for(want);
      const std::shared_lock read{lane.mutex};
      if (!lane.ready || !lane.ready->kernel ||
          lane.ready->kernel.object().empty()) {
        continue;
      }
      const auto code = lane.ready->kernel.object();
      out.push_back({.want = static_cast<std::uint8_t>(want),
                     .symbol = std::string{lane.ready->kernel.symbol()},
                     .host = std::string{c->host_identity()},
                     .digest = rt::digest(*lane.ready->graph),
                     // What the compile was given, not what the caller
                     // stated: effective_options() settles the lane width.
                     .options = effective_options(),
                     .code = {code.begin(), code.end()}});
    }
    return out;
  }

  // Graph, host and codegen options must all agree: adopt() cannot see that an
  // object came from another graph, so nothing reaches it unchecked.
  [[nodiscard]] const rt::Object *stored(Want want,
                                         const rt::Graph<T> &g) const {
    auto *const c = compiler();
    if (c == nullptr || objects_.empty()) {
      return nullptr;
    }
    const auto digest = rt::digest(g);
    const auto it = std::ranges::find_if(objects_, [&](const rt::Object &o) {
      return o.want == static_cast<std::uint8_t>(want) && o.digest == digest &&
             o.host == c->host_identity() &&
             rt::same_codegen(o.options, effective_options()) &&
             !o.code.empty();
    });
    return it == objects_.end() ? nullptr : &*it;
  }
#endif

  // The roots as well as the digest: two equations over one arena share every
  // node and differ only in which ids they call outputs.
  [[nodiscard]] bool describes(const rt::Snapshot<T> &snap) const {
    return snap.roots.size() == output_dim &&
           std::ranges::equal(snap.roots, roots_) &&
           rt::digest<T>(snap.symbols, snap.nodes, snap.model_nodes) ==
               rt::digest<T>(arena_->symbols(), arena_->nodes(), model_nodes_);
  }

  // Poisoned: arena_ null, roots_ empty, every accessor short-circuiting first.
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
    // Before the sweeps, which append: what a saved file is keyed on.
    model_nodes_ = static_cast<std::uint32_t>(arena_->size());
    derivative_ = rt::build_jacobian_impl(*arena_, roots_);
    // Boost.Graph colouring and a Cache's mutex: neither is available while
    // constant-evaluating, and neither is reached there.
    if !consteval {
#ifdef DDX_HAS_JIT
      // Here rather than at the freeze: appending to a borrowed arena is the
      // constructor's alone, and prepare() runs under a lane lock.
      compile_roots_ = rt::detail::rebalance(*arena_, roots_);
      compile_derivative_ = rt::build_jacobian_impl(*arena_, compile_roots_);
#endif
      hessians_ = roots_ | std::views::transform([this](rt::NodeId r) {
                    return rt::build_hessian_impl(*arena_, r);
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

  // The snapshot's node array is installed verbatim, not replayed: make() folds
  // and would renumber the ids the saved sweeps name.
  explicit Equation(rt::Snapshot<T> &&snap)
      : arena_(rt::rebuild(snap).release(), reclaim),
        roots_(std::move(snap.roots)), model_nodes_(snap.model_nodes),
        derivative_(std::move(snap.jacobian)),
        hessians_(std::move(snap.hessians)), loaded_(true) {
#ifdef DDX_HAS_JIT
    options_ = snap.options;
    objects_ = std::move(snap.objects);
#endif
    cache_ = std::make_unique<Cache>();
  }

  // Poison first, and by the code it carries: a sealed or absent arena is a
  // different mistake from a literal that reached no graph.
  [[nodiscard]] static constexpr std::optional<error>
  why_not(const rt::RTExpression<T> &first, const Rest &...rest) noexcept {
    for (const auto why : {first.why(), rest.why()...}) {
      if (why) {
        return error{*why};
      }
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

  // At one point every column is one value, so a column pointer is its address.
  template <typename U>
  [[nodiscard]] static auto single_columns(std::span<U> values) {
    boost::container::small_vector<U *, 32> out(values.size());
    std::ranges::transform(values, out.begin(), [](U &v) { return &v; });
    return out;
  }

  // One point through the batch path at n = 1, not the arena walk -- which would
  // compute the Hessian the constructor swept in for a caller wanting a
  // gradient.  Constant evaluation keeps it, having no Cache.
  constexpr void gather(Want want, const std::vector<T> &at, std::span<T> f,
                        std::span<T> g = {}, std::span<T> h = {}) const {
    if !consteval {
      const auto xs = single_columns(std::span<const T>{at});
      const auto fs = single_columns(f);
      const auto gs = single_columns(g);
      const auto hs = single_columns(h);
      (void)dispatch(*snapshot(want, 1), as_columns(xs), as_columns(fs),
                     as_columns(gs), as_columns(hs), 1);
      return;
    }
    auto wanted = roots_;
    impl::append(wanted, derivative_.partial);
    const auto values = rt::evaluate_reachable(*arena_, wanted, at);
    const auto pick = [&values](std::span<T> out, const auto &nodes) {
      std::ranges::transform(nodes | std::views::take(out.size()), out.begin(),
                             [&values](rt::NodeId v) { return values[v]; });
    };
    pick(f, roots_);
    pick(g, derivative_.partial);
  }

  // Published, never written: a kernel arrives as a *new* Compiled.
  struct Compiled {
    // Shared, so the graph outlives an Equation that went away mid-compile.
    std::shared_ptr<const rt::Graph<T>> graph;
#ifdef DDX_HAS_JIT
    // Not what the sweep walks: spines are blocked for the kernel, where the
    // chain is latency, and left alone for the sweep, where the same rewrite
    // costs tape locality.  Aliases `graph` where they agree.
    std::shared_ptr<const rt::Graph<T>> compile_graph{};
    jit::Kernel kernel{};
    // Rungs share one pool and need not land in the order they were asked for,
    // so this is what refuses a late one.
    unsigned level = 0;
#endif
  };

#ifdef DDX_HAS_JIT
  // `level` is also the rank: a higher rung may replace a lower one, never the
  // other way about.
  struct Rung {
    std::shared_future<jit::result<jit::Kernel>> pending; // set once, then read
    unsigned level = 0;
  };
#endif

  // One shape of graph, filled once and never invalidated
  struct Lane {
    mutable std::shared_mutex mutex;
    std::shared_ptr<const Compiled> ready;
#ifdef DDX_HAS_JIT
    // Submission order, so rungs[0] answers soonest.
    std::array<Rung, 2> rungs;
    // Adapt's counter, and how many rungs it has already bought.  Outside the
    // mutex because every batch call touches it and only a crossing needs the
    // write lock; relaxed because the count orders nothing -- climb() re-reads
    // both under the lock, which is what makes the launch happen once.  At 2
    // there is nothing left to earn and count() stops touching the line.
    mutable std::atomic<std::size_t> points{0};
    mutable std::atomic<unsigned> asked{0};
#endif
  };

  struct Cache {
    Lane values;       // the roots alone
    Lane plain;        // values and Jacobian
    Lane with_hessian; // and the compressed Hessian columns
  };

  [[nodiscard]] Lane &lane_for(Want want) const {
    switch (want) {
    case Want::Values:
      return cache_->values;
    case Want::Jacobian:
      return cache_->plain;
    default:
      return cache_->with_hessian;
    }
  }

  // Ownership, not a reference: a reader holds its share for the whole call, so
  // publishing cannot pull the graph out from under a dispatch() in flight.
  //
  // `n` is the batch about to be run and is what Adapt counts; an observer asks
  // for the lane without buying anything, hence the default.
  [[nodiscard]] std::shared_ptr<const Compiled> snapshot(Want want,
                                                         std::size_t n = 0)
      const {
    Lane &lane = lane_for(want);
    const bool earned = count(lane, n);
    {
      const std::shared_lock read{lane.mutex};
      if (lane.ready && !earned && !settling(lane)) {
        return lane.ready;
      }
    }
    const std::unique_lock fill{lane.mutex}; // another thread may have won
    if (!lane.ready) {
      prepare(lane, want);
    }
    // No call waits for a compile: until one lands the graph is swept, and the
    // kernel replaces the sweep the moment it arrives.
    if (arrived(lane)) {
      adopt(lane);
    }
    climb(lane);
    return lane.ready;
  }

  // Charge the lane for the batch and say whether that bought a rung.  False
  // for everything but Adapt, and false once both rungs are spoken for, so a
  // settled lane never writes to the counter at all.
  [[nodiscard]] bool count([[maybe_unused]] const Lane &lane,
                           [[maybe_unused]] std::size_t n) const {
#ifdef DDX_HAS_JIT
    if constexpr (std::same_as<T, double>) {
      if (n == 0 || bad_ || options_.backend != jit::Backend::Adapt) {
        return false;
      }
      const unsigned asked = lane.asked.load(std::memory_order_relaxed);
      if (asked >= 2) {
        return false;
      }
      return lane.points.fetch_add(n, std::memory_order_relaxed) + n >=
             rung_at(asked);
    }
#endif
    return false;
  }

#ifdef DDX_HAS_JIT
  // Where the next rung is bought, counting from nothing: hot_points is the
  // batch the *cheap* rung has to earn, so it is added to what came before.
  // Saturating, because a caller spells "never" as a huge threshold.
  [[nodiscard]] std::size_t rung_at(unsigned asked) const noexcept {
    constexpr auto ceiling = std::numeric_limits<std::size_t>::max();
    const std::size_t warm = options_.warm_points;
    const std::size_t hot = options_.hot_points;
    return asked == 0 ? warm : (warm > ceiling - hot ? ceiling : warm + hot);
  }
#endif

  // Ask for the rung the counter has bought.  Under the write lock and
  // idempotent: two threads may both arrive owing one, and the second re-reads
  // an `asked` the first has already moved.
  void climb([[maybe_unused]] Lane &lane) const {
#ifdef DDX_HAS_JIT
    if constexpr (std::same_as<T, double>) {
      if (bad_ || options_.backend != jit::Backend::Adapt) {
        return;
      }
      const unsigned asked = lane.asked.load(std::memory_order_relaxed);
      if (asked >= 2 ||
          lane.points.load(std::memory_order_relaxed) < rung_at(asked)) {
        return;
      }
      auto *const c = compiler();
      if (c == nullptr) {
        // No ladder on this host, and the sweep answers everything: stop
        // counting rather than take the write lock on every call from here on.
        lane.asked.store(2, std::memory_order_relaxed);
        return;
      }
      const jit::Options top = effective_options();
      if (asked == 0 && top.codegen_level > 0) {
        jit::Options low = top;
        low.codegen_level = 0;
        lane.rungs[0] = {c->compile_async(lane.ready->graph, low), 0};
        lane.asked.store(1, std::memory_order_relaxed);
        return;
      }
      // Either the top rung over a cheap one, or -- at codegen 0 -- the only
      // rung there is.
      lane.rungs[asked == 0 ? 0 : 1] = {
          c->compile_async(lane.ready->graph, top), top.codegen_level};
      lane.asked.store(2, std::memory_order_relaxed);
    }
#endif
  }

  // Whether the lane owes the reader anything: only ever a kernel to publish.
  [[nodiscard]] bool settling(const Lane &lane) const { return arrived(lane); }

  // Republish rather than write into what a reader holds; a rung that lost the
  // race is dropped rather than allowed to demote a live kernel.
  void adopt([[maybe_unused]] Lane &lane) const {
#ifdef DDX_HAS_JIT
    jit::Kernel best;
    unsigned level = 0;
    for (Rung &rung : lane.rungs) {
      if (!ready(rung)) {
        continue;
      }
      // A refused compile leaves the kernel empty, and the sweep stays.
      const auto &landed = rung.pending.get();
      if (landed && (!best || rung.level > level)) {
        best = *landed;
        level = rung.level;
      }
      rung.pending = {};
    }
    if (best && (!lane.ready->kernel || level > lane.ready->level)) {
      lane.ready = std::make_shared<const Compiled>(
          Compiled{.graph = lane.ready->graph,
                   .compile_graph = lane.ready->compile_graph,
                   .kernel = std::move(best),
                   .level = level});
    }
#endif
  }

#ifdef DDX_HAS_JIT
  // Polling a shared_future from many threads is well-defined, and a rung is
  // written once under the write lock.
  [[nodiscard]] static bool ready(const Rung &rung) {
    using namespace std::chrono_literals;
    return rung.pending.valid() &&
           rung.pending.wait_for(0s) == std::future_status::ready;
  }
#endif

  // Whether any rung has finished but not yet been collected.
  [[nodiscard]] static bool arrived([[maybe_unused]] const Lane &lane) {
#ifdef DDX_HAS_JIT
    return std::ranges::any_of(lane.rungs, ready);
#else
    return false;
#endif
  }

  // Freeze the lane's graph and, under a compiling backend, put the compile in
  // flight.  Every derivative was swept in the constructor, so this only reads
  // the arena.  Under the lane's write lock.
  //
  // One step deliberately: split, a lane could be `ready` having compiled
  // nothing, and the next caller would interpret forever with every answer
  // still right.
  void prepare(Lane &lane, Want want) const {
    if (bad_) {
      lane.ready = std::make_shared<const Compiled>(
          Compiled{.graph = std::make_shared<const rt::Graph<T>>()});
      return;
    }
    rt::GraphBuilder<T> gb{*arena_};
    gb.values_from(roots_);
    if (want != Want::Values) {
      gb.jacobian_from(derivative_.partial);
    }
    if (want == Want::Hessian) {
      gb.hessian_from(hessians_.front());
    }
    auto swept = std::make_shared<const rt::Graph<T>>(gb.build(contracts()));
#ifdef DDX_HAS_JIT
    // The Hessian lane shares one graph: blocking it would cost a second
    // colouring and reverse-over-reverse for a lane the ladder never climbs.
    auto compiled = swept;
    if constexpr (std::same_as<T, double>) {
      if (want != Want::Hessian && !compile_roots_.empty()) {
        rt::GraphBuilder<T> cb{*arena_};
        cb.values_from(compile_roots_);
        if (want != Want::Values) {
          cb.jacobian_from(compile_derivative_.partial);
        }
        compiled = std::make_shared<const rt::Graph<T>>(cb.build(contracts()));
      }
    }
    lane.ready = std::make_shared<const Compiled>(
        Compiled{.graph = swept, .compile_graph = compiled});
#else
    lane.ready = std::make_shared<const Compiled>(Compiled{.graph = swept});
#endif
#ifdef DDX_HAS_JIT
    if constexpr (std::same_as<T, double>) {
      // A compiler this host cannot give is not an error a caller handles --
      // run() falls back to interpret().
      auto *const c =
          options_.backend != jit::Backend::Interpret ? compiler() : nullptr;
      if (c == nullptr) {
        return;
      }
      // A rung already climbed: published at its own level, with the ladder
      // launched as always and adopt()'s rank dropping what cannot beat it.
      if (const rt::Object *const have = stored(want, *lane.ready->compile_graph)) {
        // Shapes come from the graph just frozen, never the file: a forged
        // entry supplies code and a symbol, never a column count.
        const auto &layout = lane.ready->compile_graph->layout();
        if (auto adopted = c->adopt(have->code, have->symbol,
                                    lane.ready->compile_graph->symbols().size(),
                                    layout.values, layout.jacobian,
                                    layout.hessian)) {
          const unsigned level = have->options.codegen_level;
          lane.ready = std::make_shared<const Compiled>(
              Compiled{.graph = lane.ready->graph,
                       .compile_graph = lane.ready->compile_graph,
                       .kernel = std::move(*adopted),
                       .level = level});
          // Nothing left to climb to, so nothing is launched and, under Adapt,
          // nothing more is counted.
          if (level >= effective_options().codegen_level) {
            lane.asked.store(2, std::memory_order_relaxed);
            return;
          }
          // A cheap rung came off the file for nothing, so Adapt has only the
          // top one left to earn.
          lane.asked.store(1, std::memory_order_relaxed);
        }
        // A refused link is a miss, never a failure: the compile below stands.
      }
      // Adapt buys its rungs in climb(); Compile asks for both outright.
      if (options_.backend == jit::Backend::Compile) {
        launch(lane, *c);
      }
    }
#endif
  }

#ifdef DDX_HAS_JIT
  // Cheapest rung first, so it is also first in the pool's queue: codegen 0
  // lands sooner for a slower kernel, and every level agrees to the bit.  At
  // codegen 0 there is nothing cheaper underneath, so there is one rung.
  void launch(Lane &lane, jit::Compiler &c) const {
    const jit::Options top = effective_options();
    std::size_t next = 0;
    if (top.codegen_level > 0) {
      jit::Options low = top;
      low.codegen_level = 0;
      lane.rungs[next++] = {c.compile_async(lane.ready->compile_graph, low), 0};
    }
    lane.rungs[next] = {c.compile_async(lane.ready->compile_graph, top),
                        top.codegen_level};
  }

  // How wide to emit, from the batch the caller stated: a wide kernel computes
  // `w` points to answer for one, so a short batch emits scalar -- the same
  // threshold the sweep uses.  A stated `lanes` is honoured.
  [[nodiscard]] jit::Options effective_options() const noexcept {
    jit::Options opt = options_;
    if (opt.lanes == 0 && opt.points < kLanes) {
      opt.lanes = 1;
    }
    return opt;
  }

  // A Kernel does not own its code, so the compiler outlives every kernel.
  // Null on a host with no JIT.
  static jit::Compiler *compiler() { return rt_detail::shared_compiler(); }
#endif

  // Points per block sweep: two AVX2 registers of doubles.
  static constexpr std::size_t kLanes = 8;

  // Decided in the *graph*, so sweep and kernel fold the same products.  A
  // JIT-less build contracts too, so an answer does not depend on how the
  // library was configured.
  [[nodiscard]] bool contracts() const noexcept {
#ifdef DDX_HAS_JIT
    return options_.contract;
#else
    return true;
#endif
  }

  void interpret(const Compiled &c, std::span<const T *const> xs,
                 std::span<T *const> f, std::span<T *const> g,
                 std::span<T *const> h, std::size_t n) const {
    const auto blocks = c.graph->output_blocks();
    // The freeze settled both, so there is no choice left here.
    const auto order = c.graph->contracted_order();
    const auto contractions = c.graph->contractions();
    const std::size_t symbols = symbol_count();

    // Tapes are sized by the arena rather than the graph: a borrowed builder
    // may have grown since the freeze, and live ids index below that.
    //
    // A short batch sweeps one point at a time rather than padding out to
    // kLanes, which on a batch of one is the whole cost.
    if (n < kLanes) {
      std::vector<T> at(symbols);
      std::vector<T> tape(arena_->size());
      for (const std::size_t i : std::views::iota(0uz, n)) {
        std::ranges::transform(xs, at.begin(),
                               [i](const T *column) { return column[i]; });
        rt::evaluate_into(*arena_, at, order, contractions, std::span<T>{tape});
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
    // final block repeats its last point, and those lanes are never read back.
    std::vector<T> lanes(symbols * kLanes);
    std::vector<T> tape(arena_->size() * kLanes);

    for (std::size_t base = 0; base < n; base += kLanes) {
      const std::size_t width = std::min(kLanes, n - base);
      for (const auto [j, column] : std::views::enumerate(xs)) {
        T *const dst = lanes.data() + static_cast<std::size_t>(j) * kLanes;
        T *const tail = std::ranges::copy_n(column + base, width, dst).out;
        std::ranges::fill(tail, dst + kLanes, column[base + width - 1]);
      }
      rt::evaluate_block<kLanes>(*arena_, std::span<const T>{lanes}, order,
                                 contractions, std::span<T>{tape});

      for (const auto [columns, block] : std::views::zip(
               std::array{f, g, h},
               std::array{blocks.values, blocks.jacobian, blocks.hessian})) {
        for (const auto [column, o] : std::views::zip(columns, block)) {
          std::ranges::copy_n(tape.data() + std::size_t{o} * kLanes, width,
                              column + base);
        }
      }
    }
  }

  static auto as_columns(const std::ranges::contiguous_range auto &r) {
    using Ptr = std::ranges::range_value_t<std::remove_cvref_t<decltype(r)>>;
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

  // The deleter is the only difference between a borrowed arena and an owned one.
  using ArenaPtr = std::unique_ptr<rt::Builder<T>, void (*)(rt::Builder<T> *)>;
  static constexpr void borrow(rt::Builder<T> *) noexcept {}
  static constexpr void reclaim(rt::Builder<T> *b) noexcept { delete b; }

  ArenaPtr arena_{nullptr, borrow};
  std::vector<rt::NodeId> roots_;
  // Where the model ends and the sweeps begin.  A file is keyed on the arena up
  // to here, so rebuilding the model reproduces the key.
  std::uint32_t model_nodes_ = 0;

#ifdef DDX_HAS_JIT
  jit::Options options_{};
  // Machine code a file handed over, consulted once per lane by prepare().
  std::vector<rt::Object> objects_{};
#endif
  // Eager: one reverse sweep is microseconds, and it keeps every per-point
  // accessor const and constexpr.
  rt::Jacobian derivative_;
  // Swept in the constructor and never touched again, which is what makes the
  // const members safe to call concurrently: rt::build_hessian_impl *appends to
  // the arena*, and a borrowed arena may be lent to another Equation too.
  std::vector<rt::Hessian> hessians_;
#ifdef DDX_HAS_JIT
  std::vector<rt::NodeId> compile_roots_;
  rt::Jacobian compile_derivative_;
#endif
  // unique_ptr, not shared: its destructor is constexpr, which is what keeps a
  // constant-evaluated Equation destructible.  Null only there.
  std::unique_ptr<Cache> cache_;
  // Set exactly when why_not() refused, so poisoned() <=> arena_ == nullptr.
  std::optional<error> bad_{};
  bool loaded_ = false;
};

} // namespace ddx::impl

namespace ddx::rt {

#ifdef DDX_HAS_JIT
using jit::Backend;
#endif

// Over expressions already built in a caller's own arena.  Partial
// specialisations contribute no deduction guides, so this is the whole of CTAD.
template <impl::Numeric T, typename... Ts>
  requires(std::same_as<Ts, RTExpression<T>> && ...)
[[nodiscard]] constexpr auto equation(RTExpression<T> first, Ts... rest) {
  return impl::Equation<RTExpression<T>, Ts...>::create(first, rest...);
}

// Build an Equation:
//
//   const auto eq = ddx::rt::equation([] {
//     const auto x = var("x");
//     const auto y = var("y");
//     return exp(x) * sin(y);
//   });

template <impl::Numeric T = double>
[[nodiscard]] auto equation(std::invocable auto &&assemble) {
  auto arena = std::make_unique<Builder<T>>();
  auto built = [&] {
    const auto scope = detail::scoped_arena(*arena);
    return std::invoke(std::forward<decltype(assemble)>(assemble));
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

// An equation from a file and nothing else: symbols, arena, Jacobian and
// colouring all come off disk, and no sweep runs.  The output count is part of
// the type, so a file holding some other number is refused, not adapted to.
//
//   const auto eq = ddx::rt::load("f.ddx");        // one function
//   const auto eq = ddx::rt::load<double, 3>("f.ddx");
template <impl::Numeric T = double, std::size_t Outputs = 1>
  requires std::floating_point<T>
[[nodiscard]] auto load(const std::filesystem::path &path) {
  static_assert(Outputs > 0, "load: a system needs at least one function");
  return impl::index_apply<Outputs - 1>([&]<std::size_t... Rest>() {
    return impl::Equation<RTExpression<T>, detail::Repeat<Rest, T>...>::load(
        path);
  });
}

// The same model, with a file to keep the work in:
//
//   const auto eq = ddx::rt::equation("f.ddx", [] {
//     const auto x = var("x");
//     return exp(x) * x;
//   });
//
// The first run builds, sweeps and writes; a later one rebuilds the arena --
// interning, which is cheap -- and takes the sweeps and the colouring off disk.
// Editing the model changes the key, so the file is rebuilt, not trusted.  It
// never refuses; eq.loaded() says which happened.
template <impl::Numeric T = double>
  requires std::floating_point<T>
[[nodiscard]] auto equation(const std::filesystem::path &path,
                            std::invocable auto &&assemble) {
  auto arena = std::make_unique<Builder<T>>();
  auto built = [&] {
    const auto scope = detail::scoped_arena(*arena);
    return std::invoke(std::forward<decltype(assemble)>(assemble));
  }();

  using Built = std::remove_cvref_t<decltype(built)>;
  if constexpr (std::same_as<Built, RTExpression<T>>) {
    return impl::Equation<RTExpression<T>>::cached(path, std::move(arena),
                                                   built);
  } else {
    constexpr std::size_t outputs = std::tuple_size_v<Built>;
    static_assert(outputs > 0,
                  "equation: a system needs at least one function");
    return impl::index_apply<outputs - 1>([&]<std::size_t... Rest>() {
      return impl::Equation<RTExpression<T>, detail::Repeat<Rest, T>...>::
          cached(path, std::move(arena), built[0], built[Rest + 1]...);
    });
  }
}

} // namespace ddx::rt
