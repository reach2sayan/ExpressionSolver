#pragma once

#include "dual/taylor_dual.hpp" // the univariate sweep's scalar
#include "md/md.hpp"
#include "ops/numeric.hpp" // compile_time_factorial
#include "rt/archive/archive.hpp"
#include "rt/cache.hpp"
#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/expressions.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "rt/rebalance.hpp"
#include "symbolic/equation.hpp"
#include "util/config.hpp"
#include "util/ranges.hpp" // to<C>() and append(), ours

#include <boost/container/small_vector.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

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
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Keyed on the RTExpression<T> *pattern*: a `requires` over <TFirst, TRest...>
// would be ambiguous with the compile-time specialisation.
namespace ddx::impl {

namespace rt_detail {

// A point keyed by name at run time: any range of (name, value), which is the
// map spelling of the names rt::var(name) reads.  Entry keys off a FixedString
// and so cannot say a name the program only learns while it runs.
template <typename R, typename T>
concept CNamedRange =
    std::ranges::input_range<R> &&
    requires(const std::ranges::range_value_t<R> &e) {
      { e.first } -> std::convertible_to<std::string_view>;
      requires Numeric<std::remove_cvref_t<decltype(e.second)>>;
    };

template <typename A, typename T>
concept CPointArg =
    Numeric<std::remove_cvref_t<A>> || CEntry<A> || CNamedRange<A, T> ||
    (std::ranges::input_range<A> &&
     Numeric<std::remove_cvref_t<std::ranges::range_value_t<A>>>);

// CPointArg admits any input_range, so without this a span of columns would try
// to become a scalar.
template <typename R, typename Ptr>
concept CColumns = std::ranges::contiguous_range<R> &&
                   std::convertible_to<std::ranges::range_value_t<R>, Ptr>;

// What an Equation is spelled over: its outputs, and behind them the cache a
// factory deduced from what it was handed.  Which of the two a trailing type is
// is asked of the type system, never assumed by position: a cache is whatever
// models CValueCache, and COutputPack refuses a pack holding anything else.
template <typename... Rest>
using tail_of = boost::mp11::mp_eval_if_c<sizeof...(Rest) == 0, void,
                                          boost::mp11::mp_back,
                                          boost::mp11::mp_list<Rest...>>;

template <typename T, typename... Rest>
inline constexpr bool packs_cache = rt::CValueCache<tail_of<Rest...>, T>;

template <typename T, typename... Rest>
using cache_of = boost::mp11::mp_if_c<packs_cache<T, Rest...>, tail_of<Rest...>,
                                      rt::detail::NoCache<T>>;

template <typename T, typename... Rest>
using outputs_of = boost::mp11::mp_eval_if_c<
    !packs_cache<T, Rest...>, boost::mp11::mp_list<Rest...>,
    boost::mp11::mp_pop_back, boost::mp11::mp_list<Rest...>>;

template <typename T> struct is_output {
  template <typename U> using fn = std::is_same<U, rt::RTExpression<T>>;
};

// Outputs, then at most one cache, and nothing else.
template <typename T, typename... Rest>
concept COutputPack =
    boost::mp11::mp_all_of_q<outputs_of<T, Rest...>, is_output<T>>::value;

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
  requires rt_detail::COutputPack<T, Rest...>
class Equation<rt::RTExpression<T>, Rest...>
    : public rt::detail::Caching<Equation<rt::RTExpression<T>, Rest...>, T,
                                 rt_detail::cache_of<T, Rest...>> {
  using Want = rt::Want;
  using Cache = rt_detail::cache_of<T, Rest...>;
  using Memo = rt::detail::Caching<Equation, T, Cache>;
  // Reaches arena() and unlocked(); neither is anyone else's business.
  friend Memo;

public:
  using value_type = T;
  static constexpr std::size_t output_dim =
      1 + boost::mp11::mp_size<rt_detail::outputs_of<T, Rest...>>::value;

  // The outputs as their own pack: the class's holds the cache too, and a
  // factory hands over functions, never a cache among them.
  template <typename... Es>
  static constexpr bool are_outputs =
      sizeof...(Es) + 1 == output_dim &&
      (std::same_as<Es, rt::RTExpression<T>> && ...);

  // A refusal rides on the Equation rather than a result<Equation> at every
  // call site; see poisoned().  The cache leads, because the functions are a
  // pack and nothing can be deduced behind one.
  template <typename... Es>
    requires are_outputs<Es...>
  [[nodiscard]] static constexpr Equation create(Cache memo,
                                                 rt::RTExpression<T> first,
                                                 Es... rest) {
    if (const auto bad = why_not(first, rest...)) {
      return Equation{*bad};
    }
    first.builder()->seal();
    Equation eq{first, rest...};
    eq.take_cache(std::move(memo));
    return eq;
  }

  // Owning: the Builder is heap-allocated, so the nodes survive the move.
  template <typename... Es>
    requires are_outputs<Es...>
  [[nodiscard]] static Equation create(Cache memo,
                                       std::unique_ptr<rt::Builder<T>> owned,
                                       rt::RTExpression<T> first, Es... rest) {
    if (const auto bad = why_not(first, rest...)) {
      return Equation{*bad};
    }
    first.builder()->seal();
    Equation eq{std::move(owned), first, rest...};
    eq.take_cache(std::move(memo));
    return eq;
  }

  // Build the model, then take the sweeps off disk if the file still describes
  // it.  An absent or stale file rebuilds rather than refusing; loaded() tells.
  template <typename... Es>
    requires are_outputs<Es...> && std::floating_point<T>
  [[nodiscard]] static Equation cached(Cache memo,
                                       const std::filesystem::path &path,
                                       std::unique_ptr<rt::Builder<T>> owned,
                                       rt::RTExpression<T> first, Es... rest) {
    if (const auto why = why_not(first, rest...)) {
      return Equation{*why};
    }
    if (auto snap = rt::load_snapshot<T>(path);
        snap && describes(**snap, *owned, roots_of(*owned, first, rest...),
                          owned->size())) {
      Equation eq{std::move(*snap)};
      eq.take_cache(std::move(memo));
      return eq;
    }
    Equation eq = create(std::move(memo), std::move(owned), first, rest...);
    // A file that cannot be written is still a working equation.
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
    return unless_poisoned([this] { return symbol_count(); });
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
      std::vector<bool> seen(at.size(), false);
      ((ok = ok.and_then([&] { return assign_named(at, seen, args); })), ...);
      ok = ok.and_then([&] { return whole(seen); });
    } else if constexpr (sizeof...(Args) == 1 &&
                         (rt_detail::CNamedRange<Args, T> && ...)) {
      // Before the plain range: a map is an input_range too, and its elements
      // are pairs rather than the numbers assign_range would read.
      ok = assign_named_range(at, args...);
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
      gather(Want::Value, at, std::span<T>{f});
      if constexpr (output_dim == 1) {
        return f[0];
      } else {
        return std::vector<T>(f.begin(), f.end());
      }
    });
  }

  // Dense row-major m x n; at m == 1 the leading axis goes and this is n long.
  // The graph holds only the cells the pattern names, so the zeros go back in
  // here and this spelling never meets the compression.
  [[nodiscard]] constexpr result<std::vector<T>>
  jacobian(const rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      std::array<T, output_dim> f{};
      std::vector<T> cells(derivative_.partial.size());
      gather(Want::Jacobian, at, std::span<T>{f}, cells);
      return dense(cells);
    });
  }

  // The Jacobian alone, from a graph with no value block: whatever only the
  // value needs is not swept, which on a model whose derivative is simpler
  // than its function is most of the graph.
  [[nodiscard]] constexpr result<std::vector<T>>
  gradient(const rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      std::vector<T> cells(derivative_.partial.size());
      gather(Want::Gradient, at, {}, cells);
      return dense(cells);
    });
  }

  // H(x)v, n long.  The direction leads and is a span, so it cannot be read as
  // a point value however the point itself is spelled.
  [[nodiscard]] result<std::vector<T>>
  hvp(std::span<const T> direction,
      const rt_detail::CPointArg<T> auto &...args) const
    requires(output_dim == 1)
  {
    return seeded<Want::Hvp>(direction, args...);
  }

  // The j-th Hessian column, named and not indexed: the symbol list exists
  // only at run time.
  [[nodiscard]] result<std::vector<T>>
  hvp(std::string_view along, const rt_detail::CPointArg<T> auto &...args) const
    requires(output_dim == 1)
  {
    if (bad_) {
      return std::unexpected{*bad_};
    }
    const auto slot = slot_of(along);
    if (!slot) {
      return fail(errc::unknown_symbol);
    }
    std::vector<T> unit(symbol_count(), T{0});
    unit[*slot] = T{1};
    return hvp(std::span<const T>{unit}, args...);
  }

  // w'J, n long: one weight per function.
  [[nodiscard]] result<std::vector<T>>
  vjp(std::span<const T> weights,
      const rt_detail::CPointArg<T> auto &...args) const {
    return seeded<Want::Vjp>(weights, args...);
  }

  // J v, m long: the mirror of vjp().
  [[nodiscard]] result<std::vector<T>>
  jvp(std::span<const T> direction,
      const rt_detail::CPointArg<T> auto &...args) const {
    return seeded<Want::Jvp>(direction, args...);
  }

  // Dense row-major m x n x n; the graph holds it compressed by colour.
  [[nodiscard]] result<std::vector<T>>
  hessian(const rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      const std::size_t n = symbol_count();
      std::vector<T> out(output_dim * n * n);
      const impl::md::mdspan dense{
          out.data(), impl::md::dextents<std::size_t, 3>{output_dim, n, n}};
      // Only one root's Hessian is frozen into a lane; a system walks the
      // arena.
      if constexpr (output_dim == 1) {
        const rt::Coloring &coloring = hessians_.front().coloring;
        std::array<T, 1> f{};
        std::vector<T> g(derivative_.partial.size());
        std::vector<T> h(hessians_.front().compressed.size());
        gather(Want::Hessian, at, std::span<T>{f}, g, h);
        // Only the cells the colouring stores; `out` is already zero, which is
        // what every other column of that colour is.
        for (const rt::Cell &cell : coloring.entries()) {
          dense[0uz, cell.row, cell.column] = h[cell.slot];
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
          for (const rt::Cell &cell : block.coloring.entries()) {
            dense[static_cast<std::size_t>(k), cell.row, cell.column] =
                values[block.compressed[cell.slot]];
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
    lanes_ = std::make_unique<Lanes>();
    // codegen decides the arithmetic, so what was remembered was answered by
    // another graph.
    this->cache().clear();
    // Choosing a backend starts the build, so it overlaps whatever the caller
    // does before their first call.
    if (opt.backend != jit::Backend::Interpret && !poisoned()) {
      Lane &lane = lane_for(Want::Jacobian);
      const std::unique_lock fill{lane.mutex};
      prepare(lane, Want::Jacobian);
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
  [[nodiscard]] std::optional<jit::Level> kernel_level() const {
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
      const Lane &lane = lane_for(Want::Jacobian);
      const std::shared_lock read{lane.mutex};
      if (const auto &first = lane.rungs.front(); first.pending.valid()) {
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
    const Lane &lane = lane_for(Want::Jacobian);
    const unsigned asked = lane.asked.load(std::memory_order_relaxed);
    return asked >= 2 ? std::nullopt : std::optional{Warmup {
      lane.points.load(std::memory_order_relaxed),
      rung_at(asked)
    }};
  }
#endif

  [[nodiscard]] std::optional<std::size_t> hessian_colors() const
    requires(output_dim == 1)
  {
    return unless_poisoned([this] { return hessians_.front().colors(); });
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
    const std::span<const rt::NodeId> root{roots_.data(), 1};
    // Refuse rather than sweep: an op the Taylor scalar lacks evaluates to a
    // zero node, which is a wrong number with nothing to say so.
    if (!rt::computes_at<Taylor>(*arena_, root)) {
      return fail(errc::unsupported_scalar);
    }
    Taylor seed;
    seed.c[0] = x0;
    seed.c[1] = T{1};

    const std::array<Taylor, 1> at{seed};
    // One root out of an arena holding a gradient and a Hessian too, and a
    // Taylor node costs (Order + 1) coefficients.
    const auto values = rt::evaluate_reachable(*arena_, root, at);
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
    return dispatch(Want::Value, *snapshot(Want::Value, n), as_columns(xs),
                    {.values = as_columns(f)}, n);
  }

  [[nodiscard]] result<void>
  jacobian(const rt_detail::CColumns<const T *> auto &xs,
           const rt_detail::CColumns<T *> auto &f,
           const rt_detail::CColumns<T *> auto &g, std::size_t n) const {
    return dispatch(Want::Jacobian, *snapshot(Want::Jacobian, n),
                    as_columns(xs),
                    {.values = as_columns(f), .jacobian = as_columns(g)}, n);
  }

  [[nodiscard]] result<void>
  gradient(const rt_detail::CColumns<const T *> auto &xs,
           const rt_detail::CColumns<T *> auto &g, std::size_t n) const {
    return dispatch(Want::Gradient, *snapshot(Want::Gradient, n),
                    as_columns(xs), {.jacobian = as_columns(g)}, n);
  }

  [[nodiscard]] result<void>
  hessian(const rt_detail::CColumns<const T *> auto &xs,
          const rt_detail::CColumns<T *> auto &f,
          const rt_detail::CColumns<T *> auto &g,
          const rt_detail::CColumns<T *> auto &h, std::size_t n) const
    requires(output_dim == 1)
  {
    return dispatch(Want::Hessian, *snapshot(Want::Hessian, n), as_columns(xs),
                    {.values = as_columns(f),
                     .jacobian = as_columns(g),
                     .hessian = as_columns(h)},
                    n);
  }

  // `v` is a second block of input columns, not a second point: one column per
  // symbol, each n long, spliced behind `xs`.  `g` comes out of the same sweep.
  [[nodiscard]] result<void> hvp(const rt_detail::CColumns<const T *> auto &xs,
                                 const rt_detail::CColumns<const T *> auto &v,
                                 const rt_detail::CColumns<T *> auto &f,
                                 const rt_detail::CColumns<T *> auto &g,
                                 const rt_detail::CColumns<T *> auto &hv,
                                 std::size_t n) const
    requires(output_dim == 1)
  {
    return dispatch(Want::Hvp, *snapshot(Want::Hvp, n),
                    as_columns(widen(xs, v)),
                    {.values = as_columns(f),
                     .jacobian = as_columns(g),
                     .hessian = as_columns(hv)},
                    n);
  }

  [[nodiscard]] result<void> vjp(const rt_detail::CColumns<const T *> auto &xs,
                                 const rt_detail::CColumns<const T *> auto &w,
                                 const rt_detail::CColumns<T *> auto &f,
                                 const rt_detail::CColumns<T *> auto &out,
                                 std::size_t n) const {
    return dispatch(Want::Vjp, *snapshot(Want::Vjp, n),
                    as_columns(widen(xs, w)),
                    {.values = as_columns(f), .jacobian = as_columns(out)}, n);
  }

  [[nodiscard]] result<void> jvp(const rt_detail::CColumns<const T *> auto &xs,
                                 const rt_detail::CColumns<const T *> auto &v,
                                 const rt_detail::CColumns<T *> auto &f,
                                 const rt_detail::CColumns<T *> auto &out,
                                 std::size_t n) const {
    return dispatch(Want::Jvp, *snapshot(Want::Jvp, n),
                    as_columns(widen(xs, v)),
                    {.values = as_columns(f), .jacobian = as_columns(out)}, n);
  }

  // What a caller sizes buffers by, read off the constructor's sweep rather
  // than a frozen graph: asking must not build a lane.
  [[nodiscard]] constexpr std::optional<std::size_t> value_columns() const {
    return unless_poisoned([this] { return roots_.size(); });
  }
  [[nodiscard]] constexpr std::optional<std::size_t> jacobian_columns() const {
    return unless_poisoned([this] { return derivative_.partial.size(); });
  }
  [[nodiscard]] std::optional<std::size_t> hessian_columns() const
    requires(output_dim == 1)
  {
    return unless_poisoned(
        [this] { return hessians_.front().compressed.size(); });
  }
  [[nodiscard]] std::optional<std::size_t> hvp_columns() const
    requires(output_dim == 1)
  {
    return unless_poisoned([this] { return hvp_.product.size(); });
  }
  [[nodiscard]] std::optional<std::size_t> vjp_columns() const {
    return unless_poisoned([this] { return vjp_.product.size(); });
  }
  [[nodiscard]] std::optional<std::size_t> jvp_columns() const {
    return unless_poisoned([this] { return jvp_.product.size(); });
  }

  // What a batch caller reads the two compressed blocks by: the dense
  // spellings undo the compression, the batch ones cannot, and without these
  // the column counts describe a block nobody can index.

  // `at(i, j)` is nullopt where d f[i] / d x[j] is structurally zero.
  [[nodiscard]] constexpr std::optional<
      std::reference_wrapper<const rt::Sparsity>>
  jacobian_pattern() const {
    return unless_poisoned([this] { return std::cref(derivative_.pattern); });
  }

  // Where H(i, j) sits, or nullopt for a cell the colouring calls zero.
  [[nodiscard]] std::optional<std::size_t> hessian_cell(std::size_t i,
                                                        std::size_t j) const
    requires(output_dim == 1)
  {
    if (poisoned() || hessians_.empty() || i >= symbol_count() ||
        j >= symbol_count()) {
      return std::nullopt;
    }
    return hessians_.front().coloring.cell_of(i, j);
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
    return rt::load_snapshot<T>(path).and_then(
        [this](const rt::Verified<T> &snap) {
          return describes(*snap) ? result<void>{}
                                  : fail(errc::archive_mismatch);
        });
  }

  // Nothing built and nothing swept; the model need not exist in this program.
  // The output count is in the type, so another number is refused.
  [[nodiscard]] static result<Equation> load(const std::filesystem::path &path,
                                             Cache memo = {})
    requires std::floating_point<T>
  {
    return rt::load_snapshot<T>(path).and_then(
        [&memo](rt::Verified<T> snap) -> result<Equation> {
          if (snap->roots.size() != output_dim) {
            return fail(errc::archive_mismatch);
          }
          Equation eq{std::move(snap)};
          eq.take_cache(std::move(memo));
          return eq;
        });
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
    snap.hvp = hvp_;
    snap.vjp = vjp_;
    snap.jvp = jvp_;
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
    for (const auto [want, lane] : std::views::zip(rt::want_values, *lanes_)) {
      const std::shared_lock read{lane.mutex};
      if (!lane.ready || !lane.ready->kernel ||
          lane.ready->kernel.object().empty()) {
        continue;
      }
      const auto code = lane.ready->kernel.object();
      out.push_back({.want = want,
                     .symbol = std::string{lane.ready->kernel.symbol()},
                     .host = std::string{c->host_identity()},
                     .digest = rt::digest(*lane.ready->graph),
                     .codegen = effective_options().codegen,
                     .code = {
                       code.begin(),
                       code.end()
                     }});
    }
    return out;
  }

  [[nodiscard]] const rt::Object *stored(Want want,
                                         const rt::Graph<T> &g) const {
    auto *const c = compiler();
    if (c == nullptr || objects_.empty()) {
      return nullptr;
    }
    return rt::find_object(objects_, want, rt::digest(g), c->host_identity(),
                           effective_options().codegen);
  }
#endif

  // The roots as well as the digest: two equations over one arena share every
  // node and differ only in which ids they call outputs.
  [[nodiscard]] static bool describes(const rt::Snapshot<T> &snap,
                                      const rt::Builder<T> &arena,
                                      std::span<const rt::NodeId> roots,
                                      std::size_t model_nodes) {
    return snap.roots.size() == output_dim &&
           std::ranges::equal(snap.roots, roots) &&
           rt::digest<T>(snap.symbols, snap.nodes, snap.model_nodes) ==
               rt::digest<T>(arena.symbols(), arena.nodes(), model_nodes);
  }
  [[nodiscard]] bool describes(const rt::Snapshot<T> &snap) const {
    return describes(snap, *arena_, roots_, model_nodes_);
  }

  template <typename... Es>
  [[nodiscard]] static constexpr std::vector<rt::NodeId>
  roots_of(rt::Builder<T> &b, rt::RTExpression<T> first, Es... rest) {
    return {first.id(b), rest.id(b)...};
  }

  // nullopt on a poisoned equation, `value()` otherwise.
  [[nodiscard]] constexpr auto unless_poisoned(auto &&value) const {
    using V = std::remove_cvref_t<std::invoke_result_t<decltype(value)>>;
    return poisoned() ? std::optional<V>{} : std::optional<V>{value()};
  }

  // The slot a symbol name occupies, or nullopt.
  [[nodiscard]] constexpr std::optional<std::size_t>
  slot_of(std::string_view name) const {
    const auto &names = arena_->symbols();
    const auto it = std::ranges::find(names, name);
    return it == names.end()
               ? std::nullopt
               : std::optional{static_cast<std::size_t>(it - names.begin())};
  }

  // The three seeded products: the point widened by `along` -- symbols first,
  // then the seeds, which is the order the ABI takes and input_column() states.
  template <Want W>
  [[nodiscard]] result<std::vector<T>>
  seeded(std::span<const T> along,
         const rt_detail::CPointArg<T> auto &...args) const {
    if (bad_) {
      return std::unexpected{*bad_};
    }
    if (along.size() != (W == Want::Vjp ? output_dim : symbol_count())) {
      return fail(errc::wrong_direction);
    }
    return point(args...).transform([this, along](const auto &at) {
      std::vector<T> widened = at;
      impl::append(widened, along);
      std::array<T, output_dim> f{};
      std::vector<T> g(W == Want::Hvp ? derivative_.partial.size() : 0);
      std::vector<T> out(W == Want::Jvp ? output_dim : symbol_count());
      if constexpr (W == Want::Hvp) {
        gather(W, widened, std::span<T>{f}, g, out);
      } else {
        gather(W, widened, std::span<T>{f}, out);
      }
      return out;
    });
  }

  // Poisoned: arena_ null, roots_ empty, every accessor short-circuiting first.
  constexpr explicit Equation(error why) : bad_(why) {
    if !consteval {
      lanes_ = std::make_unique<Lanes>();
    }
  }

  template <typename... Es>
    requires are_outputs<Es...>
  constexpr explicit Equation(rt::RTExpression<T> first, Es... rest)
      : arena_(first.builder(), borrow),
        roots_(roots_of(*arena_, first, rest...)) {
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
      // Here for the same reason as the rest of this block: prepare() is
      // const, runs under a lane lock, and these append.
      if constexpr (output_dim == 1) {
        hvp_ = rt::build_hvp_impl(*arena_, roots_.front());
      }
      vjp_ = rt::build_vjp_impl(*arena_, roots_);
      jvp_ = rt::build_jvp_impl(*arena_, roots_);
      lanes_ = std::make_unique<Lanes>();
    }
  }

  template <typename... Es>
    requires are_outputs<Es...>
  Equation(std::unique_ptr<rt::Builder<T>> owned, rt::RTExpression<T> first,
           Es... rest)
      : Equation(first, rest...) {
    arena_ = ArenaPtr{owned.release(), reclaim};
  }

  // The snapshot's node array is installed verbatim, not replayed: make() folds
  // and would renumber the ids the saved sweeps name.
  explicit Equation(rt::Verified<T> snap)
      : Equation(std::move(snap).rebuild()) {}
  explicit Equation(rt::Rebuilt<T> r)
      : arena_(r.arena.release(), reclaim), roots_(std::move(r.rest.roots)),
        model_nodes_(r.rest.model_nodes),
        derivative_(std::move(r.rest.jacobian)),
        hessians_(std::move(r.rest.hessians)), hvp_(std::move(r.rest.hvp)),
        vjp_(std::move(r.rest.vjp)), jvp_(std::move(r.rest.jvp)),
        loaded_(true) {
#ifdef DDX_HAS_JIT
    options_ = r.rest.options;
    objects_ = std::move(r.rest.objects);
#endif
    lanes_ = std::make_unique<Lanes>();
  }

  // Poison first, and by the code it carries: a sealed or absent arena is a
  // different mistake from a literal that reached no graph.
  template <typename... Es>
  [[nodiscard]] static constexpr std::optional<error>
  why_not(const rt::RTExpression<T> &first, const Es &...rest) noexcept {
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

  // Naming a point is not the same as giving one: a symbol no argument reaches
  // would otherwise be read as zero, which is a value and not a refusal.
  [[nodiscard]] constexpr result<void>
  whole(const std::vector<bool> &seen) const {
    return std::ranges::contains(seen, false) ? fail(errc::short_point)
                                              : result<void>{};
  }

  [[nodiscard]] constexpr result<void> assign_one(std::vector<T> &at,
                                                  std::vector<bool> &seen,
                                                  std::string_view name,
                                                  const auto &value) const {
    const auto slot = slot_of(name);
    if (!slot) {
      return fail(errc::unknown_symbol);
    }
    at[*slot] = static_cast<T>(value);
    seen[*slot] = true;
    return {};
  }

  template <CEntry V>
  [[nodiscard]] constexpr result<void>
  assign_named(std::vector<T> &at, std::vector<bool> &seen, const V &nv) const {
    return assign_one(at, seen, std::remove_cvref_t<V>::symbol.view(),
                      nv.value);
  }

  [[nodiscard]] constexpr result<void>
  assign_named_range(std::vector<T> &at,
                     const rt_detail::CNamedRange<T> auto &vm) const {
    std::vector<bool> seen(at.size(), false);
    for (const auto &entry : vm) {
      if (const auto ok =
              assign_one(at, seen, std::string_view{entry.first}, entry.second);
          !ok) {
        return ok;
      }
    }
    return whole(seen);
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

  // The point columns and the direction's, in one array: symbols first, then
  // the seeds, which is the order the ABI takes and input_column() states.
  [[nodiscard]] static auto widen(const auto &xs, const auto &seeds) {
    boost::container::small_vector<const T *, 64> out;
    out.reserve(std::ranges::size(xs) + std::ranges::size(seeds));
    std::ranges::copy(xs, std::back_inserter(out));
    std::ranges::copy(seeds, std::back_inserter(out));
    return out;
  }

  // One point through the batch path at n = 1, not the arena walk -- which
  // would compute the Hessian the constructor swept in for a caller wanting a
  // gradient.  Constant evaluation keeps it, having no Cache.
  constexpr void gather(Want want, const std::vector<T> &at, std::span<T> f,
                        std::span<T> g = {}, std::span<T> h = {}) const {
    if !consteval {
      const auto xs = single_columns(std::span<const T>{at});
      const auto fs = single_columns(f);
      const auto gs = single_columns(g);
      const auto hs = single_columns(h);
      (void)dispatch(want, *snapshot(want, 1), as_columns(xs),
                     {.values = as_columns(fs),
                      .jacobian = as_columns(gs),
                      .hessian = as_columns(hs)},
                     1);
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

  // Row-major m x n from the cells the pattern names, the zeros put back.
  [[nodiscard]] constexpr std::vector<T> dense(std::span<const T> cells) const {
    const std::size_t n = symbol_count();
    std::vector<T> out(output_dim * n, T{0});
    for (const auto [i, j, v] : derivative_.pattern.cells(cells)) {
      out[i * n + j] = v;
    }
    return out;
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
    jit::Level level = jit::Level::O0;

    // The same graphs under a kernel that just landed.
    [[nodiscard]] std::shared_ptr<const Compiled> with(jit::Kernel k,
                                                       jit::Level l) const {
      return std::make_shared<const Compiled>(
          Compiled{.graph = graph,
                   .compile_graph = compile_graph,
                   .kernel = std::move(k),
                   .level = l});
    }
#endif
  };

#ifdef DDX_HAS_JIT
  // `level` is also the rank: a higher rung may replace a lower one, never the
  // other way about.
  struct Rung {
    std::shared_future<jit::result<jit::Kernel>> pending; // set once, then read
    jit::Level level = jit::Level::O0;
  };

  // One rung at `level`, the options otherwise the caller's.
  [[nodiscard]] static Rung rung(jit::Compiler &c,
                                 const std::shared_ptr<const rt::Graph<T>> &g,
                                 jit::Options opt, jit::Level level) {
    opt.codegen.codegen_level = level;
    return {c.compile_async(g, opt), level};
  }
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

  // One lane per Want, in its order.
  using Lanes = std::array<Lane, rt::want_count>;

  [[nodiscard]] Lane &lane_for(Want want) const {
    return (*lanes_)[std::to_underlying(want)];
  }

  // Ownership, not a reference: a reader holds its share for the whole call, so
  // publishing cannot pull the graph out from under a dispatch() in flight.
  //
  // `n` is the batch about to be run and is what Adapt counts; an observer asks
  // for the lane without buying anything, hence the default.
  [[nodiscard]] std::shared_ptr<const Compiled>
  snapshot(Want want, std::size_t n = 0) const {
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
      if (asked == 0 && top.codegen.codegen_level > jit::Level::O0) {
        lane.rungs[0] = rung(*c, lane.ready->graph, top, jit::Level::O0);
        lane.asked.store(1, std::memory_order_relaxed);
        return;
      }
      // Either the top rung over a cheap one, or -- at codegen 0 -- the only
      // rung there is.
      lane.rungs[asked == 0 ? 0 : 1] =
          rung(*c, lane.ready->graph, top, top.codegen.codegen_level);
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
    jit::Level level = jit::Level::O0;
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
      lane.ready = lane.ready->with(std::move(best), level);
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
    // The gradient lane differentiates the roots without computing them: the
    // value is not an output, so the nodes only it needs are not live.
    if (want == Want::Gradient) {
      gb.roots_from(roots_);
    } else {
      gb.values_from(roots_);
    }
    // Vjp replaces the Jacobian block rather than adding to it: n columns
    // instead of the pattern's nonzeros is the whole point of asking.
    if (want == Want::Vjp) {
      gb.vector_jacobian_from(vjp_);
    } else if (want == Want::Jvp) {
      gb.tangent_from(jvp_);
    } else {
      if (want != Want::Value) {
        gb.jacobian_from(derivative_);
      }
      if (want == Want::Hessian) {
        gb.hessian_from(hessians_.front());
      }
      if (want == Want::Hvp) {
        gb.hessian_vector_from(hvp_);
      }
    }
    auto swept = std::make_shared<const rt::Graph<T>>(gb.finish(contracts()));
#ifdef DDX_HAS_JIT
    auto compiled = swept;
    if constexpr (std::same_as<T, double>) {
      if (rebalanceable(want) && !compile_roots_.empty()) {
        rt::GraphBuilder<T> cb{*arena_};
        if (want == Want::Gradient) {
          cb.roots_from(compile_roots_);
        } else {
          cb.values_from(compile_roots_);
        }
        if (want != Want::Value) {
          cb.jacobian_from(compile_derivative_);
        }
        compiled = std::make_shared<const rt::Graph<T>>(cb.finish(contracts()));
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
      if (const rt::Object *const have =
              stored(want, *lane.ready->compile_graph)) {
        // The shape comes from the graph just frozen, never the file: a
        // forged entry supplies code and a symbol, never a column count.
        if (auto adopted =
                c->adopt(have->code, have->symbol,
                         jit::KernelShape::of(*lane.ready->compile_graph))) {
          const jit::Level level = have->codegen.codegen_level;
          lane.ready = lane.ready->with(std::move(*adopted), level);
          // Nothing left to climb to, so nothing is launched and, under Adapt,
          // nothing more is counted.
          if (level >= effective_options().codegen.codegen_level) {
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
    if (top.codegen.codegen_level > jit::Level::O0) {
      lane.rungs[next++] =
          rung(c, lane.ready->compile_graph, top, jit::Level::O0);
    }
    lane.rungs[next] =
        rung(c, lane.ready->compile_graph, top, top.codegen.codegen_level);
  }

  // How wide to emit, from the batch the caller stated: a wide kernel computes
  // `w` points to answer for one, so a short batch emits scalar -- the same
  // threshold the sweep uses.  A stated `lanes` is honoured.
  [[nodiscard]] jit::Options effective_options() const noexcept {
    return jit::for_batch(options_, rt::block_lanes);
  }

  // A Kernel does not own its code, so the compiler outlives every kernel.
  // Null on a host with no JIT.
  static jit::Compiler *compiler() { return rt_detail::shared_compiler(); }
#endif

  // Decided in the *graph*, so sweep and kernel fold the same products.  A
  // JIT-less build contracts too, so an answer does not depend on how the
  // library was configured.
  [[nodiscard]] bool contracts() const noexcept {
#ifdef DDX_HAS_JIT
    return options_.codegen.contract;
#else
    return true;
#endif
  }

  void interpret(const Compiled &c, std::span<const T *const> xs,
                 const rt::Columns<T> &out, std::size_t n) const {
    const auto blocks = c.graph->output_blocks();
    const auto schedule = c.graph->schedule();
    // The graph's width, not the arena's symbol count: `xs` carries one column
    // per input, and a lane may hold leaves that are not symbols.
    const std::size_t symbols = c.graph->arity();

    // Tapes are sized by the arena rather than the graph: a borrowed builder
    // may have grown since the freeze, and live ids index below that.
    //
    // A short batch sweeps one point at a time rather than padding out to
    // rt::block_lanes, which on a batch of one is the whole cost.
    if (n < rt::block_lanes) {
      // Grown once and kept: a point at a time is the shape a minimiser asks
      // in, and two allocations per gradient were most of a small one.
      static thread_local std::vector<T> at;
      static thread_local std::vector<T> tape;
      at.resize(symbols);
      tape.resize(arena_->size());
      for (const std::size_t i : std::views::iota(0uz, n)) {
        std::ranges::transform(xs, at.begin(),
                               [i](const T *column) { return column[i]; });
        rt::evaluate_into(*arena_, at, schedule, std::span<T>{tape});
        rt::scatter_blocks(blocks, std::span<const T>{tape}, out, i, 1, 1);
      }
      return;
    }

    // rt::block_lanes points per sweep: the switch is paid once per node per
    // block, and each operation becomes a lane loop wide enough to vectorise.
    // A short final block repeats its last point, and those lanes are never
    // read back.
    std::vector<T> lanes(symbols * rt::block_lanes);
    std::vector<T> tape(arena_->size() * rt::block_lanes);

    for (const std::size_t base :
         std::views::iota(0uz, n) | std::views::stride(rt::block_lanes)) {
      const std::size_t width = std::min(rt::block_lanes, n - base);
      for (const auto [dst, column] :
           std::views::zip(lanes | std::views::chunk(rt::block_lanes), xs)) {
        const auto tail =
            std::ranges::copy(std::span{column + base, width}, dst.begin()).out;
        std::ranges::fill(tail, dst.end(), column[base + width - 1]);
      }
      rt::evaluate_block<rt::block_lanes>(*arena_, std::span<const T>{lanes},
                                          schedule, std::span<T>{tape});
      rt::scatter_blocks(blocks, std::span<const T>{tape}, out, base,
                         rt::block_lanes, width);
    }
  }

  static auto as_columns(const std::ranges::contiguous_range auto &r) {
    using Ptr = std::ranges::range_value_t<std::remove_cvref_t<decltype(r)>>;
    return std::span<Ptr const>{std::ranges::data(r), std::ranges::size(r)};
  }

  [[nodiscard]] result<void> dispatch(Want want, const Compiled &c,
                                      std::span<const T *const> xs,
                                      const rt::Columns<T> &out,
                                      std::size_t n) const {
    if (bad_) {
      return std::unexpected{*bad_};
    }
    if (xs.size() != c.graph->arity() ||
        std::ranges::any_of(rt::zip_blocks(out, c.graph->output_blocks()),
                            [](const auto &pair) {
                              const auto &[columns, block] = pair;
                              return columns.size() != block.size();
                            })) {
      return fail(errc::wrong_column_count);
    }
    // A point already answered is answered again from what it left behind; with
    // no cache this is the call below and nothing else.
    this->through(want, *c.graph, answered(c), xs, out, n,
                  [&] { run(c, xs, out, n); });
    return {};
  }

  // Whether this lane's answer comes from a kernel, which computes its whole
  // graph, or from a sweep, which leaves a tape an amendment can work from.
  [[nodiscard]] static rt::Answered
  answered([[maybe_unused]] const Compiled &c) noexcept {
#ifdef DDX_HAS_JIT
    if constexpr (std::same_as<T, double>) {
      if (c.kernel) {
        return rt::Answered::ByKernel;
      }
    }
#endif
    return rt::Answered::BySweep;
  }

  void run(const Compiled &c, std::span<const T *const> xs,
           const rt::Columns<T> &out, std::size_t n) const {
#ifdef DDX_HAS_JIT
    if constexpr (std::same_as<T, double>) {
      if (c.kernel) {
        c.kernel(xs, out.values, out.jacobian, out.hessian, n);
        return;
      }
    }
#endif
    interpret(c, xs, out, n);
  }

  // What the equation holds the ids of, and what it gives up around a sweep:
  // nothing here, and the GIL on the Python side.
  [[nodiscard]] const rt::Builder<T> &arena() const noexcept { return *arena_; }
  struct Nothing {};
  [[nodiscard]] static constexpr Nothing unlocked() noexcept { return {}; }

  // The deleter is the only difference between a borrowed arena and an owned
  // one.
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
  // Empty for a system: only root 0's second-order block is ever a lane, as
  // hessian() has it.
  rt::HessianVector hvp_;
  rt::VectorJacobian vjp_;
  rt::Tangent jvp_;
#ifdef DDX_HAS_JIT
  std::vector<rt::NodeId> compile_roots_;
  rt::Jacobian compile_derivative_;
#endif
  // unique_ptr, not shared: its destructor is constexpr, which is what keeps a
  // constant-evaluated Equation destructible.  Null only there.
  std::unique_ptr<Lanes> lanes_;
  // Set exactly when why_not() refused, so poisoned() <=> arena_ == nullptr.
  std::optional<error> bad_{};
  bool loaded_ = false;
};

} // namespace ddx::impl

namespace ddx::rt {

#ifdef DDX_HAS_JIT
using jit::Backend;
using jit::Lanes;
using jit::Level;
#endif

// Over expressions already built in a caller's own arena.  Partial
// specialisations contribute no deduction guides, so this is the whole of CTAD.
//
// A cache leads here and trails everywhere else: nothing can be deduced behind
// a pack, and the functions are the pack.
template <impl::Numeric T, typename... Ts>
  requires(std::same_as<Ts, RTExpression<T>> && ...)
[[nodiscard]] constexpr auto equation(RTExpression<T> first, Ts... rest) {
  return equation(detail::NoCache<T>{}, first, rest...);
}

template <typename C, impl::Numeric T, typename... Ts>
  requires(CValueCache<C, T> && (std::same_as<Ts, RTExpression<T>> && ...))
[[nodiscard]] constexpr auto equation(C memo, RTExpression<T> first,
                                      Ts... rest) {
  return impl::Equation<RTExpression<T>, Ts..., C>::create(std::move(memo),
                                                           first, rest...);
}

// Build an Equation:
//
//   const auto eq = ddx::rt::equation([] {
//     const auto x = var("x");
//     const auto y = var("y");
//     return exp(x) * sin(y);
//   });

namespace detail {

// Run `assemble` inside a fresh arena, and hand both back.
template <impl::Numeric T>
[[nodiscard]] auto assemble(std::invocable auto &&assemble) {
  auto arena = std::make_unique<Builder<T>>();
  auto built = [&] {
    const auto scope = scoped_arena(*arena);
    return std::invoke(DDX_FWD(assemble));
  }();
  return std::pair{std::move(arena), std::move(built)};
}

// `make(type_identity<Eq>, roots...)` over the Equation type the build's shape
// names: one root, or a tuple-like of them -- output_dim lives in the type, so
// the size must too.
template <impl::Numeric T, typename Cache, typename Built>
[[nodiscard]] auto over(const Built &built, auto &&make) {
  if constexpr (std::same_as<Built, RTExpression<T>>) {
    return make(std::type_identity<impl::Equation<RTExpression<T>, Cache>>{},
                built);
  } else {
    constexpr std::size_t outputs = std::tuple_size_v<Built>;
    static_assert(outputs > 0,
                  "equation: a system needs at least one function");
    return impl::index_apply<outputs - 1>([&]<std::size_t... Rest>() {
      return make(
          std::type_identity<
              impl::Equation<RTExpression<T>, Repeat<Rest, T>..., Cache>>{},
          built[0], built[Rest + 1]...);
    });
  }
}

} // namespace detail

template <impl::Numeric T = double, CValueCache<T> C = detail::NoCache<T>>
[[nodiscard]] auto equation(std::invocable auto &&assemble, C memo = {}) {
  auto [arena, built] = detail::assemble<T>(DDX_FWD(assemble));
  return detail::over<T, C>(
      built, [&]<typename Eq>(std::type_identity<Eq>, auto... roots) {
        return Eq::create(std::move(memo), std::move(arena), roots...);
      });
}

// An equation from a file and nothing else: symbols, arena, Jacobian and
// colouring all come off disk, and no sweep runs.  The output count is part of
// the type, so a file holding some other number is refused, not adapted to.
//
//   const auto eq = ddx::rt::load("f.ddx");        // one function
//   const auto eq = ddx::rt::load<double, 3>("f.ddx");
template <impl::Numeric T = double, std::size_t Outputs = 1,
          CValueCache<T> C = detail::NoCache<T>>
  requires std::floating_point<T>
[[nodiscard]] auto load(const std::filesystem::path &path, C memo = {}) {
  static_assert(Outputs > 0, "load: a system needs at least one function");
  return impl::index_apply<Outputs - 1>([&]<std::size_t... Rest>() {
    return impl::Equation<RTExpression<T>, detail::Repeat<Rest, T>...,
                          C>::load(path, std::move(memo));
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
template <impl::Numeric T = double, CValueCache<T> C = detail::NoCache<T>>
  requires std::floating_point<T>
[[nodiscard]] auto equation(const std::filesystem::path &path,
                            std::invocable auto &&assemble, C memo = {}) {
  auto [arena, built] = detail::assemble<T>(DDX_FWD(assemble));
  return detail::over<T, C>(
      built, [&]<typename Eq>(std::type_identity<Eq>, auto... roots) {
        return Eq::cached(std::move(memo), path, std::move(arena), roots...);
      });
}

} // namespace ddx::rt
