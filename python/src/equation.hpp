#pragma once

#include "columns.hpp"
#include "error.hpp"
#include "expression.hpp"

#include "rt/archive.hpp"
#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/dot.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "util/ranges.hpp"

// Unconditional: jit::Options and jit::Kernel are header types, so the Python
// surface has one shape whether or not the backend was compiled in.
#include "jit/kernel.hpp"
#include "rt/equation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <future>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

// What ddx::impl::Equation is, with the output count a runtime value.  Much
// smaller than the facade because the GIL is the lock: every freeze, poll and
// adopt below runs holding it, and only the kernel or the sweep gives it up.
namespace ddx::py {

class PyEquation {
public:
  // Ordered
  enum class Want : std::uint8_t { Values, Jacobian, Hessian };

  PyEquation(std::shared_ptr<rt::Builder<double>> arena,
             std::vector<rt::NodeId> roots)
      : arena_(std::move(arena)), roots_(std::move(roots)),
        model_nodes_(static_cast<std::uint32_t>(arena_->size())) {}

  // From a file: the sweeps arrive filled, so nothing below computes them.
  explicit PyEquation(rt::Snapshot<double> &&snap)
      : arena_(rt::rebuild(snap)), roots_(std::move(snap.roots)),
        derivative_(std::move(snap.jacobian)),
        sweeps_(std::move(snap.hessians)), options_(snap.options),
        objects_(std::move(snap.objects)), model_nodes_(snap.model_nodes),
        loaded_(true) {}

  // The same struct the C++ facade fills: one serialiser, two equations.
  [[nodiscard]] rt::Snapshot<double> snapshot() {
    rt::Snapshot<double> snap;
    // The sweeps first, the arena after: they *append*, so a node array taken
    // before they run lacks the ids they hand back.  The order matters here and
    // not in the facade because this side sweeps lazily.
    snap.jacobian = derivative();
    snap.hessians = sweeps();
    snap.symbols = arena_->symbols();
    snap.nodes.assign(arena_->nodes().begin(), arena_->nodes().end());
    snap.roots = roots_;
    snap.options = options_;
    snap.model_nodes = model_nodes_;
    snap.objects = objects();
    return snap;
  }

  [[nodiscard]] constexpr bool loaded() const noexcept { return loaded_; }

  [[nodiscard]] std::size_t arity() const { return arena_->symbols().size(); }
  [[nodiscard]] constexpr std::size_t outputs() const { return roots_.size(); }
  [[nodiscard]] const std::vector<std::string> &symbols() const {
    return arena_->symbols();
  }

  // --- the three lanes, as Python sees them --------------------------------

  [[nodiscard]] pyb::object evaluate(const pyb::handle &x) {
    const Point at{x, symbols()};
    Lane &l = lane(Want::Values);
    Block f{count(l, &rt::Graph<double>::Blocks::values), at.size()};
    run(l, at, f.rows(), {}, {});
    return finish(std::move(f), {ssize(outputs())}, at);
  }

  [[nodiscard]] pyb::tuple jacobian(const pyb::handle &x) {
    const Point at{x, symbols()};
    Lane &l = lane(Want::Jacobian);
    Block f{count(l, &rt::Graph<double>::Blocks::values), at.size()};
    Block g{count(l, &rt::Graph<double>::Blocks::jacobian), at.size()};
    run(l, at, f.rows(), g.rows(), {});
    return pyb::make_tuple(
        finish(std::move(f), {ssize(outputs())}, at),
        finish(std::move(g), {ssize(outputs()), ssize(arity())}, at));
  }

  [[nodiscard]] pyb::tuple hessian(const pyb::handle &x) {
    const Point at{x, symbols()};
    return outputs() == 1 ? hessian_from_lane(at) : hessian_from_arena(at);
  }

  // The machine code the lanes are holding, for the file to carry.  Only
  // kernels that kept their bytes -- Options.retain_object, on by default.
  [[nodiscard]] std::vector<rt::Object> objects() {
    jit::Compiler *const c = compiler();
    if (c == nullptr) {
      return {};
    }
    std::vector<rt::Object> out;
    for (const auto [i, l] : lanes_ | std::views::enumerate) {
      if (!l.graph || !l.kernel || l.kernel.object().empty()) {
        continue;
      }
      const auto code = l.kernel.object();
      out.push_back({.want = static_cast<std::uint8_t>(i),
                     .symbol = std::string{l.kernel.symbol()},
                     .host = std::string{c->host_identity()},
                     .digest = rt::digest(*l.graph),
                     .options = effective_options(),
                     .code = {code.begin(), code.end()}});
    }
    return out;
  }

  // --- the JIT -------------------------------------------------------------

  [[nodiscard]] constexpr const jit::Options &options() const noexcept {
    return options_;
  }

  // Discards whatever was compiled, so the choice holds however late.  Choosing
  // a compiling backend starts the compile, so it overlaps what comes next.
  void set_options(const jit::Options &opt) {
    if (opt == options_) {
      return;
    }
    options_ = opt;
    lanes_ = {};
    if (opt.backend != jit::Backend::Interpret) {
      (void)lane(Want::Jacobian);
    }
  }

  // The only thing that waits.  A call does not: until the kernel lands the
  // graph is swept, and the kernel replaces the sweep when it arrives.
  [[nodiscard]] bool wait_for_kernel() {
    Lane &l = lane(Want::Jacobian);
    if (l.pending.valid()) {
      // Given up around the wait alone, never around the lane bookkeeping.
      const pyb::gil_scoped_release unlocked;
      l.pending.wait();
    }
    adopt(l);
    return static_cast<bool>(l.kernel);
  }

  [[nodiscard]] bool uses_kernel() {
    return static_cast<bool>(lane(Want::Jacobian).kernel);
  }

  // Colours, not n: the Hessian is stored compressed and scattered on read, so
  // this is what the compiled lane actually computes per point.
  [[nodiscard]] std::size_t hessian_colors() {
    return sweeps().front().colors();
  }

  // Only what is free to answer: uses_kernel() would freeze a lane, and a repr
  // that compiles something is one nobody can call in a debugger.
  [[nodiscard]] std::string repr() const {
    return std::format("<ddx.Equation ({}) -> {} output{}>",
                       symbols() | std::views::join_with(std::string{", "}) |
                           impl::to<std::string>(),
                       outputs(), outputs() == 1 ? "" : "s");
  }

  // --- the equation on disk ------------------------------------------------

  void save(const std::filesystem::path &path) {
    unwrap(rt::save(snapshot(), path));
  }

  // Raises rather than answering false: "no" has three reasons -- unreadable,
  // unloadable, a different equation -- and only the errc says which.
  void verify(const std::filesystem::path &path) {
    auto snap = rt::load_snapshot<double>(path);
    if (!snap) {
      throw PyError{snap.error()};
    }
    if (!describes(*snap)) {
      fail_with(errc::archive_mismatch);
    }
  }

  // The roots as well as the model: two equations over one arena share every
  // node and differ only in which ids they call outputs.
  [[nodiscard]] bool describes(const rt::Snapshot<double> &snap) const {
    return snap.roots.size() == roots_.size() &&
           std::ranges::equal(snap.roots, roots_) &&
           rt::digest<double>(snap.symbols, snap.nodes, snap.model_nodes) ==
               rt::digest<double>(arena_->symbols(), arena_->nodes(),
                                  model_nodes_);
  }

  [[nodiscard]] std::string to_dot(bool all) {
    return rt::Dot<double>{*lane(Want::Jacobian).graph,
                           all ? rt::Scope::All : rt::Scope::Live}
        .str();
  }

private:
  // Filled once and never invalidated; a new backend throws the whole set away
  // rather than editing one.
  struct Lane {
    std::shared_ptr<const rt::Graph<double>> graph;
    jit::Kernel kernel{};
    std::shared_future<jit::result<jit::Kernel>> pending;
    // Adapt's counter.  Plain, not atomic: the GIL is held for every call that
    // touches a lane.  `settled` is set once nothing is left to buy, so a lane
    // that will never compile again stops counting.
    std::size_t points = 0;
    bool settled = false;
  };

  // Points per block sweep, as in the facade: two AVX2 registers of doubles.
  static constexpr std::size_t kLanes = 8;

  [[nodiscard]] Lane &lane(Want want) {
    Lane &l = lanes_[static_cast<std::size_t>(want)];
    if (!l.graph) {
      prepare(l, want);
    }
    adopt(l);
    return l;
  }

  // Freezing a lane and launching its compile are one step deliberately: split,
  // a lane could be ready having compiled nothing, and the next caller would
  // find it ready and interpret forever with every answer still right.
  void prepare(Lane &l, Want want) {
    rt::GraphBuilder<double> gb{*arena_};
    gb.values_from(roots_);
    if (want != Want::Values) {
      gb.jacobian_from(derivative().partial);
    }
    if (want == Want::Hessian) {
      gb.hessian_from(sweeps().front());
    }
    l.graph = std::make_shared<const rt::Graph<double>>(gb.finish(contracts()));
    if (options_.backend == jit::Backend::Interpret) {
      l.settled = true;
      return;
    }
    jit::Compiler *const c = compiler();
    if (c == nullptr) {
      l.settled = true;
      return;
    }
    // A compile already done, adopted only where graph, host and codegen
    // options all still agree: adopt() cannot see that an object came from
    // another graph, and running one that did is silently wrong arithmetic.
    const auto lane_id = static_cast<std::uint8_t>(want);
    const auto digest = rt::digest(*l.graph);
    const auto stored =
        std::ranges::find_if(objects_, [&](const rt::Object &o) {
          return o.want == lane_id && o.digest == digest &&
                 o.host == c->host_identity() &&
                 rt::same_codegen(o.options, effective_options()) &&
                 !o.code.empty();
        });
    if (stored != objects_.end()) {
      // Shapes from the graph just frozen, never from the file: a forged entry
      // may supply code and a symbol but cannot claim an arity.
      const auto &layout = l.graph->layout();
      if (auto adopted =
              c->adopt(stored->code, stored->symbol, l.graph->symbols().size(),
                       layout.values, layout.jacobian, layout.hessian)) {
        l.kernel = std::move(*adopted);
        l.settled = true;
        return; // nothing left to compile
      }
      // A refused link is a miss, never a failure: the compile below stands.
    }
    // Adapt earns its compile in charge(); Compile asks for it outright.  There
    // is one rung here rather than the facade's ladder, so warm_points is the
    // only threshold this lane has and hot_points says nothing.
    if (options_.backend == jit::Backend::Compile) {
      l.pending = c->compile_async(l.graph, effective_options());
      l.settled = true;
    }
  }

  // Charge the lane for a batch and launch the compile the points have bought.
  void charge(Lane &l, std::size_t n) {
    if (l.settled || n == 0 || options_.backend != jit::Backend::Adapt) {
      return;
    }
    l.points += n;
    if (l.points < options_.warm_points) {
      return;
    }
    if (jit::Compiler *const c = compiler(); c != nullptr) {
      l.pending = c->compile_async(l.graph, effective_options());
    }
    l.settled = true;
  }

  // A refused compile leaves the kernel empty and the sweep stays: the JIT is
  // never a correctness dependency.
  void adopt(Lane &l) {
    using namespace std::chrono_literals;
    if (!l.pending.valid() ||
        l.pending.wait_for(0s) != std::future_status::ready) {
      return;
    }
    if (const auto &landed = l.pending.get()) {
      l.kernel = *landed;
    }
    l.pending = {};
  }

  // How wide to emit, from the batch the caller said they have: a wide kernel
  // does a register's worth of work to answer for a single point.
  [[nodiscard]] constexpr jit::Options effective_options() const noexcept {
    jit::Options opt = options_;
    if (opt.lanes == 0 && opt.points < kLanes) {
      opt.lanes = 1;
    }
    return opt;
  }

  // The process's one LLJIT, borrowed rather than founded: making its own would
  // stand a second one up beside whatever C++ Equations already use.  Null in a
  // build without the backend, where the sweep answers everything -- the path a
  // Compile backend takes anyway while its compile is in flight.
  [[nodiscard]] static jit::Compiler *compiler() {
#ifdef DDX_HAS_JIT
    return impl::rt_detail::shared_compiler();
#else
    return nullptr;
#endif
  }

  // Decided in the graph, so a batch answers the same whichever path ran it.
  [[nodiscard]] constexpr bool contracts() const noexcept {
    return options_.contract;
  }

  using BlockMember = std::span<const rt::NodeId> rt::Graph<double>::Blocks::*;

  [[nodiscard]] static constexpr std::size_t count(const Lane &l,
                                                   BlockMember which) {
    return (l.graph->output_blocks().*which).size();
  }

  // --- running -------------------------------------------------------------

  void run(Lane &l, const Point &at, std::span<double *const> f,
           std::span<double *const> g, std::span<double *const> h) {
    const auto xs = at.columns();
    const auto blocks = l.graph->output_blocks();
    if (xs.size() != arity() || f.size() != blocks.values.size() ||
        g.size() != blocks.jacobian.size() ||
        h.size() != blocks.hessian.size()) {
      fail_with(errc::wrong_column_count);
    }
    // Before the GIL goes, and after the shape check: a call that will not run
    // has bought nothing.
    charge(l, at.size());
    // Past here nothing touches Python, so the GIL goes -- which is what lets
    // another thread call in while a long batch runs.
    const pyb::gil_scoped_release unlocked;
    if (l.kernel) {
      l.kernel(xs, f, g, h, at.size());
      return;
    }
    interpret(*l.graph, xs, f, g, h, at.size());
  }

  // The block sweep, within ~1.4x of a kernel and not a sad path.
  void interpret(const rt::Graph<double> &graph,
                 std::span<const double *const> xs, std::span<double *const> f,
                 std::span<double *const> g, std::span<double *const> h,
                 std::size_t n) const {
    const auto blocks = graph.output_blocks();
    // The freeze settled both; there is no choice left to make here.
    const auto order = graph.contracted_order();
    const auto contractions = graph.contractions();
    const std::size_t symbols = arity();
    const auto scatter = [&](const auto &tape, std::size_t i,
                             std::size_t stride, std::size_t width) {
      for (const auto [columns, block] : std::views::zip(
               std::array{f, g, h},
               std::array{blocks.values, blocks.jacobian, blocks.hessian})) {
        for (const auto [column, o] : std::views::zip(columns, block)) {
          std::ranges::copy_n(tape.data() + std::size_t{o} * stride, width,
                              column + i);
        }
      }
    };

    // Tapes are sized by the arena rather than the graph: it may have grown
    // since the freeze, and live ids index below that.  A short batch sweeps one
    // point at a time rather than padding out to kLanes.
    if (n < kLanes) {
      std::vector<double> at(symbols);
      std::vector<double> tape(arena_->size());
      for (const std::size_t i : std::views::iota(0uz, n)) {
        std::ranges::transform(xs, at.begin(),
                               [i](const double *column) { return column[i]; });
        rt::evaluate_into(*arena_, at, order, contractions,
                          std::span<double>{tape});
        scatter(tape, i, 1, 1);
      }
      return;
    }

    // kLanes points per sweep: the switch is paid once per node per block, and
    // each operation becomes a lane loop wide enough to vectorise.  A short
    // final block repeats its last point, and those lanes are never read.
    std::vector<double> lanes(symbols * kLanes);
    std::vector<double> tape(arena_->size() * kLanes);

    for (const std::size_t base :
         std::views::iota(0uz, n) | std::views::stride(kLanes)) {
      const std::size_t width = std::min(kLanes, n - base);
      for (const auto [dst, column] :
           std::views::zip(lanes | std::views::chunk(kLanes), xs)) {
        const auto tail =
            std::ranges::copy_n(column + base, width, dst.begin());
        std::ranges::fill(tail.out, dst.end(), column[base + width - 1]);
      }
      rt::evaluate_block<kLanes>(*arena_, std::span<const double>{lanes}, order,
                                 contractions, std::span<double>{tape});
      scatter(tape, base, kLanes, width);
    }
  }

  // --- the Hessian ---------------------------------------------------------

  // One function: the lane holds it compressed by colour, and the colouring says
  // which (colour, row) column owns a given (i, j).
  [[nodiscard]] pyb::tuple hessian_from_lane(const Point &at) {
    Lane &l = lane(Want::Hessian);
    const std::size_t n = arity();
    Block f{count(l, &rt::Graph<double>::Blocks::values), at.size()};
    Block g{count(l, &rt::Graph<double>::Blocks::jacobian), at.size()};
    Block compressed{count(l, &rt::Graph<double>::Blocks::hessian), at.size()};
    run(l, at, f.rows(), g.rows(), compressed.rows());

    const rt::Coloring &c = l.graph->coloring();
    const auto column = rt::by_color(compressed.rows(), c.count, n);
    Block dense{n * n, at.size()};
    const auto dims = std::views::iota(0uz, n);
    for (const auto [i, j] : std::views::cartesian_product(dims, dims)) {
      double *const out = dense.at(i * n + j);
      if (c.target(c.color[j], i) == j) {
        std::ranges::copy_n(column[c.color[j], i], at.size(), out);
      } else {
        std::ranges::fill_n(out, static_cast<std::ptrdiff_t>(at.size()), 0.0);
      }
    }
    return pyb::make_tuple(
        finish(std::move(f), {ssize(outputs())}, at),
        finish(std::move(g), {ssize(outputs()), ssize(n)}, at),
        finish(std::move(dense), {ssize(outputs()), ssize(n), ssize(n)}, at));
  }

  // A system: one sweep per root, read off the arena.  A frozen graph carries a
  // single colouring, so no lane could hold m of these -- which is why this is a
  // point at a time and says so.
  [[nodiscard]] pyb::tuple hessian_from_arena(const Point &at) {
    if (at.batched() && at.size() != 1) {
      fail_with(errc::wrong_column_count);
    }
    const std::size_t n = arity();
    const auto &blocks = sweeps();

    std::vector<double> point(n);
    std::ranges::transform(at.columns(), point.begin(),
                           [](const double *c) { return *c; });
    // Values, partials and Hessian::at's compressed cells: what the three blocks
    // below read, and nothing of the arena beyond it.
    auto wanted = blocks | std::views::transform(&rt::Hessian::compressed) |
                  std::views::join | impl::to<std::vector<rt::NodeId>>();
    impl::append(wanted, blocks |
                             std::views::transform(&rt::Hessian::partial) |
                             std::views::join);
    impl::append(wanted, blocks | std::views::transform(&rt::Hessian::zero));
    impl::append(wanted, roots_);
    const auto values = rt::evaluate_reachable(*arena_, wanted, point);

    Block f{outputs(), at.size()};
    Block g{outputs() * n, at.size()};
    Block dense{outputs() * n * n, at.size()};
    const auto dims = std::views::iota(0uz, n);
    for (const auto [k, block] : std::views::enumerate(blocks)) {
      const auto root = static_cast<std::size_t>(k);
      *f.at(root) = values[roots_[root]];
      for (const std::size_t i : dims) {
        *g.at(root * n + i) = values[block.partial[i]];
      }
      for (const auto [i, j] : std::views::cartesian_product(dims, dims)) {
        *dense.at((root * n + i) * n + j) = values[block.at(i, j)];
      }
    }
    return pyb::make_tuple(
        finish(std::move(f), {ssize(outputs())}, at),
        finish(std::move(g), {ssize(outputs()), ssize(n)}, at),
        finish(std::move(dense), {ssize(outputs()), ssize(n), ssize(n)}, at));
  }

  // Swept once and kept.  Hash consing makes a repeat sweep add no nodes, so
  // this is about the colouring's cost, not the arena's size.
  [[nodiscard]] const std::vector<rt::Hessian> &sweeps() {
    if (!sweeps_) {
      sweeps_ = roots_ | std::views::transform([this](rt::NodeId r) {
                  return rt::build_hessian_impl(*arena_, r);
                }) |
                impl::to<std::vector<rt::Hessian>>();
    }
    return *sweeps_;
  }

  // Held for the same reason the Hessians are: swept per lane otherwise, which
  // leaves a loaded equation recomputing what its file just handed it.
  [[nodiscard]] const rt::Jacobian &derivative() {
    if (!derivative_) {
      derivative_ = rt::build_jacobian_impl(*arena_, roots_);
    }
    return *derivative_;
  }

  // --- shapes --------------------------------------------------------------

  // The leading axis goes at one function and the trailing one at one point, so
  // a scalar model at a point answers with a float, an (n,) gradient and an
  // (n, n) Hessian rather than three arrays with 1s in them.
  [[nodiscard]] pyb::object finish(Block &&b, std::vector<pyb::ssize_t> dims,
                                   const Point &at) const {
    if (outputs() == 1 && !dims.empty()) {
      dims.erase(dims.begin());
    }
    if (at.batched()) {
      dims.push_back(ssize(at.size()));
    }
    if (dims.empty()) {
      return pyb::float_(*b.at(0));
    }
    return std::move(b).reshaped(std::move(dims));
  }

  std::shared_ptr<rt::Builder<double>> arena_;
  std::vector<rt::NodeId> roots_;
  std::optional<rt::Jacobian> derivative_;
  std::optional<std::vector<rt::Hessian>> sweeps_;
  jit::Options options_{};
  // Machine code a file handed over, consulted once per lane by prepare().
  std::vector<rt::Object> objects_{};
  // Where the model ends and the sweeps begin; set by make_equation once the
  // roots are in.
  std::uint32_t model_nodes_ = 0;
  bool loaded_ = false;
  std::array<Lane, 3> lanes_;
};

} // namespace ddx::py
