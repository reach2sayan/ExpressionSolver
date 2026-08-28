#pragma once

#include "columns.hpp"
#include "error.hpp"
#include "expression.hpp"

#include "rt/archive/archive.hpp"
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

class PyCall;

class PyEquation {
public:
  // Ordered
  enum class Want : std::uint8_t { Values, Jacobian, Hessian, Hvp, Vjp,
                                   Jvp };

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

  // The same struct the C++ facade fills: one serialiser, two equations.  Also
  // what the cache path in module.cpp writes, which is why it is not private.
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
  // The symbols as interned Python strings, built once.  Lazy, and safe: every
  // caller holds the GIL here -- run() does not drop it until after the Point
  // is built.
  [[nodiscard]] std::span<const pyb::object> names() {
    if (names_.size() != symbols().size()) {
      names_.clear();
      names_.reserve(symbols().size());
      for (const std::string &s : symbols()) {
        names_.emplace_back(pyb::str(s));
      }
    }
    return names_;
  }

  [[nodiscard]] const std::vector<std::string> &symbols() const {
    return arena_->symbols();
  }

  // --- the three lanes, as Python sees them --------------------------------

  [[nodiscard]] pyb::object evaluate(const pyb::handle &x) {
    const Point at{x, symbols(), names()};
    Lane &l = lane(Want::Values);
    if (scalar(at)) {
      Cell f;
      run(l, at, f.rows(), {}, {});
      return pyb::float_(f.value);
    }
    Block f{shape_of({ssize(outputs())}, at),
            count(l, &rt::Graph<double>::Blocks::values), at.size()};
    run(l, at, f.rows(), {}, {});
    return std::move(f).array();
  }

  [[nodiscard]] pyb::tuple jacobian(const pyb::handle &x) {
    const Point at{x, symbols(), names()};
    Lane &l = lane(Want::Jacobian);
    Scratch cells{count(l, &rt::Graph<double>::Blocks::jacobian), at.size()};
    Block g{shape_of({ssize(outputs()), ssize(arity())}, at),
            outputs() * arity(), at.size()};
    Cell cell;
    auto f = values_block(l, at);
    run(l, at, f ? f->rows() : cell.rows(), cells.rows(), {});
    scatter_jacobian(cells, g, at.size());
    return pyb::make_tuple(value_of(f, cell), std::move(g).array());
  }

  [[nodiscard]] pyb::tuple hessian(const pyb::handle &x) {
    const Point at{x, symbols(), names()};
    return outputs() == 1 ? hessian_from_lane(at) : hessian_from_arena(at);
  }

  // --- the seeded products -------------------------------------------------
  //
  // J(x)v, w'J(x) and H(x)v, none of which forms the matrix it is named after.
  // The direction rides in its own argument, so it can never be read as part
  // of the point.

  [[nodiscard]] pyb::tuple jvp(const pyb::handle &v, const pyb::handle &x) {
    const Point at{x, symbols(), names()};
    const Point dir{v, symbols(), names()};
    Lane &l = lane(Want::Jvp);
    Block out{shape_of({ssize(outputs())}, at),
              count(l, &rt::Graph<double>::Blocks::jacobian), at.size()};
    Cell cell;
    auto f = values_block(l, at);
    run_seeded(l, at, dir, f ? f->rows() : cell.rows(), out.rows(), {});
    return pyb::make_tuple(value_of(f, cell), std::move(out).array());
  }

  [[nodiscard]] pyb::tuple vjp(const pyb::handle &w, const pyb::handle &x) {
    const Point at{x, symbols(), names()};
    // By position: the covector is indexed by function, and a dict would have
    // nothing to key on.
    const Point dir{w, outputs()};
    Lane &l = lane(Want::Vjp);
    Block out{symbol_shape(at),
              count(l, &rt::Graph<double>::Blocks::jacobian), at.size()};
    Cell cell;
    auto f = values_block(l, at);
    run_seeded(l, at, dir, f ? f->rows() : cell.rows(), out.rows(), {});
    return pyb::make_tuple(value_of(f, cell), std::move(out).array());
  }

  // (value, gradient, H v): the gradient comes off the same sweep, so it is
  // handed back rather than thrown away.  One output only, as hessian() is.
  [[nodiscard]] pyb::tuple hvp(const pyb::handle &v, const pyb::handle &x) {
    if (outputs() != 1) {
      fail_with(errc::not_univariate);
    }
    const Point at{x, symbols(), names()};
    const Point dir{v, symbols(), names()};
    Lane &l = lane(Want::Hvp);
    const std::size_t n = arity();

    Scratch partials{count(l, &rt::Graph<double>::Blocks::jacobian), at.size()};
    Block g{shape_of({ssize(outputs()), ssize(n)}, at), outputs() * n,
            at.size()};
    Block out{symbol_shape(at),
              count(l, &rt::Graph<double>::Blocks::hessian), at.size()};
    Cell cell;
    auto f = values_block(l, at);
    run_seeded(l, at, dir, f ? f->rows() : cell.rows(), partials.rows(),
               out.rows());
    scatter_jacobian(partials, g, at.size());
    return pyb::make_tuple(value_of(f, cell), std::move(g).array(),
                           std::move(out).array());
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
  friend class PyCall;

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
    // The seeded lanes replace the Jacobian block rather than adding to it,
    // except Hvp, whose gradient falls out of the same sweep.
    if (want == Want::Vjp) {
      gb.vector_jacobian_from(vjp_sweep());
    } else if (want == Want::Jvp) {
      gb.tangent_from(jvp_sweep());
    } else {
      if (want != Want::Values) {
        gb.jacobian_from(derivative());
      }
      if (want == Want::Hessian) {
        gb.hessian_from(sweeps().front());
      }
      if (want == Want::Hvp) {
        gb.hessian_vector_from(hvp_sweep());
      }
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
                 jit::same_codegen(o.options, effective_options()) &&
                 !o.code.empty();
        });
    if (stored != objects_.end()) {
      // Shapes from the graph just frozen, never from the file: a forged entry
      // may supply code and a symbol but cannot claim an arity.
      const auto &layout = l.graph->layout();
      if (auto adopted =
              c->adopt(stored->code, stored->symbol, l.graph->arity(),
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
    run_columns(l, at.columns(), at.size(), f, g, h);
  }

  // The point's columns and the direction's, in one array: symbols first, then
  // the seeds, which is the order input_column() states.
  void run_seeded(Lane &l, const Point &at, const Point &dir,
                  std::span<double *const> f, std::span<double *const> g,
                  std::span<double *const> h) {
    if (dir.size() != at.size()) {
      fail_with(errc::wrong_direction);
    }
    std::vector<const double *> xs;
    xs.reserve(at.columns().size() + dir.columns().size());
    std::ranges::copy(at.columns(), std::back_inserter(xs));
    std::ranges::copy(dir.columns(), std::back_inserter(xs));
    run_columns(l, xs, at.size(), f, g, h);
  }

  void run_columns(Lane &l, std::span<const double *const> xs, std::size_t n,
                   std::span<double *const> f, std::span<double *const> g,
                   std::span<double *const> h) {
    const auto blocks = l.graph->output_blocks();
    if (xs.size() != l.graph->arity() || f.size() != blocks.values.size() ||
        g.size() != blocks.jacobian.size() ||
        h.size() != blocks.hessian.size()) {
      fail_with(errc::wrong_column_count);
    }
    // Before the GIL goes, and after the shape check: a call that will not run
    // has bought nothing.
    charge(l, n);
    // Past here nothing touches Python, so the GIL goes -- which is what lets
    // another thread call in while a long batch runs.
    const pyb::gil_scoped_release unlocked;
    if (l.kernel) {
      l.kernel(xs, f, g, h, n);
      return;
    }
    interpret(*l.graph, xs, f, g, h, n);
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
    // The graph's width, not the symbol count: `xs` carries one column per
    // input, and a seeded lane holds leaves that are not symbols.
    const std::size_t symbols = graph.arity();
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
    //
    // Scratch, and thread_local rather than a member: run() drops the GIL, so
    // two Python threads can be inside one Equation at once and a shared buffer
    // would be a race.  Grown and never shrunk, so a loop allocates once.
    //
    // resize() rather than assign(): every id in `order` is written by
    // evaluate_block before it is read, and the scatter reads only live output
    // ids, which are in `order` -- so a stale value in an id this graph does not
    // touch is never seen, and zeroing the whole arena per call is waste.
    if (n < kLanes) {
      static thread_local std::vector<double> at;
      static thread_local std::vector<double> tape;
      at.resize(symbols);
      tape.resize(arena_->size());
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
    static thread_local std::vector<double> lanes;
    static thread_local std::vector<double> tape;
    lanes.resize(symbols * kLanes);
    tape.resize(arena_->size() * kLanes);

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

  // This side always hands back the dense matrix, so the zeros go back in
  // here: one fill, then one copy per cell the pattern names.
  void scatter_jacobian(Scratch &cells, Block &dense, std::size_t points) {
    const std::size_t n = arity();
    const std::size_t wide = outputs() * n;
    if (wide == 0) {
      return;
    }
    std::ranges::fill_n(dense.at(0), static_cast<std::ptrdiff_t>(wide * points),
                        0.0);
    const auto compressed = cells.rows();
    const rt::Sparsity &pattern = derivative().pattern;
    for (const std::size_t i : std::views::iota(0uz, outputs())) {
      const auto columns = pattern.row(i);
      const auto cell = compressed.subspan(pattern.rowptr[i], columns.size());
      for (const auto [j, src] : std::views::zip(columns, cell)) {
        std::ranges::copy_n(src, points, dense.at(i * n + j));
      }
    }
  }

  // --- the Hessian ---------------------------------------------------------

  // One function: the lane holds it compressed by colour, and the colouring says
  // which (colour, row) column owns a given (i, j).
  [[nodiscard]] pyb::tuple hessian_from_lane(const Point &at) {
    Lane &l = lane(Want::Hessian);
    const std::size_t n = arity();
    Block g{shape_of({ssize(outputs()), ssize(n)}, at), outputs() * n,
            at.size()};
    Scratch partials{count(l, &rt::Graph<double>::Blocks::jacobian), at.size()};
    Scratch compressed{count(l, &rt::Graph<double>::Blocks::hessian), at.size()};
    Cell cell;
    auto f = values_block(l, at);
    run(l, at, f ? f->rows() : cell.rows(), partials.rows(),
        compressed.rows());
    scatter_jacobian(partials, g, at.size());

    const rt::Coloring &c = l.graph->coloring();
    // Indexed through Coloring::column, never as a dense colours x n grid: the
    // block holds only the cells a column owns, so `cells` is below count * n
    // wherever the pattern is sparse and the dense reading runs off the end.
    const auto cells = compressed.rows();
    Block dense{shape_of({ssize(outputs()), ssize(n), ssize(n)}, at), n * n,
                at.size()};
    const auto dims = std::views::iota(0uz, n);
    for (const auto [i, j] : std::views::cartesian_product(dims, dims)) {
      double *const out = dense.at(i * n + j);
      if (c.target(c.color[j], i) == j) {
        std::ranges::copy_n(cells[c.column(c.color[j], i)], at.size(), out);
      } else {
        std::ranges::fill_n(out, static_cast<std::ptrdiff_t>(at.size()), 0.0);
      }
    }
    return pyb::make_tuple(value_of(f, cell), std::move(g).array(),
                           std::move(dense).array());
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

    Block f{shape_of({ssize(outputs())}, at), outputs(), at.size()};
    Block g{shape_of({ssize(outputs()), ssize(n)}, at), outputs() * n,
            at.size()};
    Block dense{shape_of({ssize(outputs()), ssize(n), ssize(n)}, at),
                outputs() * n * n, at.size()};
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
    return pyb::make_tuple(std::move(f).array(), std::move(g).array(),
                           std::move(dense).array());
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

  // The three seeded products, each swept on first ask.  Hash consing makes a
  // repeat free, so this is about not sweeping for a caller who never asks.
  [[nodiscard]] const rt::HessianVector &hvp_sweep() {
    if (!hvp_) {
      hvp_ = rt::build_hvp_impl(*arena_, roots_.front());
    }
    return *hvp_;
  }
  [[nodiscard]] const rt::VectorJacobian &vjp_sweep() {
    if (!vjp_) {
      vjp_ = rt::build_vjp_impl(*arena_, roots_);
    }
    return *vjp_;
  }
  [[nodiscard]] const rt::Tangent &jvp_sweep() {
    if (!jvp_) {
      jvp_ = rt::build_jvp_impl(*arena_, roots_);
    }
    return *jvp_;
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
  // (n, n) Hessian rather than three arrays with 1s in them.  Computed before
  // the array is built, so the array is built at this shape and never reshaped.
  [[nodiscard]] std::vector<pyb::ssize_t>
  shape_of(std::initializer_list<pyb::ssize_t> base, const Point &at) const {
    const auto *first = base.begin();
    if (outputs() == 1 && first != base.end()) {
      ++first;
    }
    std::vector<pyb::ssize_t> dims(first, base.end());
    if (at.batched()) {
      dims.push_back(ssize(at.size()));
    }
    return dims;
  }

  // A block indexed by symbol rather than by function, so shape_of's leading
  // axis -- which it drops at one output -- is not one this has.
  [[nodiscard]] std::vector<pyb::ssize_t> symbol_shape(const Point &at) const {
    std::vector<pyb::ssize_t> dims{ssize(arity())};
    if (at.batched()) {
      dims.push_back(ssize(at.size()));
    }
    return dims;
  }

  // Exactly the case where the values block is one double and the answer is a
  // Python float, so it needs no array at all.
  [[nodiscard]] bool scalar(const Point &at) const {
    return outputs() == 1 && !at.batched();
  }

  // One double on the stack for that case.  The ABI wants a row pointer, and
  // this is the whole of what it writes.
  struct Cell {
    double value{};
    double *row{&value};
    [[nodiscard]] constexpr std::span<double *const> rows() {
      return {&row, 1};
    }
  };

  // A float at one point, an array across a batch: only the first can skip the
  // allocation, and every derivative call serves both.
  [[nodiscard]] std::optional<Block> values_block(const Lane &l,
                                                  const Point &at) {
    return scalar(at) ? std::nullopt
                      : std::optional<Block>{
                            std::in_place,
                            shape_of({ssize(outputs())}, at),
                            count(l, &rt::Graph<double>::Blocks::values),
                            at.size()};
  }

  [[nodiscard]] static pyb::object value_of(std::optional<Block> &f,
                                            Cell &cell) {
    return f ? std::move(*f).array() : pyb::object{pyb::float_(cell.value)};
  }

  std::shared_ptr<rt::Builder<double>> arena_;
  std::vector<pyb::object> names_;
  std::vector<rt::NodeId> roots_;
  std::optional<rt::Jacobian> derivative_;
  std::optional<std::vector<rt::Hessian>> sweeps_;
  std::optional<rt::HessianVector> hvp_;
  std::optional<rt::VectorJacobian> vjp_;
  std::optional<rt::Tangent> jvp_;
  jit::Options options_{};
  // Machine code a file handed over, consulted once per lane by prepare().
  std::vector<rt::Object> objects_{};
  // Where the model ends and the sweeps begin; set by make_equation once the
  // roots are in.
  std::uint32_t model_nodes_ = 0;
  bool loaded_ = false;
  std::array<Lane, 6> lanes_;
};

// A call bound to its buffers: the point's columns, the output arrays and the
// row pointers are taken once, so a repeat is a lane lookup and the sweep.
// Owning them rather than binding the caller's is what leaves no shape, dtype
// or stride to check.
class PyCall {
public:
  PyCall(pyb::object owner, PyEquation &eq, PyEquation::Want want,
         const pyb::handle &x)
      : owner_(std::move(owner)), eq_(&eq), want_(want),
        x_(pyb::cast<Array>(x)), at_(x_, eq.symbols(), eq.names()) {
    // A frozen graph carries one colouring, so a system's Hessian is read off
    // the arena a point at a time and there is no lane to bind.
    if (want_ == PyEquation::Want::Hessian && eq_->outputs() != 1) {
      fail_with(errc::wrong_column_count);
    }
    const PyEquation::Lane &l = eq_->lane(want_);
    const auto outputs = ssize(eq_->outputs());
    const auto n = eq_->arity();
    f_.emplace(eq_->shape_of({outputs}, at_),
               PyEquation::count(l, &rt::Graph<double>::Blocks::values),
               at_.size());
    if (want_ != PyEquation::Want::Values) {
      partials_.emplace(
          PyEquation::count(l, &rt::Graph<double>::Blocks::jacobian),
          at_.size());
      g_.emplace(eq_->shape_of({outputs, ssize(n)}, at_), eq_->outputs() * n,
                 at_.size());
    }
    if (want_ == PyEquation::Want::Hessian) {
      compressed_.emplace(
          PyEquation::count(l, &rt::Graph<double>::Blocks::hessian),
          at_.size());
      h_.emplace(eq_->shape_of({outputs, ssize(n), ssize(n)}, at_), n * n,
                 at_.size());
    }
  }

  // The lane is looked up per call, never cached: adopt() swaps a kernel in,
  // and a call holding the old one interprets on with every answer still
  // right.
  void operator()() {
    PyEquation::Lane &l = eq_->lane(want_);
    eq_->run(l, at_, f_->rows(),
             partials_ ? partials_->rows() : std::span<double *const>{},
             compressed_ ? compressed_->rows() : std::span<double *const>{});
    if (g_) {
      eq_->scatter_jacobian(*partials_, *g_, at_.size());
    }
    if (h_) {
      scatter(l);
    }
  }

  [[nodiscard]] pyb::object x() const { return x_; }
  [[nodiscard]] pyb::object value() const { return read(f_); }
  [[nodiscard]] pyb::object jacobian() const { return read(g_); }
  [[nodiscard]] pyb::object hessian() const { return read(h_); }

  [[nodiscard]] std::string repr() const {
    static constexpr std::array kWants{"value", "jacobian", "hessian"};
    return std::format("<ddx.Call {} at {} point{}>",
                       kWants[static_cast<std::size_t>(want_)], at_.size(),
                       at_.size() == 1 ? "" : "s");
  }

private:
  // Into the dense block a caller reads; every other column of a colour is
  // structurally zero.
  void scatter(const PyEquation::Lane &l) {
    const std::size_t n = eq_->arity();
    const rt::Coloring &c = l.graph->coloring();
    const auto cells = compressed_->rows();
    const auto dims = std::views::iota(0uz, n);
    for (const auto [i, j] : std::views::cartesian_product(dims, dims)) {
      double *const out = h_->at(i * n + j);
      if (c.target(c.color[j], i) == j) {
        std::ranges::copy_n(cells[c.column(c.color[j], i)], at_.size(), out);
      } else {
        std::ranges::fill_n(out, static_cast<std::ptrdiff_t>(at_.size()), 0.0);
      }
    }
  }

  // A rank-0 block is the scalar case, where the allocating calls answer with
  // a float; matching them costs a PyFloat only when the value is read.
  [[nodiscard]] static pyb::object read(const std::optional<Block> &b) {
    if (!b) {
      fail_with(errc::wrong_column_count);
    }
    const Array &a = b->bound();
    return a.ndim() == 0 ? pyb::object{pyb::float_(*a.data())}
                         : pyb::object{a};
  }

  pyb::object owner_; // the equation, kept alive under the raw pointer below
  PyEquation *eq_;
  PyEquation::Want want_;
  Array x_;   // declared before at_, which points into it
  Point at_;
  std::optional<Block> f_;
  std::optional<Scratch> partials_; // the pattern's cells, on their way to g_
  std::optional<Block> g_;
  std::optional<Scratch> compressed_; // the colouring's, on their way to h_
  std::optional<Block> h_;
};

} // namespace ddx::py
