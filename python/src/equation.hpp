#pragma once

#include "columns.hpp"
#include "error.hpp"
#include "expression.hpp"

#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/dot.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "util/ranges.hpp"

// Unconditional, both of them: jit::Options and jit::Kernel are header types,
// so the Python surface has one shape whether or not the backend was compiled
// in.  Only bringing an LLJIT up needs the library, and that is compiler().
#include "jit/kernel.hpp"
#include "rt/equation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <format>
#include <future>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

// What ddx::impl::Equation is, with the output count a runtime value.
//
// That count is the only thing keeping the facade out of Python: it is a
// template parameter there (output_dim = 1 + sizeof...(Rest)), chosen from
// std::tuple_size_v.  Everything underneath already takes it as a number --
// GraphBuilder::values_from(span), Compiler::compile_async, and jit::Kernel,
// which reports its column counts at run time -- so this assembles the same
// three lanes off those pieces and holds `roots_` in a vector.
//
// It is much smaller than the facade for one reason: the GIL is the lock.  The
// facade needs a shared_mutex, a republished shared_ptr<const Compiled> and the
// settling/adopt dance because C++ callers race.  Python callers cannot: every
// freeze, poll and adopt below runs holding the GIL, and only the kernel or the
// sweep itself gives it up.
namespace ddx::py {

class PyEquation {
public:
  // Ordered, and each level is the one below plus a block, which is what lets
  // one pass through prepare() build any of them.
  enum class Want : std::uint8_t { Values, Jacobian, Hessian };

  PyEquation(std::shared_ptr<rt::Builder<double>> arena,
             std::vector<rt::NodeId> roots)
      : arena_(std::move(arena)), roots_(std::move(roots)) {}

  [[nodiscard]] std::size_t arity() const { return arena_->symbols().size(); }
  [[nodiscard]] std::size_t outputs() const { return roots_.size(); }
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

  // --- the JIT -------------------------------------------------------------

  [[nodiscard]] const jit::Options &options() const noexcept {
    return options_;
  }

  // Discards whatever was compiled, so the choice holds however late it is
  // made.  Choosing a compiling backend is also what starts the compile, so it
  // overlaps whatever the caller does next rather than landing in their first
  // call.
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
  // graph is swept, and the kernel replaces the sweep the moment it arrives.
  [[nodiscard]] bool wait_for_kernel() {
    Lane &l = lane(Want::Jacobian);
    if (l.pending.valid()) {
      // The GIL is what serialises everything else here, so it is given up
      // around the wait alone -- never around the lane bookkeeping.
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
  // that compiles something is a repr nobody can call in a debugger.
  [[nodiscard]] std::string repr() const {
    return std::format("<ddx.Equation ({}) -> {} output{}>",
                       symbols() | std::views::join_with(std::string{", "}) |
                           impl::to<std::string>(),
                       outputs(), outputs() == 1 ? "" : "s");
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

  // Freezing a lane and launching its compile are one step, deliberately.
  // Split, a lane could be ready having compiled nothing, and the next caller
  // would find it ready, conclude there was nothing to do, and interpret
  // forever -- with every answer still right, so nothing would ever say so.
  void prepare(Lane &l, Want want) {
    rt::GraphBuilder<double> gb{*arena_};
    gb.values_from(roots_);
    if (want != Want::Values) {
      gb.build_jacobian();
    }
    if (want == Want::Hessian) {
      gb.build_hessian();
    }
    l.graph = std::make_shared<const rt::Graph<double>>(gb.build());
    if (options_.backend == jit::Backend::Interpret) {
      return;
    }
    if (jit::Compiler *const c = compiler()) {
      l.pending = c->compile_async(l.graph, effective_options());
    }
  }

  // Publish a compile that has landed.  A refused one leaves the kernel empty
  // and the sweep simply stays -- the JIT is never a correctness dependency.
  void adopt([[maybe_unused]] Lane &l) {
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

  // How wide to emit, from the batch the caller said they have.  A kernel `w`
  // lanes wide does a register's worth of work to answer for a single point.
  [[nodiscard]] jit::Options effective_options() const noexcept {
    jit::Options opt = options_;
    if (opt.lanes == 0 && opt.points < kLanes) {
      opt.lanes = 1;
    }
    return opt;
  }

  // The process's one LLJIT, borrowed rather than founded: a module that made
  // its own would stand a second one up beside whatever C++ Equations in the
  // same process are already using, which is the thing that static exists to
  // prevent.
  //
  // Null in a build without the backend, which is the only difference such a
  // build makes: prepare() launches nothing, no kernel ever lands, and the
  // sweep answers everything -- the same path a Compile backend takes anyway
  // while its compile is still in flight.
  [[nodiscard]] static jit::Compiler *compiler() {
#ifdef DDX_HAS_JIT
    return impl::rt_detail::shared_compiler();
#else
    return nullptr;
#endif
  }

  // Whether this equation's arithmetic contracts a multiply and an add into an
  // FMA.  Decided in the graph, so both paths agree: a batch answers the same
  // whichever one answered it.
  [[nodiscard]] bool contracts() const noexcept {
    return options_.contract;
  }

  using BlockMember = std::span<const rt::NodeId> rt::Graph<double>::Blocks::*;

  [[nodiscard]] static std::size_t count(const Lane &l, BlockMember which) {
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
    // Past here nothing touches Python, so the GIL goes -- which is also what
    // lets another thread call into this equation while a long batch runs.
    const pyb::gil_scoped_release unlocked;
    if (l.kernel) {
      l.kernel(xs, f, g, h, at.size());
      return;
    }
    interpret(*l.graph, xs, f, g, h, at.size());
  }

  // The block sweep, which runs within ~1.4x of a kernel and is not a sad path.
  void interpret(const rt::Graph<double> &graph,
                 std::span<const double *const> xs, std::span<double *const> f,
                 std::span<double *const> g, std::span<double *const> h,
                 std::size_t n) const {
    const auto blocks = graph.output_blocks();
    const bool contract = contracts();
    const auto order =
        contract ? graph.contracted_order() : graph.live_order();
    const std::size_t symbols = arity();
    const auto scatter = [&](auto &&tape, std::size_t i, std::size_t stride,
                             std::size_t width) {
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
    // since the freeze, and live ids index below that.
    //
    // A batch shorter than a block sweeps one point at a time rather than
    // padding out to kLanes -- padding is nearly free on the tail of a long
    // batch and is the whole cost on a short one.
    if (n < kLanes) {
      std::vector<double> at(symbols);
      std::vector<double> tape(arena_->size());
      for (const std::size_t i : std::views::iota(0uz, n)) {
        std::ranges::transform(xs, at.begin(),
                               [i](const double *column) { return column[i]; });
        rt::evaluate_into(*arena_, at, order, std::span<double>{tape},
                          contract);
        scatter(tape, i, 1, 1);
      }
      return;
    }

    // kLanes points per sweep: the switch is paid once per node per block, and
    // each operation becomes a lane loop wide enough to vectorise.  A short
    // final block repeats its last point; the repeated lanes are never read.
    std::vector<double> lanes(symbols * kLanes);
    std::vector<double> tape(arena_->size() * kLanes);

    for (std::size_t base = 0; base < n; base += kLanes) {
      const std::size_t width = std::min(kLanes, n - base);
      for (const auto [j, column] : std::views::enumerate(xs)) {
        double *const dst = lanes.data() + static_cast<std::size_t>(j) * kLanes;
        double *const tail = std::ranges::copy_n(column + base, width, dst).out;
        std::ranges::fill(tail, dst + kLanes, column[base + width - 1]);
      }
      rt::evaluate_block<kLanes>(*arena_, std::span<const double>{lanes}, order,
                                 std::span<double>{tape}, contract);
      scatter(tape, base, kLanes, width);
    }
  }

  // --- the Hessian ---------------------------------------------------------

  // One function: the compiled lane holds it compressed by colour, and the
  // colouring is what says which (colour, row) column owns a given (i, j).
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
  // single colouring, so there is no lane that could hold m of these -- which
  // is also why this one is a point at a time and says so rather than quietly
  // walking the whole arena once per column of a batch.
  [[nodiscard]] pyb::tuple hessian_from_arena(const Point &at) {
    if (at.batched() && at.size() != 1) {
      fail_with(errc::wrong_column_count);
    }
    const std::size_t n = arity();
    const auto &blocks = sweeps();

    std::vector<double> point(n);
    std::ranges::transform(at.columns(), point.begin(),
                           [](const double *c) { return *c; });
    const auto values = rt::evaluate_all(*arena_, point);

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

  // Swept once and kept.  Hash consing makes a repeat sweep return the same
  // ids and add no nodes, so this is about the colouring's cost, not the
  // arena's size.
  [[nodiscard]] const std::vector<rt::Hessian> &sweeps() {
    if (!sweeps_) {
      sweeps_.emplace();
      sweeps_->reserve(roots_.size());
      for (const rt::NodeId r : roots_) {
        sweeps_->push_back(rt::build_hessian_impl(*arena_, r));
      }
    }
    return *sweeps_;
  }

  // --- shapes --------------------------------------------------------------

  // The leading axis goes when there is one function and the trailing one when
  // the caller gave one point, so a scalar model at a point answers with a
  // float, an (n,) gradient and an (n, n) Hessian rather than three arrays with
  // 1s in them.
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
  std::optional<std::vector<rt::Hessian>> sweeps_;
  jit::Options options_{};
  std::array<Lane, 3> lanes_;
};

} // namespace ddx::py
