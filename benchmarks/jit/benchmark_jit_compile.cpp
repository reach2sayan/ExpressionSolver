// What a compile costs as a graph grows, phase by phase.
//
// Not a Google Benchmark loop: each cell is compiled once, the compile being
// measured in seconds.  What matters is the exponent -- which phase is
// superlinear in node count -- so the table prints the log-log slope against
// the row above.  Sizes ascend and a model stops after the first cell over the
// budget, a compile in progress not being abandonable.
//
// `--ladder` asks whether a compile was worth starting: each cell at codegen 0
// and at the stated level, both kernels and the sweep they replace, and how
// many points each has to be spread over before it has paid for itself.

#include "jit/kernel.hpp"
#include "rt/derivative.hpp"
#include "rt/expressions.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "util/ranges.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ddx::rt::Builder;
using RE = ddx::rt::RTExpression<double>;

// --- the models ---------------------------------------------------------------
// The same three shapes the comparison harness uses: rss is O(n) entropy with
// an O(n^2) interaction, uniquac nests a sum inside a log, and mse puts uniquac
// underneath a second double sum.  The thermodynamics is not what is measured.

constexpr double pseudo(std::size_t i, std::size_t j, double lo, double hi) {
  const double t = static_cast<double>((i * 7919 + j * 104729) % 1000) / 1000.0;
  return lo + t * (hi - lo);
}

RE rss(std::span<const RE> x) {
  const std::size_t n = x.size();
  RE g = x[0] * log(x[0]);
  for (std::size_t i = 1; i < n; ++i) {
    g += x[i] * log(x[i]);
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      g += pseudo(i, j, 0.2, 0.8) * x[i] * x[j];
    }
  }
  return g;
}

RE uniquac(std::span<const RE> x) {
  const std::size_t n = x.size();
  RE sum_r = pseudo(0, 1, 0.9, 5.2) * x[0];
  RE sum_q = pseudo(0, 2, 0.8, 4.4) * x[0];
  for (std::size_t i = 1; i < n; ++i) {
    sum_r += pseudo(i, 1, 0.9, 5.2) * x[i];
    sum_q += pseudo(i, 2, 0.8, 4.4) * x[i];
  }

  std::vector<RE> phi, theta;
  phi.reserve(n);
  theta.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    phi.push_back(pseudo(i, 1, 0.9, 5.2) * x[i] / sum_r);
    theta.push_back(pseudo(i, 2, 0.8, 4.4) * x[i] / sum_q);
  }

  RE out = x[0] * log(phi[0] / x[0]);
  for (std::size_t i = 1; i < n; ++i) {
    out += x[i] * log(phi[i] / x[i]);
  }
  // The residual: a sum over every species inside a log, per species.  This is
  // the O(n^2) term, and the one that makes the graph deep as well as wide.
  for (std::size_t i = 0; i < n; ++i) {
    RE inner = theta[0] * pseudo(0, i, 0.2, 1.8);
    for (std::size_t j = 1; j < n; ++j) {
      inner += theta[j] * pseudo(j, i, 0.2, 1.8);
    }
    out -= pseudo(i, 2, 0.8, 4.4) * x[i] * log(inner);
  }
  return out;
}

RE mse(std::span<const RE> x) {
  const std::size_t n = x.size();
  RE ionic = 0.5 * pseudo(0, 5, -2.0, 2.0) * pseudo(0, 5, -2.0, 2.0) * x[0];
  for (std::size_t i = 1; i < n; ++i) {
    const double z = pseudo(i, 5, -2.0, 2.0);
    ionic += 0.5 * z * z * x[i];
  }
  const RE damp = exp(-sqrt(ionic));
  RE mr = -0.3915 * sqrt(ionic) / (1.0 + 1.2 * sqrt(ionic));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      mr += x[i] * x[j] * pseudo(i, j, -0.4, 0.4) * damp;
    }
  }
  return mr + uniquac(x);
}

// Peng-Robinson's cubic in Z with van der Waals mixing; the last variable is Z.
// Arithmetic only, where the other three are transcendental-heavy.
RE pr(std::span<const RE> v) {
  const auto x = v.first(v.size() - 1);
  const RE &Z = v.back();
  const std::size_t n = x.size();
  RE a_mix = x[0] * x[0] * pseudo(0, 3, 0.3, 2.6);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double aij = std::sqrt(pseudo(i, 3, 0.3, 2.6) * pseudo(j, 3, 0.3, 2.6)) *
                         (1.0 - pseudo(i, j, 0.0, 0.12));
      a_mix += x[i] * x[j] * aij;
    }
  }
  RE b_mix = x[0] * pseudo(0, 4, 0.02, 0.09);
  for (std::size_t i = 1; i < n; ++i) {
    b_mix += x[i] * pseudo(i, 4, 0.02, 0.09);
  }
  const RE A = a_mix * 0.45724;
  const RE B = b_mix * 0.07780;
  return Z * Z * Z - (1.0 - B) * Z * Z + (A - 3.0 * B * B - 2.0 * B) * Z -
         (A * B - B * B - B * B * B);
}

struct Model {
  std::string_view name;
  RE (*build)(std::span<const RE>);
};

constexpr Model kModels[] = {
    {"rss", rss}, {"uniquac", uniquac}, {"pr", pr}, {"mse", mse}};

// --- one cell ------------------------------------------------------------------

struct Cell {
  std::size_t n = 0;
  double total_ms = 0;
  ddx::jit::CompileReport report;
};

[[nodiscard]] double ms(std::chrono::nanoseconds d) {
  return static_cast<double>(d.count()) / 1e6;
}

// A frozen graph and the arena it was frozen from, which the sweep still reads:
// a frozen graph carries the schedule, the nodes stay where they were built.
struct Frozen {
  std::unique_ptr<Builder<>> arena;
  ddx::rt::Graph<double> graph;
};

// The gradient of one model at n variables, frozen with its value and every
// partial, which is what a caller compiles.
Frozen gradient_graph(const Model &m, std::size_t n, bool contract = true) {
  auto arena = std::make_unique<Builder<>>();
  std::vector<RE> v;
  v.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    v.push_back(ddx::rt::var(*arena, "x" + std::to_string(i)));
  }
  const ddx::rt::NodeId root = m.build(std::span<const RE>{v}).id(*arena);
  // The Jacobian block is sparse: a partial the sweep folded to zero has no
  // column, so the buffers below are sized by the graph, not by n.
  auto graph = ddx::rt::GraphBuilder{*arena}
                   .values_from(std::span<const ddx::rt::NodeId>{&root, 1})
                   .build_jacobian()
                   .finish(contract);
  return {.arena = std::move(arena), .graph = std::move(graph)};
}


// --- what the kernel it produced is worth ------------------------------------
// A pipeline that compiles faster and produces a slower kernel is not a saving.
// Min of several passes over the same batch, reported per point.
[[nodiscard]] double kernel_ns_per_point(const ddx::jit::Kernel &k,
                                         const Frozen &fr, std::size_t n,
                                         std::size_t count) {
  std::vector<std::vector<double>> in(n, std::vector<double>(count));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t p = 0; p < count; ++p) {
      in[i][p] = 0.05 + 0.9 * static_cast<double>((i * 37 + p * 11) % 97) / 97.0;
    }
  }
  const std::size_t columns = fr.graph.layout().jacobian;
  std::vector<double> value(count);
  std::vector<std::vector<double>> partial(columns, std::vector<double>(count));

  std::vector<const double *> xs(n);
  std::vector<double *> gs(columns);
  for (std::size_t i = 0; i < n; ++i) {
    xs[i] = in[i].data();
  }
  for (std::size_t c = 0; c < columns; ++c) {
    gs[c] = partial[c].data();
  }
  double *v = value.data();

  double best = 1e300;
  for (int rep = 0; rep < 5; ++rep) {
    const auto start = std::chrono::steady_clock::now();
    k(xs, std::span<double *const>{&v, 1}, gs, {}, count);
    const double ns =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count()) /
        static_cast<double>(count);
    best = ns < best ? ns : best;
  }
  return best;
}

// --- what the sweep it replaces is worth --------------------------------------
// A compile only repays itself against what the interpreter would have cost.
// Equation::interpret's block sweep at the same width, so this is what a caller
// under Backend::Interpret actually gets.
constexpr std::size_t kSweepLanes = 8; // Equation::kLanes

[[nodiscard]] double interp_ns_per_point(const Frozen &fr, std::size_t n,
                                         std::size_t count) {
  const auto blocks = fr.graph.output_blocks();
  const auto order = fr.graph.contracted_order();
  const auto contractions = fr.graph.contractions();

  std::vector<std::vector<double>> in(n, std::vector<double>(count));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t p = 0; p < count; ++p) {
      in[i][p] = 0.05 + 0.9 * static_cast<double>((i * 37 + p * 11) % 97) / 97.0;
    }
  }
  std::vector<double> value(count);
  std::vector<std::vector<double>> partial(fr.graph.layout().jacobian,
                                           std::vector<double>(count));

  std::vector<double> lanes(fr.graph.symbols().size() * kSweepLanes);
  std::vector<double> tape(fr.arena->size() * kSweepLanes);

  double best = 1e300;
  for (int rep = 0; rep < 3; ++rep) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t base = 0; base < count; base += kSweepLanes) {
      const std::size_t width = std::min(kSweepLanes, count - base);
      for (std::size_t j = 0; j < n; ++j) {
        double *const dst = lanes.data() + j * kSweepLanes;
        double *const tail =
            std::ranges::copy_n(in[j].data() + base, width, dst).out;
        std::ranges::fill(tail, dst + kSweepLanes, in[j][base + width - 1]);
      }
      ddx::rt::evaluate_block<kSweepLanes>(*fr.arena,
                                           std::span<const double>{lanes},
                                           order, contractions,
                                           std::span<double>{tape});
      std::ranges::copy_n(tape.data() + std::size_t{blocks.values[0]} * kSweepLanes,
                          width, value.data() + base);
      for (const auto [column, o] : std::views::zip(partial, blocks.jacobian)) {
        std::ranges::copy_n(tape.data() + std::size_t{o} * kSweepLanes, width,
                            column.data() + base);
      }
    }
    const double ns =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count()) /
        static_cast<double>(count);
    best = ns < best ? ns : best;
  }
  return best;
}

// How many points a compile has to be spread over before it has paid for
// itself against what it replaced.  Infinite where the thing compiled is not
// actually faster, which is a real outcome and prints as such.
[[nodiscard]] double break_even(double compile_ms, double replaced_ns,
                                double with_ns) {
  const double saved = replaced_ns - with_ns;
  return saved <= 0.0 ? std::numeric_limits<double>::infinity()
                      : compile_ms * 1e6 / saved;
}

// log-log slope against the row above: 1.0 is linear in node count, 2.0
// quadratic.  The exponent is the whole point of the table.
[[nodiscard]] double slope(double t1, double t0, double n1, double n0) {
  return (t0 <= 0.0 || n0 <= 0.0) ? 0.0 : std::log(t1 / t0) / std::log(n1 / n0);
}

// --- the ladder ---------------------------------------------------------------
// What Backend::Compile looks like with a cheap rung under it.  Every number in
// a row comes from one session against one graph: this machine moves an
// unchanged binary by up to 70% between runs, so only within-row means anything.
//
//   t0/t1     compile ms at codegen 0 and at the stated level
//   k0/k1     what each kernel is worth, ns per point
//   sweep     what the interpreter costs for the same point
//   be0/be1   points before each rung has paid for its own compile, from
//             scratch, against the sweep
//   climb     the marginal question the ladder actually asks: given rung 0 is
//             already running, how many points before rung 1 repays *its*
//             compile out of the difference between the two kernels
[[nodiscard]] int run_ladder(ddx::jit::Compiler &compiler,
                             ddx::jit::Options options,
                             std::span<const std::size_t> sizes,
                             double budget_s) {
  const unsigned top = options.codegen_level;
  std::printf("tier ladder, codegen 0 under codegen %u, budget %.0fs per cell, "
              "lanes %u (0 = host), opt %u\n\n",
              top, budget_s, options.lanes, options.opt_level);
  if (top == 0) {
    std::printf("codegen 0 is the top rung: there is no ladder to measure.\n");
    return 0;
  }
  std::printf("%-8s %5s %9s %9s %9s %6s %10s %10s %10s %10s %10s %10s\n",
              "model", "n", "nodes", "t0 ms", "t1 ms", "t1/t0", "sweep ns",
              "k0 ns", "k1 ns", "be0", "be1", "climb");

  ddx::jit::Options low = options;
  low.codegen_level = 0;

  for (const auto &m : kModels) {
    for (const std::size_t n : sizes) {
      const auto frozen = gradient_graph(m, n, options.contract);

      ddx::jit::CompileReport r0;
      const auto s0 = std::chrono::steady_clock::now();
      auto k0 = compiler.compile(frozen.graph, low, &r0);
      const double t0 = ms(std::chrono::steady_clock::now() - s0);

      ddx::jit::CompileReport r1;
      const auto s1 = std::chrono::steady_clock::now();
      auto k1 = compiler.compile(frozen.graph, options, &r1);
      const double t1 = ms(std::chrono::steady_clock::now() - s1);

      if (!k0 || !k1) {
        std::fprintf(stderr, "%s n=%zu failed: %s\n", m.name.data(), n,
                     (!k0 ? k0 : k1).error().detail.c_str());
        break;
      }

      const double sweep =
          interp_ns_per_point(frozen, n, 256);
      const double n0 = kernel_ns_per_point(*k0, frozen, n, 4096);
      const double n1 = kernel_ns_per_point(*k1, frozen, n, 4096);

      std::printf("%-8s %5zu %9zu %9.1f %9.1f %6.2f %10.1f %10.1f %10.1f "
                  "%10.0f %10.0f %10.0f\n",
                  m.name.data(), n, r1.nodes, t0, t1, t0 > 0.0 ? t1 / t0 : 0.0,
                  sweep, n0, n1, break_even(t0, sweep, n0),
                  break_even(t1, sweep, n1), break_even(t1, n0, n1));
      std::fflush(stdout);

      if (t1 / 1000.0 > budget_s) {
        std::printf("%-8s stopping: %.1fs is over the %.0fs budget\n",
                    m.name.data(), t1 / 1000.0, budget_s);
        break;
      }
    }
    std::printf("\n");
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  double budget_s = 120.0;
  bool ladder = false;
  ddx::jit::Options options;
  std::vector<std::size_t> sizes{16, 32, 64, 128, 256, 512};

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    if (arg.starts_with("--lanes=")) {
      options.lanes = static_cast<unsigned>(std::strtoul(arg.data() + 8, nullptr, 10));
    } else if (arg.starts_with("--opt=")) {
      options.opt_level = static_cast<unsigned>(std::strtoul(arg.data() + 6, nullptr, 10));
    } else if (arg.starts_with("--codegen=")) {
      options.codegen_level = static_cast<unsigned>(std::strtoul(arg.data() + 10, nullptr, 10));
    } else if (arg == "--slp") {
      options.slp = true;
    } else if (arg == "--loop-vectorize") {
      options.loop_vectorize = true;
    } else if (arg == "--time-passes") {
      options.time_passes = true;
    } else if (arg == "--ladder") {
      ladder = true;
    } else if (arg.starts_with("--cache=")) {
      options.cache_dir = std::string{arg.substr(8)};
    } else if (arg.starts_with("--budget=")) {
      budget_s = std::strtod(arg.data() + 9, nullptr);
    } else if (arg.starts_with("--sizes=")) {
      const auto parsed =
          arg.substr(8) | std::views::split(',') |
          std::views::transform([](const auto field) {
            std::size_t value = 0;
            std::from_chars(std::ranges::data(field),
                            std::ranges::data(field) + std::ranges::size(field),
                            value);
            return value;
          }) |
          std::views::filter([](std::size_t n) { return n != 0; }) |
          ddx::impl::to<std::vector<std::size_t>>();
      if (!parsed.empty()) {
        sizes = parsed;
      }
    }
  }

  auto compiler = ddx::jit::Compiler::create();
  if (!compiler) {
    std::fprintf(stderr, "JIT unavailable: %s\n", compiler.error().detail.c_str());
    return 1;
  }

  if (ladder) {
    return run_ladder(*compiler, options, sizes, budget_s);
  }

  std::printf("compile cost against graph size, budget %.0fs per cell, "
              "lanes %u (0 = host), opt %u, codegen %u\n\n",
              budget_s, options.lanes, options.opt_level,
              options.codegen_level);
  std::printf("%-8s %5s %9s %10s %9s %9s %9s %9s %7s %10s\n", "model", "n",
              "nodes", "ir instrs", "emit ms", "opt ms", "codegen ms",
              "total ms", "slope", "kernel ns");

  for (const auto &m : kModels) {
    std::vector<Cell> cells;
    for (const std::size_t n : sizes) {
      const auto frozen = gradient_graph(m, n, options.contract);

      Cell cell{.n = n, .total_ms = 0.0, .report = {}};
      const auto start = std::chrono::steady_clock::now();
      auto kernel = compiler->compile(frozen.graph, options, &cell.report);
      cell.total_ms = ms(std::chrono::steady_clock::now() - start);
      if (!kernel) {
        std::fprintf(stderr, "%s n=%zu failed: %s\n", m.name.data(), n,
                     kernel.error().detail.c_str());
        break;
      }

      const double sl =
          cells.empty()
              ? 0.0
              : slope(cell.total_ms, cells.back().total_ms,
                      static_cast<double>(cell.report.nodes),
                      static_cast<double>(cells.back().report.nodes));
      std::printf("%-8s %5zu %9zu %10zu %9.1f %9.1f %9.1f %9.1f %7.2f %10.1f\n",
                  m.name.data(), n, cell.report.nodes, cell.report.instructions,
                  ms(cell.report.emit), ms(cell.report.optimize),
                  ms(cell.report.codegen), cell.total_ms, sl,
                  kernel_ns_per_point(*kernel, frozen, n, 4096));
      std::fflush(stdout);

      cells.push_back(cell);
      if (cell.total_ms / 1000.0 > budget_s) {
        std::printf("%-8s stopping: %.1fs is over the %.0fs budget\n",
                    m.name.data(), cell.total_ms / 1000.0, budget_s);
        break;
      }
    }
    std::printf("\n");
  }
  return 0;
}
