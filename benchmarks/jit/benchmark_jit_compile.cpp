// What a compile costs as a graph grows, phase by phase.
//
// Not a Google Benchmark loop: each cell is compiled once, because the question
// is the cost of the compile itself and it is measured in seconds, not
// nanoseconds.  What matters is the exponent -- which phase is superlinear in
// node count -- so the table prints the log-log slope against the row above.
//
// Sizes ascend and a model stops after the first cell over the budget, since a
// compile in progress cannot be abandoned and the next size is only larger.

#include "jit/kernel.hpp"
#include "rt/derivative.hpp"
#include "rt/expressions.hpp"
#include "rt/graph.hpp"

#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ddx::rt::Builder;
using RE = ddx::rt::RTExpression<double>;

// --- the models ---------------------------------------------------------------
// The same three shapes the comparison harness uses, thin: rss is O(n) entropy
// with an O(n^2) interaction, uniquac nests a sum inside a log, and mse puts
// uniquac underneath a second O(n^2) double sum.  Deterministic parameters of
// the right magnitude; the thermodynamics is not what is being measured.

constexpr double pseudo(std::size_t i, std::size_t j, double lo, double hi) {
  const double t = static_cast<double>((i * 7919 + j * 104729) % 1000) / 1000.0;
  return lo + t * (hi - lo);
}

RE rss(std::span<const RE> x) {
  const std::size_t n = x.size();
  RE g = x[0] * log(x[0]);
  for (std::size_t i = 1; i < n; ++i) {
    g = g + x[i] * log(x[i]);
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      g = g + pseudo(i, j, 0.2, 0.8) * x[i] * x[j];
    }
  }
  return g;
}

RE uniquac(std::span<const RE> x) {
  const std::size_t n = x.size();
  RE sum_r = pseudo(0, 1, 0.9, 5.2) * x[0];
  RE sum_q = pseudo(0, 2, 0.8, 4.4) * x[0];
  for (std::size_t i = 1; i < n; ++i) {
    sum_r = sum_r + pseudo(i, 1, 0.9, 5.2) * x[i];
    sum_q = sum_q + pseudo(i, 2, 0.8, 4.4) * x[i];
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
    out = out + x[i] * log(phi[i] / x[i]);
  }
  // The residual: a sum over every species inside a log, per species.  This is
  // the O(n^2) term, and the one that makes the graph deep as well as wide.
  for (std::size_t i = 0; i < n; ++i) {
    RE inner = theta[0] * pseudo(0, i, 0.2, 1.8);
    for (std::size_t j = 1; j < n; ++j) {
      inner = inner + theta[j] * pseudo(j, i, 0.2, 1.8);
    }
    out = out - pseudo(i, 2, 0.8, 4.4) * x[i] * log(inner);
  }
  return out;
}

RE mse(std::span<const RE> x) {
  const std::size_t n = x.size();
  RE ionic = 0.5 * pseudo(0, 5, -2.0, 2.0) * pseudo(0, 5, -2.0, 2.0) * x[0];
  for (std::size_t i = 1; i < n; ++i) {
    const double z = pseudo(i, 5, -2.0, 2.0);
    ionic = ionic + 0.5 * z * z * x[i];
  }
  const RE damp = exp(-sqrt(ionic));
  RE mr = -0.3915 * sqrt(ionic) / (1.0 + 1.2 * sqrt(ionic));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      mr = mr + x[i] * x[j] * pseudo(i, j, -0.4, 0.4) * damp;
    }
  }
  return mr + uniquac(x);
}

struct Model {
  std::string_view name;
  RE (*build)(std::span<const RE>);
};

constexpr Model kModels[] = {
    {"rss", rss}, {"uniquac", uniquac}, {"mse", mse}};

// --- one cell ------------------------------------------------------------------

struct Cell {
  std::size_t n = 0;
  double total_ms = 0;
  ddx::jit::CompileReport report;
};

[[nodiscard]] double ms(std::chrono::nanoseconds d) {
  return static_cast<double>(d.count()) / 1e6;
}

// The gradient of one model at n variables, frozen with its value and every
// partial, which is what a caller compiles.
ddx::rt::Graph<double> gradient_graph(const Model &m, std::size_t n) {
  Builder<> b;
  std::vector<RE> v;
  v.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    v.push_back(ddx::rt::var(b, "x" + std::to_string(i)));
  }
  const ddx::rt::NodeId root = m.build(std::span<const RE>{v}).id(b);
  const auto row = ddx::rt::jacobian<ddx::impl::DiffMode::Reverse>(b, root);
  return ddx::rt::GraphBuilder{b}
      .values_from(std::span<const ddx::rt::NodeId>{&root, 1})
      .jacobian_from(row.partial)
      .build();
}


// --- what the kernel it produced is worth ------------------------------------
// Compile time is only half the trade: a pipeline that compiles faster and
// produces a slower kernel is not a saving.  Min of several passes over the
// same batch, reported per point.
[[nodiscard]] double kernel_ns_per_point(const ddx::jit::Kernel &k,
                                         std::size_t n, std::size_t count) {
  std::vector<std::vector<double>> in(n, std::vector<double>(count));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t p = 0; p < count; ++p) {
      in[i][p] = 0.05 + 0.9 * static_cast<double>((i * 37 + p * 11) % 97) / 97.0;
    }
  }
  std::vector<double> value(count);
  std::vector<std::vector<double>> partial(n, std::vector<double>(count));

  std::vector<const double *> xs(n);
  std::vector<double *> gs(n);
  for (std::size_t i = 0; i < n; ++i) {
    xs[i] = in[i].data();
    gs[i] = partial[i].data();
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

// log-log slope against the row above: 1.0 is linear in node count, 2.0
// quadratic.  The exponent is the whole point of the table.
[[nodiscard]] double slope(double t1, double t0, double n1, double n0) {
  return (t0 <= 0.0 || n0 <= 0.0) ? 0.0 : std::log(t1 / t0) / std::log(n1 / n0);
}

} // namespace

int main(int argc, char **argv) {
  double budget_s = 120.0;
  ddx::jit::Options options;
  std::vector<std::size_t> sizes{16, 32, 64, 128, 256, 512};

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    if (arg == "--lean") {
      options.pipeline = ddx::jit::Pipeline::Lean;
    } else if (arg == "--default-pipeline") {
      options.pipeline = ddx::jit::Pipeline::Default;
    } else if (arg == "--time-passes") {
      options.time_passes = true;
    } else if (arg.starts_with("--split-above=")) {
      options.split_above = std::strtoull(arg.data() + 14, nullptr, 10);
    } else if (arg.starts_with("--slab-factor=")) {
      options.slab_factor = std::strtoull(arg.data() + 14, nullptr, 10);
    } else if (arg.starts_with("--min-slab=")) {
      options.min_slab = std::strtoull(arg.data() + 11, nullptr, 10);
    } else if (arg.starts_with("--budget=")) {
      budget_s = std::strtod(arg.data() + 9, nullptr);
    } else if (arg.starts_with("--sizes=")) {
      std::vector<std::size_t> v;
      const std::string_view list = arg.substr(8);
      for (std::size_t p = 0; p < list.size();) {
        std::size_t value = 0;
        const auto *const end = list.data() + list.size();
        const auto r = std::from_chars(list.data() + p, end, value);
        if (r.ec != std::errc{}) {
          break;
        }
        v.push_back(value);
        p = static_cast<std::size_t>(r.ptr - list.data()) + 1;
      }
      if (!v.empty()) {
        sizes = v;
      }
    }
  }

  auto compiler = ddx::jit::Compiler::create();
  if (!compiler) {
    std::fprintf(stderr, "JIT unavailable: %s\n", compiler.error().detail.c_str());
    return 1;
  }

  std::printf("compile cost against graph size, budget %.0fs per cell\n\n",
              budget_s);
  std::printf("%-8s %5s %9s %10s %9s %9s %9s %9s %7s %10s %6s %9s\n", "model", "n",
              "nodes", "ir instrs", "emit ms", "opt ms", "codegen ms",
              "total ms", "slope", "kernel ns", "slabs", "scratch");

  for (const auto &m : kModels) {
    std::vector<Cell> cells;
    for (const std::size_t n : sizes) {
      const auto graph = gradient_graph(m, n);

      Cell cell{.n = n, .total_ms = 0.0, .report = {}};
      const auto start = std::chrono::steady_clock::now();
      auto kernel = compiler->compile(graph, options, &cell.report);
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
      std::printf("%-8s %5zu %9zu %10zu %9.1f %9.1f %9.1f %9.1f %7.2f %10.1f %6zu %9zu\n",
                  m.name.data(), n, cell.report.nodes, cell.report.instructions,
                  ms(cell.report.emit), ms(cell.report.optimize),
                  ms(cell.report.codegen), cell.total_ms, sl,
                  kernel_ns_per_point(*kernel, n, 4096), cell.report.slabs,
                  cell.report.scratch_slots);
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
