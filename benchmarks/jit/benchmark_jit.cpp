#include "ddx.hpp"
#include "jit/kernel.hpp"
#include "rt/bridge.hpp"
#include "rt/derivative.hpp"
#include "rt/graph.hpp"

#include <benchmark/benchmark.h>

#include <vector>

// What the JIT is and is not for.  Per point it cannot beat AOT-compiled ddx:
// the expression is already straight-line code there, and README measures the
// libm call at around three quarters of a gradient.  The batch kernel wins by
// vectorising exactly that call, so these run the same gradient both ways over
// the same points and report items/s.

namespace {
using ddx::rt::Builder;
using ddx::rt::Graph;

constexpr auto sym_x = ddx::var<"x">;
constexpr auto sym_y = ddx::var<"y">;

std::vector<double> ramp(std::size_t n, double base) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = base + 0.5 * static_cast<double>(i % 97) / 97.0;
  }
  return v;
}

struct Batch {
  std::vector<double> x, y, f, dx, dy;
  explicit Batch(std::size_t n)
      : x(ramp(n, 0.3)), y(ramp(n, 0.7)), f(n), dx(n), dy(n) {}
};

// Compile the gradient of a compile-time expression, through the bridge.
ddx::jit::Kernel compile_gradient(ddx::jit::Compiler &c, const auto &expr,
                                  Builder<> &b) {
  const auto root = ddx::rt::to_graph(b, expr);
  return *c.compile(ddx::rt::GraphBuilder{b}.value(root).gradient().build());
}

template <ddx::impl::CExpression E> void aot_gradient(benchmark::State &state, E expr) {
  const auto n = static_cast<std::size_t>(state.range(0));
  Batch data(n);
  // Store the results rather than discarding them: the kernel has to write its
  // columns, and at a million points that traffic outweighs the arithmetic, so
  // a loop that keeps its answer in a register is not the same measurement.
  // One gradient() call, which is ddx's natural shape -- the kernel also hands
  // back f from the same pass, which ddx would need a second sweep for, so this
  // comparison is if anything generous to the AOT side.
  for (auto _ : state) {
    for (std::size_t i = 0; i < n; ++i) {
      const auto g = ddx::Equation{expr}.gradient(data.x[i], data.y[i]);
      data.dx[i] = g[0];
      data.dy[i] = g[1];
    }
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(n) * state.iterations());
}

template <ddx::impl::CExpression E> void jit_gradient(benchmark::State &state, E expr) {
  const auto n = static_cast<std::size_t>(state.range(0));
  Batch data(n);
  auto compiler = *ddx::jit::Compiler::create();
  Builder<> b;
  const auto kernel = compile_gradient(compiler, expr, b);

  const std::array<const double *, 2> xs{data.x.data(), data.y.data()};
  std::array<double *, 2> gp{data.dx.data(), data.dy.data()};
  double *const value_columns[]{data.f.data()};
  for (auto _ : state) {
    kernel(xs, value_columns, gp, {}, n);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(n) * state.iterations());
}

// What it costs to get a kernel at all -- the price of the dynamism, paid once.
void jit_compile_time(benchmark::State &state) {
  auto compiler = *ddx::jit::Compiler::create();
  for (auto _ : state) {
    Builder<> b;
    auto k = compile_gradient(compiler, exp(sym_x) * sin(sym_y), b);
    benchmark::DoNotOptimize(k);
  }
}

const auto transcendental = exp(sym_x) * sin(sym_y);
const auto polynomial = sym_x * sym_x * sym_x + sym_y * sym_y - sym_x * sym_y;

} // namespace

BENCHMARK_CAPTURE(aot_gradient, transcendental, transcendental)
    ->Arg(1)
    ->Arg(1000)
    ->Arg(1000000);
BENCHMARK_CAPTURE(jit_gradient, transcendental, transcendental)
    ->Arg(1)
    ->Arg(1000)
    ->Arg(1000000);
BENCHMARK_CAPTURE(aot_gradient, polynomial, polynomial)
    ->Arg(1)
    ->Arg(1000)
    ->Arg(1000000);
BENCHMARK_CAPTURE(jit_gradient, polynomial, polynomial)
    ->Arg(1)
    ->Arg(1000)
    ->Arg(1000000);
BENCHMARK(jit_compile_time)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
