#include "ddx.hpp"
#include "jit/kernel.hpp"
#include "rt/bridge.hpp"
#include "rt/derivative.hpp"
#include "rt/graph.hpp"

#include <benchmark/benchmark.h>

#include <vector>

// Per point the JIT cannot beat AOT-compiled ddx; the batch kernel wins by
// vectorising the arithmetic and amortising the call over the batch.  The libm
// calls stay scalar unless a caller asks for VecLib::Libmvec, so a
// transcendental-heavy graph is where the margin is thinnest.

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

// Compile the Jacobian of a compile-time expression, through the bridge.
ddx::jit::Kernel compile_jacobian(ddx::jit::Compiler &c, const auto &expr,
                                  Builder<> &b) {
  const auto root = ddx::rt::to_graph(b, expr);
  return *c.compile(ddx::rt::GraphBuilder{b}.value(root).build_jacobian().finish());
}

template <ddx::impl::CExpression E>
void aot_jacobian(benchmark::State &state, E expr) {
  const auto n = static_cast<std::size_t>(state.range(0));
  Batch data(n);
  // Stored, not discarded: at a million points the write traffic outweighs
  // the arithmetic, so a register-only loop measures something else.
  for (auto _ : state) {
    for (std::size_t i = 0; i < n; ++i) {
      const auto g = ddx::Equation{expr}.jacobian(data.x[i], data.y[i]);
      data.dx[i] = g[0];
      data.dy[i] = g[1];
    }
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(n) * state.iterations());
}

template <ddx::impl::CExpression E>
void jit_jacobian(benchmark::State &state, E expr) {
  const auto n = static_cast<std::size_t>(state.range(0));
  Batch data(n);
  auto compiler = *ddx::jit::Compiler::create();
  Builder<> b;
  const auto kernel = compile_jacobian(compiler, expr, b);

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
    auto k = compile_jacobian(compiler, exp(sym_x) * sin(sym_y), b);
    benchmark::DoNotOptimize(k);
  }
}

const auto transcendental = exp(sym_x) * sin(sym_y);
const auto polynomial = sym_x * sym_x * sym_x + sym_y * sym_y - sym_x * sym_y;

} // namespace

BENCHMARK_CAPTURE(aot_jacobian, transcendental, transcendental)
    ->Arg(1)
    ->Arg(1000)
    ->Arg(1000000);
BENCHMARK_CAPTURE(jit_jacobian, transcendental, transcendental)
    ->Arg(1)
    ->Arg(1000)
    ->Arg(1000000);
BENCHMARK_CAPTURE(aot_jacobian, polynomial, polynomial)
    ->Arg(1)
    ->Arg(1000)
    ->Arg(1000000);
BENCHMARK_CAPTURE(jit_jacobian, polynomial, polynomial)
    ->Arg(1)
    ->Arg(1000)
    ->Arg(1000000);
BENCHMARK(jit_compile_time)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
