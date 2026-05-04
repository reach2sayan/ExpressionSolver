#include "dual.hpp"
#include "equation.hpp"
#include "gradient.hpp"
#include "values.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include <benchmark/benchmark.h>
#include <array>
#include <cstddef>
#include <vector>

using namespace diff;

template <typename Eq>
static void run_symbolic(benchmark::State &state, Eq &eq) {
  for (auto _ : state) {
    auto gradients = eq.eval_derivatives();
    benchmark::DoNotOptimize(gradients);
    benchmark::ClobberMemory();
  }
}

template <CExpression Expr>
static void run_reverse(benchmark::State &state, const Expr &expr) {
  for (auto _ : state) {
    auto gradients = gradient<DiffMode::Reverse>(expr);
    benchmark::DoNotOptimize(gradients);
    benchmark::ClobberMemory();
  }
}

template <CExpression Expr, std::size_t N>
static void run_forward(
    benchmark::State &state, Expr &expr,
    const std::array<scalar_base_t<typename Expr::value_type>, N> &values) {
  for (auto _ : state) {
    auto gradients = derivative_tensor<1>(expr, values);
    benchmark::DoNotOptimize(gradients);
    benchmark::ClobberMemory();
  }
}

template <typename VE>
static void run_symbolic_jacobian(benchmark::State &state, VE &ve) {
  for (auto _ : state) {
    auto J = ve.template symbolic_mode_jac();
    benchmark::DoNotOptimize(J);
    benchmark::ClobberMemory();
  }
}

template <typename VE>
static void run_reverse_jacobian(benchmark::State &state, const VE &ve) {
  for (auto _ : state) {
    auto J = ve.template reverse_mode_jac();
    benchmark::DoNotOptimize(J);
    benchmark::ClobberMemory();
  }
}

template <typename VE>
static void run_forward_jacobian(benchmark::State &state, VE &ve) {
  for (auto _ : state) {
    auto J = ve.template derivative_tensor<1>();
    benchmark::DoNotOptimize(J);
    benchmark::ClobberMemory();
  }
}

static void BM_Symbolic_F1_Univariate(benchmark::State &state) {
  auto x = PV(1.25, 'x');
  auto eq = Equation(exp(x) * sin(x) + x * x * x + 2.0 * x);
  run_symbolic(state, eq);
}
BENCHMARK(BM_Symbolic_F1_Univariate);

static void BM_Forward_F1_Univariate(benchmark::State &state) {
  Variable<double, 'x'> x{1.25};
  auto expr = exp(x) * sin(x) + x * x * x + 2.0 * x;
  run_forward(state, expr, std::array{1.25});
}
BENCHMARK(BM_Forward_F1_Univariate);

static void BM_Reverse_F1_Univariate(benchmark::State &state) {
  auto x = PV(1.25, 'x');
  auto expr = exp(x) * sin(x) + x * x * x + 2.0 * x;
  run_reverse(state, expr);
}
BENCHMARK(BM_Reverse_F1_Univariate);

static void BM_Symbolic_F2_Bivariate(benchmark::State &state) {
  auto x = PV(1.3, 'x');
  auto y = PV(0.7, 'y');
  auto eq = Equation(x * y + sin(x) + y * y + exp(x + y));
  run_symbolic(state, eq);
}
BENCHMARK(BM_Symbolic_F2_Bivariate);

static void BM_Forward_F2_Bivariate(benchmark::State &state) {
  Variable<double, 'x'> x{1.3};
  Variable<double, 'y'> y{0.7};
  auto expr = x * y + sin(x) + y * y + exp(x + y);
  run_forward(state, expr, std::array{1.3, 0.7});
}
BENCHMARK(BM_Forward_F2_Bivariate);

static void BM_Reverse_F2_Bivariate(benchmark::State &state) {
  auto x = PV(1.3, 'x');
  auto y = PV(0.7, 'y');
  auto expr = x * y + sin(x) + y * y + exp(x + y);
  run_reverse(state, expr);
}
BENCHMARK(BM_Reverse_F2_Bivariate);

static void BM_Symbolic_F3_Trivariate(benchmark::State &state) {
  auto x = PV(0.9, 'x');
  auto y = PV(1.1, 'y');
  auto z = PV(0.4, 'z');
  auto eq = Equation(exp(x * y) + sin(z) * x + y * z + x * x * z);
  run_symbolic(state, eq);
}
BENCHMARK(BM_Symbolic_F3_Trivariate);

static void BM_Forward_F3_Trivariate(benchmark::State &state) {
  Variable<double, 'x'> x{0.9};
  Variable<double, 'y'> y{1.1};
  Variable<double, 'z'> z{0.4};
  auto expr = exp(x * y) + sin(z) * x + y * z + x * x * z;
  run_forward(state, expr, std::array{0.9, 1.1, 0.4});
}
BENCHMARK(BM_Forward_F3_Trivariate);

static void BM_Reverse_F3_Trivariate(benchmark::State &state) {
  auto x = PV(0.9, 'x');
  auto y = PV(1.1, 'y');
  auto z = PV(0.4, 'z');
  auto expr = exp(x * y) + sin(z) * x + y * z + x * x * z;
  run_reverse(state, expr);
}
BENCHMARK(BM_Reverse_F3_Trivariate);

static void BM_Symbolic_F4_FourVariables(benchmark::State &state) {
  auto x = PV(1.0, 'x');
  auto y = PV(0.5, 'y');
  auto z = PV(1.7, 'z');
  auto w = PV(M_PI / 6.0, 'w');
  auto eq =
      Equation((x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w);
  run_symbolic(state, eq);
}
BENCHMARK(BM_Symbolic_F4_FourVariables);

static void BM_Forward_F4_FourVariables(benchmark::State &state) {
  Variable<double, 'x'> x{1.0};
  Variable<double, 'y'> y{0.5};
  Variable<double, 'z'> z{1.7};
  Variable<double, 'w'> w{M_PI / 6.0};
  auto expr = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
  run_forward(state, expr,
              std::array{1.0, 0.5, 1.7, M_PI / 6.0});
}
BENCHMARK(BM_Forward_F4_FourVariables);

static void BM_Reverse_F4_FourVariables(benchmark::State &state) {
  auto x = PV(1.0, 'x');
  auto y = PV(0.5, 'y');
  auto z = PV(1.7, 'z');
  auto w = PV(M_PI / 6.0, 'w');
  auto expr = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
  run_reverse(state, expr);
}
BENCHMARK(BM_Reverse_F4_FourVariables);

static void BM_Symbolic_Vector_F4(benchmark::State &state) {
  auto x = PV(1.0, 'x');
  auto y = PV(0.5, 'y');
  auto z = PV(1.7, 'z');
  auto w = PV(M_PI / 6.0, 'w');
  auto ve =
      Equation((x + y) * (z - w) + exp(x * z), sin(y * w) + x * y * z * w);
  run_symbolic_jacobian(state, ve);
}
BENCHMARK(BM_Symbolic_Vector_F4);

static void BM_Forward_Vector_F4(benchmark::State &state) {
  Variable<double, 'x'> x{1.0};
  Variable<double, 'y'> y{0.5};
  Variable<double, 'z'> z{1.7};
  Variable<double, 'w'> w{M_PI / 6.0};
  auto ve =
      Equation((x + y) * (z - w) + exp(x * z), sin(y * w) + x * y * z * w);
  run_forward_jacobian(state, ve);
}
BENCHMARK(BM_Forward_Vector_F4);

static void BM_Reverse_Vector_F4(benchmark::State &state) {
  auto x = PV(1.0, 'x');
  auto y = PV(0.5, 'y');
  auto z = PV(1.7, 'z');
  auto w = PV(M_PI / 6.0, 'w');
  auto ve =
      Equation((x + y) * (z - w) + exp(x * z), sin(y * w) + x * y * z * w);
  run_reverse_jacobian(state, ve);
}
BENCHMARK(BM_Reverse_Vector_F4);

// ===========================================================================
// Parallel reverse-mode Jacobian — breakeven sweep (output_dim 2 / 4 / 6)
// Each row is  f_i(x,y,z,w) = exp(x*z) + sin(y*w) + x*y*z*w.
// Compare BM_Symbolic_Vector_F4 (2 rows) vs these to see thread-spawn overhead
// vs parallelism payoff as output_dim grows.
// ===========================================================================

static void BM_Reverse_Parallel_2Rows(benchmark::State &state) {
  auto x = PV(1.0, 'x');
  auto y = PV(0.5, 'y');
  auto z = PV(1.7, 'z');
  auto w = PV(M_PI / 6.0, 'w');
  auto f = [&] { return exp(x * z) + sin(y * w) + x * y * z * w; };
  auto ve = Equation(f(), f());
  run_reverse_jacobian(state, ve);
}
BENCHMARK(BM_Reverse_Parallel_2Rows);

static void BM_Symbolic_Parallel_2Rows(benchmark::State &state) {
  auto x = PV(1.0, 'x');
  auto y = PV(0.5, 'y');
  auto z = PV(1.7, 'z');
  auto w = PV(M_PI / 6.0, 'w');
  auto f = [&] { return exp(x * z) + sin(y * w) + x * y * z * w; };
  auto ve = Equation(f(), f());
  run_symbolic_jacobian(state, ve);
}
BENCHMARK(BM_Symbolic_Parallel_2Rows);

static void BM_Reverse_Parallel_4Rows(benchmark::State &state) {
  auto x = PV(1.0, 'x');
  auto y = PV(0.5, 'y');
  auto z = PV(1.7, 'z');
  auto w = PV(M_PI / 6.0, 'w');
  auto f = [&] { return exp(x * z) + sin(y * w) + x * y * z * w; };
  auto ve = Equation(f(), f(), f(), f());
  run_reverse_jacobian(state, ve);
}
BENCHMARK(BM_Reverse_Parallel_4Rows);

static void BM_Symbolic_Parallel_4Rows(benchmark::State &state) {
  auto x = PV(1.0, 'x');
  auto y = PV(0.5, 'y');
  auto z = PV(1.7, 'z');
  auto w = PV(M_PI / 6.0, 'w');
  auto f = [&] { return exp(x * z) + sin(y * w) + x * y * z * w; };
  auto ve = Equation(f(), f(), f(), f());
  run_symbolic_jacobian(state, ve);
}
BENCHMARK(BM_Symbolic_Parallel_4Rows);

static void BM_Reverse_Parallel_6Rows(benchmark::State &state) {
  auto x = PV(1.0, 'x');
  auto y = PV(0.5, 'y');
  auto z = PV(1.7, 'z');
  auto w = PV(M_PI / 6.0, 'w');
  auto f = [&] { return exp(x * z) + sin(y * w) + x * y * z * w; };
  auto ve = Equation(f(), f(), f(), f(), f(), f());
  run_reverse_jacobian(state, ve);
}
BENCHMARK(BM_Reverse_Parallel_6Rows);

static void BM_Symbolic_Parallel_6Rows(benchmark::State &state) {
  auto x = PV(1.0, 'x');
  auto y = PV(0.5, 'y');
  auto z = PV(1.7, 'z');
  auto w = PV(M_PI / 6.0, 'w');
  auto f = [&] { return exp(x * z) + sin(y * w) + x * y * z * w; };
  auto ve = Equation(f(), f(), f(), f(), f(), f());
  run_symbolic_jacobian(state, ve);
}
BENCHMARK(BM_Symbolic_Parallel_6Rows);

// ===========================================================================
// Parallel reverse — large heterogeneous system  f: ℝ⁴ → ℝ⁵
// Mix of trig, exp, and polynomial rows to stress the parallel path with
// diverse per-row work.
// ===========================================================================

static void BM_Reverse_Parallel_Large(benchmark::State &state) {
  auto x = PV(1.0, 'x');
  auto y = PV(0.5, 'y');
  auto z = PV(1.7, 'z');
  auto w = PV(M_PI / 6.0, 'w');
  auto ve =
      Equation(x * y + exp(z), sin(x) * cos(y) + z * w, exp(x + y) + z * z,
               x * z * w + sin(y), cos(x * y) + exp(z + w));
  run_reverse_jacobian(state, ve);
}
BENCHMARK(BM_Reverse_Parallel_Large);

static void BM_Symbolic_Parallel_Large(benchmark::State &state) {
  auto x = PV(1.0, 'x');
  auto y = PV(0.5, 'y');
  auto z = PV(1.7, 'z');
  auto w = PV(M_PI / 6.0, 'w');
  auto ve =
      Equation(x * y + exp(z), sin(x) * cos(y) + z * w, exp(x + y) + z * z,
               x * z * w + sin(y), cos(x * y) + exp(z + w));
  run_symbolic_jacobian(state, ve);
}
BENCHMARK(BM_Symbolic_Parallel_Large);

static void BM_Footprint_F4(benchmark::State &state) {
  auto xs = PV(1.0, 'x');
  auto ys = PV(0.5, 'y');
  auto zs = PV(1.7, 'z');
  auto ws = PV(M_PI / 6.0, 'w');
  auto sym_expr =
      (xs + ys) * (zs - ws) + exp(xs * zs) + sin(ys * ws) + xs * ys * zs * ws;
  auto sym_eq = Equation(sym_expr);

  using D = Dual<double>;
  auto xf = PDV(1.0, 'x');
  Variable<D, 'y'> yf{D{0.5}};
  Variable<D, 'z'> zf{D{1.7}};
  Variable<D, 'w'> wf{D{M_PI / 6.0}};
  auto fwd_expr =
      (xf + yf) * (zf - wf) + exp(xf * zf) + sin(yf * wf) + xf * yf * zf * wf;

  for (auto _ : state)
    benchmark::DoNotOptimize(sym_eq);

  state.counters["symbolic_expr_bytes"] = static_cast<double>(sizeof(sym_expr));
  state.counters["symbolic_eq_bytes"] = static_cast<double>(sizeof(sym_eq));
  state.counters["reverse_expr_bytes"] = static_cast<double>(sizeof(sym_expr));
  state.counters["forward_expr_bytes"] = static_cast<double>(sizeof(fwd_expr));
  state.counters["dual_value_bytes"] = static_cast<double>(sizeof(D));
}
BENCHMARK(BM_Footprint_F4);

static void BM_Symbolic_Batched_F4(benchmark::State &state) {
  const auto count = static_cast<std::size_t>(state.range(0));

  auto x0 = PV(1.0, 'x');
  auto y0 = PV(0.5, 'y');
  auto z0 = PV(1.7, 'z');
  auto w0 = PV(M_PI / 6.0, 'w');
  auto expr0 =
      (x0 + y0) * (z0 - w0) + exp(x0 * z0) + sin(y0 * w0) + x0 * y0 * z0 * w0;
  using equation_type = Equation<decltype(expr0)>;
  std::vector<equation_type> equations;
  equations.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    auto x = PV(1.0 + 0.001 * i, 'x');
    auto y = PV(0.5 + 0.001 * i, 'y');
    auto z = PV(1.7 + 0.001 * i, 'z');
    auto w = PV(M_PI / 6.0 + 0.001 * i, 'w');
    auto expr = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
    equations.emplace_back(expr);
  }

  for (auto _ : state) {
    double sink = 0.0;
    for (auto &eq : equations) {
      auto grads = eq.eval_derivatives();
      sink += grads[0];
    }
    benchmark::DoNotOptimize(sink);
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(count * sizeof(equation_type)));
  state.counters["object_bytes"] = static_cast<double>(sizeof(equation_type));
}
BENCHMARK(BM_Symbolic_Batched_F4)->Arg(256)->Arg(1024)->Arg(4096);

static void BM_Reverse_Batched_F4(benchmark::State &state) {
  const auto count = static_cast<std::size_t>(state.range(0));

  auto x0 = PV(1.0, 'x');
  auto y0 = PV(0.5, 'y');
  auto z0 = PV(1.7, 'z');
  auto w0 = PV(M_PI / 6.0, 'w');
  auto expr0 =
      (x0 + y0) * (z0 - w0) + exp(x0 * z0) + sin(y0 * w0) + x0 * y0 * z0 * w0;
  using expr_type = decltype(expr0);

  std::vector<expr_type> expressions;
  expressions.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    auto x = PV(1.0 + 0.001 * i, 'x');
    auto y = PV(0.5 + 0.001 * i, 'y');
    auto z = PV(1.7 + 0.001 * i, 'z');
    auto w = PV(M_PI / 6.0 + 0.001 * i, 'w');
    expressions.emplace_back((x + y) * (z - w) + exp(x * z) + sin(y * w) +
                             x * y * z * w);
  }

  for (auto _ : state) {
    double sink = 0.0;
    for (const auto &expr : expressions) {
      auto grads = gradient<DiffMode::Reverse>(expr);
      sink += grads[0];
    }
    benchmark::DoNotOptimize(sink);
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(count * sizeof(expr_type)));
  state.counters["object_bytes"] = static_cast<double>(sizeof(expr_type));
}
BENCHMARK(BM_Reverse_Batched_F4)->Arg(256)->Arg(1024)->Arg(4096);

static void BM_Forward_Batched_F4(benchmark::State &state) {
  const auto count = static_cast<std::size_t>(state.range(0));

  Variable<double, 'x'> x0{1.0};
  Variable<double, 'y'> y0{0.5};
  Variable<double, 'z'> z0{1.7};
  Variable<double, 'w'> w0{M_PI / 6.0};
  auto expr0 =
      (x0 + y0) * (z0 - w0) + exp(x0 * z0) + sin(y0 * w0) + x0 * y0 * z0 * w0;
  using expr_type = decltype(expr0);

  std::vector<expr_type> expressions;
  std::vector<std::array<double, 4>> points;
  expressions.reserve(count);
  points.reserve(count);

  for (std::size_t i = 0; i < count; ++i) {
    Variable<double, 'x'> x{1.0 + 0.001 * i};
    Variable<double, 'y'> y{0.5 + 0.001 * i};
    Variable<double, 'z'> z{1.7 + 0.001 * i};
    Variable<double, 'w'> w{M_PI / 6.0 + 0.001 * i};
    expressions.emplace_back((x + y) * (z - w) + exp(x * z) + sin(y * w) +
                             x * y * z * w);
    points.push_back({1.0 + 0.001 * i, 0.5 + 0.001 * i, 1.7 + 0.001 * i,
                      M_PI / 6.0 + 0.001 * i});
  }

  for (auto _ : state) {
    double sink = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
      auto grads = derivative_tensor<1>(expressions[i], points[i]);
      sink += grads(0);
    }
    benchmark::DoNotOptimize(sink);
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(count * sizeof(expr_type)));
  state.counters["object_bytes"] = static_cast<double>(sizeof(expr_type));
}
BENCHMARK(BM_Forward_Batched_F4)->Arg(256)->Arg(1024)->Arg(4096);

// ===========================================================================
// Dual-variable reverse mode (PDV path)
// ===========================================================================

static void BM_Reverse_Dual_F1_Univariate(benchmark::State &state) {
  auto x = PDV(1.25, 'x');
  auto expr = exp(x) * sin(x) + x * x * x + 2.0 * x;
  run_reverse(state, expr);
}
BENCHMARK(BM_Reverse_Dual_F1_Univariate);

static void BM_Reverse_Dual_F2_Bivariate(benchmark::State &state) {
  auto x = PDV(1.3, 'x');
  auto y = PDV(0.7, 'y');
  auto expr = x * y + sin(x) + y * y + exp(x + y);
  run_reverse(state, expr);
}
BENCHMARK(BM_Reverse_Dual_F2_Bivariate);

static void BM_Reverse_Dual_F4_FourVariables(benchmark::State &state) {
  auto x = PDV(1.0, 'x');
  auto y = PDV(0.5, 'y');
  auto z = PDV(1.7, 'z');
  auto w = PDV(M_PI / 6.0, 'w');
  auto expr = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
  run_reverse(state, expr);
}
BENCHMARK(BM_Reverse_Dual_F4_FourVariables);

static void BM_Reverse_Dual_Batched_F4(benchmark::State &state) {
  const auto count = static_cast<std::size_t>(state.range(0));

  auto x0 = PDV(1.0, 'x');
  auto y0 = PDV(0.5, 'y');
  auto z0 = PDV(1.7, 'z');
  auto w0 = PDV(M_PI / 6.0, 'w');
  auto expr0 =
      (x0 + y0) * (z0 - w0) + exp(x0 * z0) + sin(y0 * w0) + x0 * y0 * z0 * w0;
  using expr_type = decltype(expr0);

  std::vector<expr_type> expressions;
  expressions.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    auto x = PDV(1.0 + 0.001 * i, 'x');
    auto y = PDV(0.5 + 0.001 * i, 'y');
    auto z = PDV(1.7 + 0.001 * i, 'z');
    auto w = PDV(M_PI / 6.0 + 0.001 * i, 'w');
    expressions.emplace_back((x + y) * (z - w) + exp(x * z) + sin(y * w) +
                             x * y * z * w);
  }

  for (auto _ : state) {
    double sink = 0.0;
    for (const auto &expr : expressions) {
      auto grads = reverse_mode_grad(expr);
      sink += grads[0];
    }
    benchmark::DoNotOptimize(sink);
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(count * sizeof(expr_type)));
  state.counters["object_bytes"] = static_cast<double>(sizeof(expr_type));
}
BENCHMARK(BM_Reverse_Dual_Batched_F4)->Arg(256)->Arg(1024)->Arg(4096);

BENCHMARK_MAIN();
