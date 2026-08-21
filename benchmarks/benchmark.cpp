#include "ddx.hpp"
#include "drivers/symbolic.hpp"
#include "dual/dual.hpp"
#include "expr/bound.hpp"
#include "expr/equation.hpp"
#include "expr/values.hpp"
#define _USE_MATH_DEFINES
#include <array>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <math.h>
#include <vector>

using namespace ddx::impl;

template <typename Eq, CNumericBuffer Pt>
  requires requires(Eq &eq, Pt pt) {
    eq.template gradient<ddx::DiffMode::Symbolic>(pt);
  }
static void run_symbolic(benchmark::State &state, Eq &eq, Pt pt) {
  for (auto _ : state) {
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(pt);
    auto gradients = eq.template gradient<ddx::DiffMode::Symbolic>(pt);
    benchmark::DoNotOptimize(gradients);
    benchmark::ClobberMemory();
  }
}

template <CExpression Expr, CNumericBuffer Pt>
static void run_reverse(benchmark::State &state, Expr &expr, Pt pt) {
  for (auto _ : state) {
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(pt);
    auto gradients = Equation{expr}.gradient(pt);
    benchmark::DoNotOptimize(gradients);
    benchmark::ClobberMemory();
  }
}

template <CExpression Expr, std::size_t N>
static void
run_forward(benchmark::State &state, Expr &expr,
            std::array<scalar_base_t<typename Expr::value_type>, N> values) {
  for (auto _ : state) {
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(values);
    auto gradients = Equation{expr}.template derivative_tensor<1>(values);
    benchmark::DoNotOptimize(gradients);
    benchmark::ClobberMemory();
  }
}

template <typename VE, CNumericBuffer Pt>
  requires requires(VE &ve, Pt pt) {
    ve.template jacobian<ddx::DiffMode::Symbolic>(pt);
  }
static void run_symbolic_jacobian(benchmark::State &state, VE &ve, Pt pt) {
  for (auto _ : state) {
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(pt);
    auto J = ve.template jacobian<ddx::DiffMode::Symbolic>(pt);
    benchmark::DoNotOptimize(J);
    benchmark::ClobberMemory();
  }
}

template <typename VE, CNumericBuffer Pt>
  requires requires(VE &ve, Pt pt) { ve.jacobian(pt); }
static void run_reverse_jacobian(benchmark::State &state, VE &ve, Pt pt) {
  for (auto _ : state) {
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(pt);
    auto J = ve.jacobian(pt);
    benchmark::DoNotOptimize(J);
    benchmark::ClobberMemory();
  }
}

template <typename VE, CNumericBuffer Pt>
  requires requires(VE &ve, Pt pt) { ve.template derivative_tensor<1>(pt); }
static void run_forward_jacobian(benchmark::State &state, VE &ve, Pt pt) {
  for (auto _ : state) {
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(pt);
    auto J = ve.template derivative_tensor<1>(pt);
    benchmark::DoNotOptimize(J);
    benchmark::ClobberMemory();
  }
}

static void BM_Symbolic_F1_Univariate(benchmark::State &state) {
  double xv = 1.25;
  benchmark::DoNotOptimize(xv);
  auto x = var_of<"x">(xv);
  auto eq = Equation(exp(x) * sin(x) + x * x * x + 2.0 * x);
  run_symbolic(state, eq, std::array{xv});
}
BENCHMARK(BM_Symbolic_F1_Univariate);

static void BM_Forward_F1_Univariate(benchmark::State &state) {
  double xv = 1.25;
  benchmark::DoNotOptimize(xv);
  Variable<double, FixedString{"x"}> x;
  auto expr = exp(x) * sin(x) + x * x * x + 2.0 * x;
  run_forward(state, expr, std::array{xv});
}
BENCHMARK(BM_Forward_F1_Univariate);

static void BM_Reverse_F1_Univariate(benchmark::State &state) {
  double xv = 1.25;
  benchmark::DoNotOptimize(xv);
  auto x = var_of<"x">(xv);
  auto expr = exp(x) * sin(x) + x * x * x + 2.0 * x;
  run_reverse(state, expr, std::array{xv});
}
BENCHMARK(BM_Reverse_F1_Univariate);

static void BM_Symbolic_F2_Bivariate(benchmark::State &state) {
  double xv = 1.3, yv = 0.7;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto eq = Equation(x * y + sin(x) + y * y + exp(x + y));
  run_symbolic(state, eq, std::array{xv, yv});
}
BENCHMARK(BM_Symbolic_F2_Bivariate);

static void BM_Forward_F2_Bivariate(benchmark::State &state) {
  double xv = 1.3, yv = 0.7;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  auto expr = x * y + sin(x) + y * y + exp(x + y);
  run_forward(state, expr, std::array{xv, yv});
}
BENCHMARK(BM_Forward_F2_Bivariate);

static void BM_Reverse_F2_Bivariate(benchmark::State &state) {
  double xv = 1.3, yv = 0.7;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto expr = x * y + sin(x) + y * y + exp(x + y);
  run_reverse(state, expr, std::array{xv, yv});
}
BENCHMARK(BM_Reverse_F2_Bivariate);

static void BM_Symbolic_F3_Trivariate(benchmark::State &state) {
  double xv = 0.9, yv = 1.1, zv = 0.4;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  benchmark::DoNotOptimize(zv);
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto z = var_of<"z">(zv);
  auto eq = Equation(exp(x * y) + sin(z) * x + y * z + x * x * z);
  run_symbolic(state, eq, std::array{xv, yv, zv});
}
BENCHMARK(BM_Symbolic_F3_Trivariate);

static void BM_Forward_F3_Trivariate(benchmark::State &state) {
  double xv = 0.9, yv = 1.1, zv = 0.4;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  benchmark::DoNotOptimize(zv);
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  Variable<double, FixedString{"z"}> z;
  auto expr = exp(x * y) + sin(z) * x + y * z + x * x * z;
  run_forward(state, expr, std::array{xv, yv, zv});
}
BENCHMARK(BM_Forward_F3_Trivariate);

static void BM_Reverse_F3_Trivariate(benchmark::State &state) {
  double xv = 0.9, yv = 1.1, zv = 0.4;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  benchmark::DoNotOptimize(zv);
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto z = var_of<"z">(zv);
  auto expr = exp(x * y) + sin(z) * x + y * z + x * x * z;
  run_reverse(state, expr, std::array{xv, yv, zv});
}
BENCHMARK(BM_Reverse_F3_Trivariate);

static void BM_Symbolic_F4_FourVariables(benchmark::State &state) {
  double xv = 1.0, yv = 0.5, zv = 1.7, wv = M_PI / 6.0;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  benchmark::DoNotOptimize(zv);
  benchmark::DoNotOptimize(wv);
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto z = var_of<"z">(zv);
  auto w = var_of<"w">(wv);
  auto eq =
      Equation((x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w);
  run_symbolic(state, eq, std::array{wv, xv, yv, zv});
}
BENCHMARK(BM_Symbolic_F4_FourVariables);

static void BM_Forward_F4_FourVariables(benchmark::State &state) {
  double xv = 1.0, yv = 0.5, zv = 1.7, wv = M_PI / 6.0;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  benchmark::DoNotOptimize(zv);
  benchmark::DoNotOptimize(wv);
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  Variable<double, FixedString{"z"}> z;
  Variable<double, FixedString{"w"}> w;
  auto expr = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
  run_forward(state, expr, std::array{xv, yv, zv, wv});
}
BENCHMARK(BM_Forward_F4_FourVariables);

static void BM_Reverse_F4_FourVariables(benchmark::State &state) {
  double xv = 1.0, yv = 0.5, zv = 1.7, wv = M_PI / 6.0;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  benchmark::DoNotOptimize(zv);
  benchmark::DoNotOptimize(wv);
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto z = var_of<"z">(zv);
  auto w = var_of<"w">(wv);
  auto expr = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
  run_reverse(state, expr, std::array{wv, xv, yv, zv});
}
BENCHMARK(BM_Reverse_F4_FourVariables);

static void BM_Symbolic_Vector_F4(benchmark::State &state) {
  double xv = 1.0, yv = 0.5, zv = 1.7, wv = M_PI / 6.0;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  benchmark::DoNotOptimize(zv);
  benchmark::DoNotOptimize(wv);
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto z = var_of<"z">(zv);
  auto w = var_of<"w">(wv);
  auto ve =
      Equation((x + y) * (z - w) + exp(x * z), sin(y * w) + x * y * z * w);
  run_symbolic_jacobian(state, ve, std::array{wv, xv, yv, zv});
}
BENCHMARK(BM_Symbolic_Vector_F4);

static void BM_Forward_Vector_F4(benchmark::State &state) {
  double xv = 1.0, yv = 0.5, zv = 1.7, wv = M_PI / 6.0;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  benchmark::DoNotOptimize(zv);
  benchmark::DoNotOptimize(wv);
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  Variable<double, FixedString{"z"}> z;
  Variable<double, FixedString{"w"}> w;
  auto ve =
      Equation((x + y) * (z - w) + exp(x * z), sin(y * w) + x * y * z * w);
  run_forward_jacobian(state, ve, std::array{wv, xv, yv, zv});
}
BENCHMARK(BM_Forward_Vector_F4);

static void BM_Reverse_Vector_F4(benchmark::State &state) {
  double xv = 1.0, yv = 0.5, zv = 1.7, wv = M_PI / 6.0;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  benchmark::DoNotOptimize(zv);
  benchmark::DoNotOptimize(wv);
  auto x = var_of<"x">(xv);
  auto y = var_of<"y">(yv);
  auto z = var_of<"z">(zv);
  auto w = var_of<"w">(wv);
  auto ve =
      Equation((x + y) * (z - w) + exp(x * z), sin(y * w) + x * y * z * w);
  run_reverse_jacobian(state, ve, std::array{wv, xv, yv, zv});
}
BENCHMARK(BM_Reverse_Vector_F4);

static void BM_Footprint_F4(benchmark::State &state) {
  auto xs = var<"x">;
  auto ys = var<"y">;
  auto zs = var<"z">;
  auto ws = var<"w">;
  auto sym_expr =
      (xs + ys) * (zs - ws) + exp(xs * zs) + sin(ys * ws) + xs * ys * zs * ws;
  auto sym_eq = Equation(sym_expr);

  using D = Dual<double>;
  auto xf = var<"x", dual>;
  Variable<D, FixedString{"y"}> yf;
  Variable<D, FixedString{"z"}> zf;
  Variable<D, FixedString{"w"}> wf;
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

// Batched point coordinates ramp for this many steps and then repeat, so every
// batch size samples the same range of values. An unbounded ramp would grow the
// arguments to sin/exp with the batch size and push libm into progressively
// more expensive argument reduction, which shows up as a per-point cost that
// climbs with the batch and makes the sizes incomparable.
static constexpr std::size_t kRampPeriod = 256;

static void BM_Symbolic_Batched_F4(benchmark::State &state) {
  const auto count = static_cast<std::size_t>(state.range(0));

  // The expression carries no values, so a batch is one expression and N
  // points rather than N copies of the tree.
  constexpr auto x = var<"x">;
  constexpr auto y = var<"y">;
  constexpr auto z = var<"z">;
  constexpr auto w = var<"w">;
  auto eq = Equation((x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w);

  using point_type = std::array<double, 4>; // canonical order: w, x, y, z
  std::vector<point_type> points;
  points.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const double t = 0.001 * static_cast<double>(i % kRampPeriod);
    points.push_back({M_PI / 6.0 + t, 1.0 + t, 0.5 + t, 1.7 + t});
  }

  for (auto _ : state) {
    for (const auto &pt : points) {
      auto grads = eq.template gradient<ddx::DiffMode::Symbolic>(pt);
      benchmark::DoNotOptimize(grads);
    }
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(count * sizeof(point_type)));
  state.counters["object_bytes"] = static_cast<double>(sizeof(point_type));
  state.counters["equation_bytes"] = static_cast<double>(sizeof(eq));
}
BENCHMARK(BM_Symbolic_Batched_F4)->Arg(256)->Arg(1024)->Arg(4096);

static void BM_Reverse_Batched_F4(benchmark::State &state) {
  const auto count = static_cast<std::size_t>(state.range(0));

  constexpr auto x = var<"x">;
  constexpr auto y = var<"y">;
  constexpr auto z = var<"z">;
  constexpr auto w = var<"w">;
  auto expr = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;

  using point_type = std::array<double, 4>; // canonical order: w, x, y, z
  std::vector<point_type> points;
  points.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const double t = 0.001 * static_cast<double>(i % kRampPeriod);
    points.push_back({M_PI / 6.0 + t, 1.0 + t, 0.5 + t, 1.7 + t});
  }

  for (auto _ : state) {
    for (const auto &pt : points) {
      auto grads = Equation{expr}.gradient(pt);
      benchmark::DoNotOptimize(grads);
    }
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(count * sizeof(point_type)));
  state.counters["object_bytes"] = static_cast<double>(sizeof(point_type));
  state.counters["expr_bytes"] = static_cast<double>(sizeof(expr));
}
BENCHMARK(BM_Reverse_Batched_F4)->Arg(256)->Arg(1024)->Arg(4096);

static void BM_Forward_Batched_F4(benchmark::State &state) {
  const auto count = static_cast<std::size_t>(state.range(0));

  Variable<double, FixedString{"x"}> x0;
  Variable<double, FixedString{"y"}> y0;
  Variable<double, FixedString{"z"}> z0;
  Variable<double, FixedString{"w"}> w0;
  auto expr0 =
      (x0 + y0) * (z0 - w0) + exp(x0 * z0) + sin(y0 * w0) + x0 * y0 * z0 * w0;
  using expr_type = decltype(expr0);

  std::vector<expr_type> expressions;
  std::vector<std::array<double, 4>> points;
  expressions.reserve(count);
  points.reserve(count);

  for (std::size_t i = 0; i < count; ++i) {
    Variable<double, FixedString{"x"}> x;
    Variable<double, FixedString{"y"}> y;
    Variable<double, FixedString{"z"}> z;
    Variable<double, FixedString{"w"}> w;
    expressions.emplace_back((x + y) * (z - w) + exp(x * z) + sin(y * w) +
                             x * y * z * w);
    const double t = 0.001 * static_cast<double>(i % kRampPeriod);
    points.push_back({1.0 + t, 0.5 + t, 1.7 + t, M_PI / 6.0 + t});
  }

  for (auto _ : state) {
    for (std::size_t i = 0; i < count; ++i) {
      auto grads = Equation{expressions[i]}.template derivative_tensor<1>(points[i]);
      benchmark::DoNotOptimize(grads);
    }
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
  double xv = 1.25;
  benchmark::DoNotOptimize(xv);
  auto x = dual_var_of<"x">(xv);
  auto expr = exp(x) * sin(x) + x * x * x + 2.0 * x;
  run_reverse(state, expr, std::array{Dual<double>{xv, 0.0}});
}
BENCHMARK(BM_Reverse_Dual_F1_Univariate);

static void BM_Reverse_Dual_F2_Bivariate(benchmark::State &state) {
  double xv = 1.3, yv = 0.7;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  auto x = dual_var_of<"x">(xv);
  auto y = dual_var_of<"y">(yv);
  auto expr = x * y + sin(x) + y * y + exp(x + y);
  run_reverse(state, expr, std::array{Dual<double>{xv, 0.0}, Dual<double>{yv, 0.0}});
}
BENCHMARK(BM_Reverse_Dual_F2_Bivariate);

static void BM_Reverse_Dual_F4_FourVariables(benchmark::State &state) {
  double xv = 1.0, yv = 0.5, zv = 1.7, wv = M_PI / 6.0;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  benchmark::DoNotOptimize(zv);
  benchmark::DoNotOptimize(wv);
  auto x = dual_var_of<"x">(xv);
  auto y = dual_var_of<"y">(yv);
  auto z = dual_var_of<"z">(zv);
  auto w = dual_var_of<"w">(wv);
  auto expr = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
  run_reverse(state, expr, std::array{Dual<double>{wv, 0.0}, Dual<double>{xv, 0.0}, Dual<double>{yv, 0.0}, Dual<double>{zv, 0.0}});
}
BENCHMARK(BM_Reverse_Dual_F4_FourVariables);

static void BM_Reverse_Dual_Batched_F4(benchmark::State &state) {
  const auto count = static_cast<std::size_t>(state.range(0));
  using D = Dual<double>;

  constexpr auto x = var<"x", D>;
  constexpr auto y = var<"y", D>;
  constexpr auto z = var<"z", D>;
  constexpr auto w = var<"w", D>;
  auto expr = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;

  using point_type = std::array<D, 4>; // canonical order: w, x, y, z
  std::vector<point_type> points;
  points.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const double t = 0.001 * static_cast<double>(i % kRampPeriod);
    points.push_back({D{M_PI / 6.0 + t}, D{1.0 + t}, D{0.5 + t}, D{1.7 + t}});
  }

  for (auto _ : state) {
    for (const auto &pt : points) {
      auto grads = Equation{expr}.gradient(pt);
      benchmark::DoNotOptimize(grads);
    }
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
  state.SetBytesProcessed(state.iterations() *
                          static_cast<int64_t>(count * sizeof(point_type)));
  state.counters["object_bytes"] = static_cast<double>(sizeof(point_type));
  state.counters["expr_bytes"] = static_cast<double>(sizeof(expr));
}
BENCHMARK(BM_Reverse_Dual_Batched_F4)->Arg(256)->Arg(1024)->Arg(4096);

BENCHMARK_MAIN();
