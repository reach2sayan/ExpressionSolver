// Apples-to-apples comparison: this library vs. autodiff v1.1.2.
//
// Three scalar functions of increasing arity:
//   F1  f(x)       = exp(x)*sin(x) + x^3 + 2x          x=1.25
//   F2  f(x,y)     = xy + sin(x) + y^2 + exp(x+y)      (1.3, 0.7)
//   F4  f(x,y,z,w) = (x+y)(z-w) + exp(xz) + sin(yw) + xyzw
//
// Each function is benchmarked four ways:
//   Ours_Forward   — Dual<double> expression template, N dual passes
//   Ours_Reverse   — backward() accumulation, one pass
//   AD_Forward     — autodiff::dual, N derivative() calls
//   AD_Reverse     — autodiff::var, one derivatives() call
//
// Build with:  cmake -DDIFF_BUILD_COMPARE=ON -DDIFF_BUILD_BENCHMARKS=OFF ..
//              cmake --build . --target benchmarks_compare
// Run  with:  ./benchmarks_compare --benchmark_counters_tabular=true

#include <autodiff/forward/dual.hpp>
#include <autodiff/reverse/var.hpp>

#include "../include/gradient.hpp"
#include "dual.hpp"
#include "gradient.hpp"
#include "values.hpp"

#include <array>
#include <benchmark/benchmark.h>
#include <numbers>

using namespace diff;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <CExpression Expr, std::size_t N>
static void run_our_forward(benchmark::State &state, Expr &expr,
                             const std::array<dual_scalar_t<typename Expr::value_type>, N> &vals) {
    for (auto _ : state) {
        auto g = forward_mode_grad(expr, vals);
        benchmark::DoNotOptimize(g);
        benchmark::ClobberMemory();
    }
}

template <CExpression Expr>
static void run_our_reverse(benchmark::State &state, const Expr &expr) {
    for (auto _ : state) {
        auto g = reverse_mode_grad(expr);
        benchmark::DoNotOptimize(g);
        benchmark::ClobberMemory();
    }
}

// ===========================================================================
// F1  f(x) = exp(x)*sin(x) + x^3 + 2x   at x = 1.25
// ===========================================================================

static void BM_Ours_Forward_F1(benchmark::State &state) {
    using D = Dual<double>;
    Variable<D, 'x'> x{D{1.25}};
    auto expr = exp(x) * sin(x) + x * x * x + 2.0 * x;
    run_our_forward(state, expr, std::array{1.25});
}
BENCHMARK(BM_Ours_Forward_F1);

static void BM_Ours_Reverse_F1(benchmark::State &state) {
    auto x = PV(1.25, 'x');
    auto expr = exp(x) * sin(x) + x * x * x + 2.0 * x;
    run_our_reverse(state, expr);
}
BENCHMARK(BM_Ours_Reverse_F1);

static void BM_AD_Forward_F1(benchmark::State &state) {
    using autodiff::dual;
    using autodiff::detail::derivative;
    using autodiff::detail::wrt;
    using autodiff::detail::at;
    auto f = [](dual x) -> dual { return exp(x) * sin(x) + x * x * x + 2.0 * x; };
    dual x = 1.25;
    for (auto _ : state) {
        double dx = derivative(f, wrt(x), at(x));
        benchmark::DoNotOptimize(dx);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_AD_Forward_F1);

static void BM_AD_Reverse_F1(benchmark::State &state) {
    using autodiff::var;
    using autodiff::reverse::detail::derivatives;
    using autodiff::reverse::detail::wrt;
    var x = 1.25;
    for (auto _ : state) {
        var u = exp(x) * sin(x) + x * x * x + 2.0 * x;
        auto [dx] = derivatives(u, wrt(x));
        benchmark::DoNotOptimize(dx);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_AD_Reverse_F1);

// ===========================================================================
// F2  f(x,y) = xy + sin(x) + y^2 + exp(x+y)   at (1.3, 0.7)
// ===========================================================================

static void BM_Ours_Forward_F2(benchmark::State &state) {
    using D = Dual<double>;
    Variable<D, 'x'> x{D{1.3}};
    Variable<D, 'y'> y{D{0.7}};
    auto expr = x * y + sin(x) + y * y + exp(x + y);
    run_our_forward(state, expr, std::array{1.3, 0.7});
}
BENCHMARK(BM_Ours_Forward_F2);

static void BM_Ours_Reverse_F2(benchmark::State &state) {
    auto x = PV(1.3, 'x');
    auto y = PV(0.7, 'y');
    auto expr = x * y + sin(x) + y * y + exp(x + y);
    run_our_reverse(state, expr);
}
BENCHMARK(BM_Ours_Reverse_F2);

static void BM_AD_Forward_F2(benchmark::State &state) {
    using autodiff::dual;
    using autodiff::detail::derivative;
    using autodiff::detail::wrt;
    using autodiff::detail::at;
    auto f = [](dual x, dual y) -> dual { return x * y + sin(x) + y * y + exp(x + y); };
    dual x = 1.3, y = 0.7;
    for (auto _ : state) {
        double dx = derivative(f, wrt(x), at(x, y));
        double dy = derivative(f, wrt(y), at(x, y));
        benchmark::DoNotOptimize(dx);
        benchmark::DoNotOptimize(dy);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_AD_Forward_F2);

static void BM_AD_Reverse_F2(benchmark::State &state) {
    using autodiff::var;
    using autodiff::reverse::detail::derivatives;
    using autodiff::reverse::detail::wrt;
    var x = 1.3, y = 0.7;
    for (auto _ : state) {
        var u = x * y + sin(x) + y * y + exp(x + y);
        auto [dx, dy] = derivatives(u, wrt(x, y));
        benchmark::DoNotOptimize(dx);
        benchmark::DoNotOptimize(dy);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_AD_Reverse_F2);

// ===========================================================================
// F4  f(x,y,z,w) = (x+y)(z-w) + exp(xz) + sin(yw) + xyzw
// ===========================================================================

static const double W0 = std::numbers::pi_v<double> / 6.0;

static void BM_Ours_Forward_F4(benchmark::State &state) {
    using D = Dual<double>;
    Variable<D, 'x'> x{D{1.0}};
    Variable<D, 'y'> y{D{0.5}};
    Variable<D, 'z'> z{D{1.7}};
    Variable<D, 'w'> w{D{W0}};
    auto expr = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
    run_our_forward(state, expr, std::array{1.0, 0.5, 1.7, W0});
}
BENCHMARK(BM_Ours_Forward_F4);

static void BM_Ours_Reverse_F4(benchmark::State &state) {
    double w0 = W0;
    auto x = PV(1.0, 'x');
    auto y = PV(0.5, 'y');
    auto z = PV(1.7, 'z');
    auto w = PV(w0, 'w');
    auto expr = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
    run_our_reverse(state, expr);
}
BENCHMARK(BM_Ours_Reverse_F4);

static void BM_AD_Forward_F4(benchmark::State &state) {
    using autodiff::dual;
    using autodiff::detail::derivative;
    using autodiff::detail::wrt;
    using autodiff::detail::at;
    auto f = [](dual x, dual y, dual z, dual w) -> dual {
        return (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
    };
    dual x = 1.0, y = 0.5, z = 1.7, w = W0;
    for (auto _ : state) {
        double dx = derivative(f, wrt(x), at(x, y, z, w));
        double dy = derivative(f, wrt(y), at(x, y, z, w));
        double dz = derivative(f, wrt(z), at(x, y, z, w));
        double dw = derivative(f, wrt(w), at(x, y, z, w));
        benchmark::DoNotOptimize(dx);
        benchmark::DoNotOptimize(dy);
        benchmark::DoNotOptimize(dz);
        benchmark::DoNotOptimize(dw);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_AD_Forward_F4);

static void BM_AD_Reverse_F4(benchmark::State &state) {
    using autodiff::var;
    using autodiff::reverse::detail::derivatives;
    using autodiff::reverse::detail::wrt;
    var x = 1.0, y = 0.5, z = 1.7, w = W0;
    for (auto _ : state) {
        var u = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
        auto [dx, dy, dz, dw] = derivatives(u, wrt(x, y, z, w));
        benchmark::DoNotOptimize(dx);
        benchmark::DoNotOptimize(dy);
        benchmark::DoNotOptimize(dz);
        benchmark::DoNotOptimize(dw);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_AD_Reverse_F4);

BENCHMARK_MAIN();
