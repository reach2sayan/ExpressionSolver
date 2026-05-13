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
static void
run_our_forward(benchmark::State &state, Expr &expr,
                std::array<dual_scalar_t<typename Expr::value_type>, N> vals) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(vals);
    auto g = derivative_tensor<1>(expr, vals);
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
  double xv = 1.25;
  benchmark::DoNotOptimize(xv);
  auto x = PV(xv, 'x');
  auto expr = exp(x) * sin(x) + x * x * x + 2.0 * x;
  run_our_reverse(state, expr);
}
BENCHMARK(BM_Ours_Reverse_F1);

static void BM_AD_Forward_F1(benchmark::State &state) {
  using autodiff::dual;
  using autodiff::detail::at;
  using autodiff::detail::derivative;
  using autodiff::detail::wrt;
  auto f = [](dual x) -> dual { return exp(x) * sin(x) + x * x * x + 2.0 * x; };
  dual x = 1.25;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
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
  double xv = 1.3, yv = 0.7;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  auto x = PV(xv, 'x');
  auto y = PV(yv, 'y');
  auto expr = x * y + sin(x) + y * y + exp(x + y);
  run_our_reverse(state, expr);
}
BENCHMARK(BM_Ours_Reverse_F2);

static void BM_AD_Forward_F2(benchmark::State &state) {
  using autodiff::dual;
  using autodiff::detail::at;
  using autodiff::detail::derivative;
  using autodiff::detail::wrt;
  auto f = [](dual x, dual y) -> dual {
    return x * y + sin(x) + y * y + exp(x + y);
  };
  dual x = 1.3, y = 0.7;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(y);
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
  double xv = 1.0, yv = 0.5, zv = 1.7, wv = W0;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  benchmark::DoNotOptimize(zv);
  benchmark::DoNotOptimize(wv);
  auto x = PV(xv, 'x');
  auto y = PV(yv, 'y');
  auto z = PV(zv, 'z');
  auto w = PV(wv, 'w');
  auto expr = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
  run_our_reverse(state, expr);
}
BENCHMARK(BM_Ours_Reverse_F4);

static void BM_AD_Forward_F4(benchmark::State &state) {
  using autodiff::dual;
  using autodiff::detail::at;
  using autodiff::detail::derivative;
  using autodiff::detail::wrt;
  auto f = [](dual x, dual y, dual z, dual w) -> dual {
    return (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
  };
  dual x = 1.0, y = 0.5, z = 1.7, w = W0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(y);
    benchmark::DoNotOptimize(z);
    benchmark::DoNotOptimize(w);
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

// ===========================================================================
// Tutorial T1  f(x) = 1 + x + x² + 1/x + log(x)   at x = 2.0
// ===========================================================================

static void BM_Ours_Forward_T1(benchmark::State &state) {
  using D = Dual<double>;
  Variable<D, 'x'> x{D{2.0}};
  auto expr = 1.0 + x + x * x + 1.0 / x + log(x);
  run_our_forward(state, expr, std::array{2.0});
}
BENCHMARK(BM_Ours_Forward_T1);

static void BM_Ours_Reverse_T1(benchmark::State &state) {
  double xv = 2.0;
  benchmark::DoNotOptimize(xv);
  auto x = PV(xv, 'x');
  auto expr = PC(1.0) + x + x * x + PC(1.0) / x + log(x);
  using Syms = boost::mp11::mp_list<std::integral_constant<char, 'x'>>;
  for (auto _ : state) {
    benchmark::DoNotOptimize(xv);
    expr.update(Syms{}, std::array{xv});
    auto g = reverse_mode_grad(expr);
    benchmark::DoNotOptimize(g);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Ours_Reverse_T1);

static void BM_AD_Forward_T1(benchmark::State &state) {
  using autodiff::dual;
  using autodiff::detail::at;
  using autodiff::detail::derivative;
  using autodiff::detail::wrt;
  auto f = [](dual x) -> dual { return 1.0 + x + x * x + 1.0 / x + log(x); };
  dual x = 2.0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    double dx = derivative(f, wrt(x), at(x));
    benchmark::DoNotOptimize(dx);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_AD_Forward_T1);

static void BM_AD_Reverse_T1(benchmark::State &state) {
  using autodiff::var;
  using autodiff::reverse::detail::derivatives;
  using autodiff::reverse::detail::wrt;
  var x = 2.0;
  for (auto _ : state) {
    var u = 1.0 + x + x * x + 1.0 / x + log(x);
    auto [dx] = derivatives(u, wrt(x));
    benchmark::DoNotOptimize(dx);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_AD_Reverse_T1);

// ===========================================================================
// Tutorial T_Multi3  f(x,y,z) = 1+x+y+z+xy+yz+xz+xyz+exp(x/y+y/z)
//   at (1.0, 2.0, 3.0)
// ===========================================================================

static void BM_Ours_Forward_TMulti3(benchmark::State &state) {
  using D = Dual<double>;
  Variable<D, 'x'> x{D{1.0}};
  Variable<D, 'y'> y{D{2.0}};
  Variable<D, 'z'> z{D{3.0}};
  auto expr =
      1.0 + x + y + z + x * y + y * z + x * z + x * y * z + exp(x / y + y / z);
  run_our_forward(state, expr, std::array{1.0, 2.0, 3.0});
}
BENCHMARK(BM_Ours_Forward_TMulti3);

static void BM_Ours_Reverse_TMulti3(benchmark::State &state) {
  double xv = 1.0, yv = 2.0, zv = 3.0;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  benchmark::DoNotOptimize(zv);
  auto x = PV(xv, 'x');
  auto y = PV(yv, 'y');
  auto z = PV(zv, 'z');
  auto expr = PC(1.0) + x + y + z + x * y + y * z + x * z + x * y * z +
              exp(x / y + y / z);
  run_our_reverse(state, expr);
}
BENCHMARK(BM_Ours_Reverse_TMulti3);

static void BM_AD_Forward_TMulti3(benchmark::State &state) {
  using autodiff::dual;
  using autodiff::detail::at;
  using autodiff::detail::derivative;
  using autodiff::detail::wrt;
  auto f = [](dual x, dual y, dual z) -> dual {
    return 1.0 + x + y + z + x * y + y * z + x * z + x * y * z +
           exp(x / y + y / z);
  };
  dual x = 1.0, y = 2.0, z = 3.0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(y);
    benchmark::DoNotOptimize(z);
    double dx = derivative(f, wrt(x), at(x, y, z));
    double dy = derivative(f, wrt(y), at(x, y, z));
    double dz = derivative(f, wrt(z), at(x, y, z));
    benchmark::DoNotOptimize(dx);
    benchmark::DoNotOptimize(dy);
    benchmark::DoNotOptimize(dz);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_AD_Forward_TMulti3);

static void BM_AD_Reverse_TMulti3(benchmark::State &state) {
  using autodiff::var;
  using autodiff::reverse::detail::derivatives;
  using autodiff::reverse::detail::wrt;
  var x = 1.0, y = 2.0, z = 3.0;
  for (auto _ : state) {
    var u = 1.0 + x + y + z + x * y + y * z + x * z + x * y * z +
            exp(x / y + y / z);
    auto [dx, dy, dz] = derivatives(u, wrt(x, y, z));
    benchmark::DoNotOptimize(dx);
    benchmark::DoNotOptimize(dy);
    benchmark::DoNotOptimize(dz);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_AD_Reverse_TMulti3);

// ===========================================================================
// Tutorial T_Grad2  f(x,y) = sin(x)*cos(y) + exp(x*y)   at (1.0, 0.5)
// ===========================================================================

static void BM_Ours_Forward_TGrad2(benchmark::State &state) {
  using D = Dual<double>;
  Variable<D, 'x'> x{D{1.0}};
  Variable<D, 'y'> y{D{0.5}};
  auto expr = sin(x) * cos(y) + exp(x * y);
  run_our_forward(state, expr, std::array{1.0, 0.5});
}
BENCHMARK(BM_Ours_Forward_TGrad2);

static void BM_Ours_Reverse_TGrad2(benchmark::State &state) {
  double xv = 1.0, yv = 0.5;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  auto x = PV(xv, 'x');
  auto y = PV(yv, 'y');
  auto expr = sin(x) * cos(y) + exp(x * y);
  run_our_reverse(state, expr);
}
BENCHMARK(BM_Ours_Reverse_TGrad2);

static void BM_AD_Forward_TGrad2(benchmark::State &state) {
  using autodiff::dual;
  using autodiff::detail::at;
  using autodiff::detail::derivative;
  using autodiff::detail::wrt;
  auto f = [](dual x, dual y) -> dual { return sin(x) * cos(y) + exp(x * y); };
  dual x = 1.0, y = 0.5;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(y);
    double dx = derivative(f, wrt(x), at(x, y));
    double dy = derivative(f, wrt(y), at(x, y));
    benchmark::DoNotOptimize(dx);
    benchmark::DoNotOptimize(dy);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_AD_Forward_TGrad2);

static void BM_AD_Reverse_TGrad2(benchmark::State &state) {
  using autodiff::var;
  using autodiff::reverse::detail::derivatives;
  using autodiff::reverse::detail::wrt;
  var x = 1.0, y = 0.5;
  for (auto _ : state) {
    var u = sin(x) * cos(y) + exp(x * y);
    auto [dx, dy] = derivatives(u, wrt(x, y));
    benchmark::DoNotOptimize(dx);
    benchmark::DoNotOptimize(dy);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_AD_Reverse_TGrad2);

// ===========================================================================
// Tutorial T_4th  f(x) = sin(x) — 4th-order derivative at x = π/4
// ===========================================================================

static const double T4_X0 = std::numbers::pi_v<double> / 4.0;

static void BM_Ours_Forward_T4th(benchmark::State &state) {
  double x0 = T4_X0;
  auto x = PV(x0, 'x');
  auto expr = sin(x);
  for (auto _ : state) {
    auto vals = std::array{T4_X0};
    benchmark::DoNotOptimize(vals);
    auto t4 = derivative_tensor<4>(expr, vals);
    benchmark::DoNotOptimize(t4);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Ours_Forward_T4th);

static void BM_AD_Forward_T4th(benchmark::State &state) {
  using autodiff::dual4th;
  using autodiff::detail::at;
  using autodiff::detail::derivative;
  using autodiff::detail::wrt;
  auto f = [](dual4th x) -> dual4th { return sin(x); };
  dual4th x = T4_X0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    double d4 = derivative(f, wrt(x, x, x, x), at(x));
    benchmark::DoNotOptimize(d4);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_AD_Forward_T4th);

static void BM_Ours_Taylor_T4th(benchmark::State &state) {
  double x0 = T4_X0;
  benchmark::DoNotOptimize(x0);
  auto x = PV(x0, 'x');
  auto expr = sin(x);
  for (auto _ : state) {
    benchmark::DoNotOptimize(x0);
    double d4 = univariate_derivative<4>(expr, x0);
    benchmark::DoNotOptimize(d4);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Ours_Taylor_T4th);

// ===========================================================================
// Tutorial T_Hess  f(x,y) = x² + xy + y²  — Hessian at (2.0, 3.0)
// ===========================================================================

static void BM_Ours_Forward_THess(benchmark::State &state) {
  double xv = 2.0, yv = 3.0;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  auto x = PV(xv, 'x');
  auto y = PV(yv, 'y');
  auto expr = x * x + x * y + y * y;
  for (auto _ : state) {
    auto vals = std::array{xv, yv};
    benchmark::DoNotOptimize(vals);
    auto H = derivative_tensor<2>(expr, vals);
    benchmark::DoNotOptimize(H);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Ours_Forward_THess);

static void BM_Ours_Reverse_THess(benchmark::State &state) {
  using D = Dual<double>;
  double xv = 2.0, yv = 3.0;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  Variable<D, 'x'> x{D{xv}};
  Variable<D, 'y'> y{D{yv}};
  auto expr = x * x + x * y + y * y;
  for (auto _ : state) {
    auto vals = std::array{xv, yv};
    benchmark::DoNotOptimize(vals);
    auto H = reverse_mode_hess(expr, vals);
    benchmark::DoNotOptimize(H);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Ours_Reverse_THess);

static void BM_AD_Forward_THess(benchmark::State &state) {
  using autodiff::dual2nd;
  using autodiff::detail::at;
  using autodiff::detail::derivative;
  using autodiff::detail::wrt;
  auto f = [](dual2nd x, dual2nd y) -> dual2nd {
    return x * x + x * y + y * y;
  };
  dual2nd x = 2.0, y = 3.0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(y);
    double hxx = derivative(f, wrt(x, x), at(x, y));
    double hxy = derivative(f, wrt(x, y), at(x, y));
    double hyy = derivative(f, wrt(y, y), at(x, y));
    benchmark::DoNotOptimize(hxx);
    benchmark::DoNotOptimize(hxy);
    benchmark::DoNotOptimize(hyy);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_AD_Forward_THess);

// ===========================================================================
// Tutorial T_Dir  f(x,y) = exp(x)*sin(y) — directional derivative
//   along u = (1/√2, 1/√2) at (1.0, 0.5)
// ===========================================================================

static const double DIR_UXY = 1.0 / std::numbers::sqrt2_v<double>;

static void BM_Ours_Forward_TDir(benchmark::State &state) {
  // Seed dual parts directly with direction components — single evaluation
  using D = Dual<double>;
  double xv = 1.0, yv = 0.5;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  Variable<D, 'x'> x{D{xv, DIR_UXY}};
  Variable<D, 'y'> y{D{yv, DIR_UXY}};
  auto expr = exp(x) * sin(y);
  for (auto _ : state) {
    auto val = expr.eval();
    double dir = val.template get<1>();
    benchmark::DoNotOptimize(dir);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Ours_Forward_TDir);

static void BM_Ours_Reverse_TDir(benchmark::State &state) {
  double xv = 1.0, yv = 0.5;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  auto x = PV(xv, 'x');
  auto y = PV(yv, 'y');
  auto expr = exp(x) * sin(y);
  for (auto _ : state) {
    auto g = reverse_mode_grad(expr);
    double dir = g[0] * DIR_UXY + g[1] * DIR_UXY;
    benchmark::DoNotOptimize(dir);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Ours_Reverse_TDir);

static void BM_AD_Forward_TDir(benchmark::State &state) {
  using autodiff::dual;
  using autodiff::detail::at;
  using autodiff::detail::derivative;
  using autodiff::detail::wrt;
  auto f = [](dual x, dual y) -> dual { return exp(x) * sin(y); };
  dual x = 1.0, y = 0.5;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(y);
    double dx = derivative(f, wrt(x), at(x, y));
    double dy = derivative(f, wrt(y), at(x, y));
    double dir = dx * DIR_UXY + dy * DIR_UXY;
    benchmark::DoNotOptimize(dir);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_AD_Forward_TDir);

static void BM_AD_Reverse_TDir(benchmark::State &state) {
  using autodiff::var;
  using autodiff::reverse::detail::derivatives;
  using autodiff::reverse::detail::wrt;
  var x = 1.0, y = 0.5;
  for (auto _ : state) {
    var u = exp(x) * sin(y);
    auto [dx, dy] = derivatives(u, wrt(x, y));
    double dir = dx * DIR_UXY + dy * DIR_UXY;
    benchmark::DoNotOptimize(dir);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_AD_Reverse_TDir);

BENCHMARK_MAIN();
