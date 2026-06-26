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
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/reverse/var.hpp>

#include "../include/gradient.hpp"
#include "dual.hpp"
#include "gradient.hpp"
#include "values.hpp"
#include "vforward_driver.hpp"

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
  Variable<D, diff::FixedString{"x"}> x{D{1.25}};
  auto expr = exp(x) * sin(x) + x * x * x + 2.0 * x;
  run_our_forward(state, expr, std::array{1.25});
}
BENCHMARK(BM_Ours_Forward_F1);

static void BM_Ours_Reverse_F1(benchmark::State &state) {
  double xv = 1.25;
  benchmark::DoNotOptimize(xv);
  auto x = PV(xv, "x");
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
  Variable<D, diff::FixedString{"x"}> x{D{1.3}};
  Variable<D, diff::FixedString{"y"}> y{D{0.7}};
  auto expr = x * y + sin(x) + y * y + exp(x + y);
  run_our_forward(state, expr, std::array{1.3, 0.7});
}
BENCHMARK(BM_Ours_Forward_F2);

static void BM_Ours_Reverse_F2(benchmark::State &state) {
  double xv = 1.3, yv = 0.7;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  auto x = PV(xv, "x");
  auto y = PV(yv, "y");
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
  Variable<D, diff::FixedString{"x"}> x{D{1.0}};
  Variable<D, diff::FixedString{"y"}> y{D{0.5}};
  Variable<D, diff::FixedString{"z"}> z{D{1.7}};
  Variable<D, diff::FixedString{"w"}> w{D{W0}};
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
  auto x = PV(xv, "x");
  auto y = PV(yv, "y");
  auto z = PV(zv, "z");
  auto w = PV(wv, "w");
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
  Variable<D, diff::FixedString{"x"}> x{D{2.0}};
  auto expr = 1.0 + x + x * x + 1.0 / x + log(x);
  run_our_forward(state, expr, std::array{2.0});
}
BENCHMARK(BM_Ours_Forward_T1);

static void BM_Ours_Reverse_T1(benchmark::State &state) {
  double xv = 2.0;
  benchmark::DoNotOptimize(xv);
  auto x = PV(xv, "x");
  auto expr = PC(1.0) + x + x * x + PC(1.0) / x + log(x);
  using Syms = boost::mp11::mp_list<diff::symbol_type<diff::FixedString{"x"}>>;
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
  Variable<D, diff::FixedString{"x"}> x{D{1.0}};
  Variable<D, diff::FixedString{"y"}> y{D{2.0}};
  Variable<D, diff::FixedString{"z"}> z{D{3.0}};
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
  auto x = PV(xv, "x");
  auto y = PV(yv, "y");
  auto z = PV(zv, "z");
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
  Variable<D, diff::FixedString{"x"}> x{D{1.0}};
  Variable<D, diff::FixedString{"y"}> y{D{0.5}};
  auto expr = sin(x) * cos(y) + exp(x * y);
  run_our_forward(state, expr, std::array{1.0, 0.5});
}
BENCHMARK(BM_Ours_Forward_TGrad2);

static void BM_Ours_Reverse_TGrad2(benchmark::State &state) {
  double xv = 1.0, yv = 0.5;
  benchmark::DoNotOptimize(xv);
  benchmark::DoNotOptimize(yv);
  auto x = PV(xv, "x");
  auto y = PV(yv, "y");
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
  auto x = PV(x0, "x");
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
  auto x = PV(x0, "x");
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
  auto x = PV(xv, "x");
  auto y = PV(yv, "y");
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
  Variable<D, diff::FixedString{"x"}> x{D{xv}};
  Variable<D, diff::FixedString{"y"}> y{D{yv}};
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
  Variable<D, diff::FixedString{"x"}> x{D{xv, DIR_UXY}};
  Variable<D, diff::FixedString{"y"}> y{D{yv, DIR_UXY}};
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
  auto x = PV(xv, "x");
  auto y = PV(yv, "y");
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

// ===========================================================================
// Vector-valued dense Hessian  f: R^n -> R, full n x n Hessian, swept over n.
//
//   f(y) = Σ_i y_i·log(y_i)
//        + Σ_{i<j} c_ij · (y_i·y_j)/(1 + y_i)        c_ij = 0.1(i+1) - 0.05 j
//        + exp(y_0 · y_{n-1})
//
// This is the workload the new vector-forward driver targets: one templated
// energy, three ways to get its Hessian —
//   Ours_VForward — hessian_vforward(): O(n) sweeps, inner pack = identity
//   Ours_Forward  — hessian():          O(n^2) scalar forward-over-forward
//   AD_Forward    — autodiff::hessian(): Eigen VectorXdual2nd, O(n^2)
//
// n is swept up to kVForwardN (=32 by default) so the vector-forward path
// stays engaged rather than falling back to the scalar O(n^2) driver.
// ===========================================================================

// Single templated energy shared by every implementation — `y` is anything
// indexable (a raw Dual<...>* from our drivers, or an Eigen VectorXdual2nd).
template <typename Vec>
static auto vf_energy(const Vec &y, std::size_t n) {
  using std::exp, std::log; // ADL selects the dual / autodiff overloads
  using Scalar = std::remove_cvref_t<decltype(y[0])>;
  Scalar g{0};
  for (std::size_t i = 0; i < n; ++i) {
    g = g + y[i] * log(y[i]);
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double c =
          0.1 * static_cast<double>(i + 1) - 0.05 * static_cast<double>(j);
      g = g + c * (y[i] * y[j]) / (Scalar{1} + y[i]);
    }
  }
  g = g + exp(y[0] * y[n - 1]);
  return g;
}

static std::vector<double> vf_point(std::size_t n) {
  std::vector<double> x(n);
  for (std::size_t k = 0; k < n; ++k) {
    x[k] = 0.15 + 0.6 * (k + 1.0) / (n + 1.0);
  }
  return x;
}

static void BM_Ours_VForward_Hess(benchmark::State &state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  auto x = vf_point(n);
  const std::span<const double> xs{x.data(), x.size()};
  auto f = [n](const auto *dof) { return vf_energy(dof, n); };
  for (auto _ : state) {
    benchmark::DoNotOptimize(xs);
    auto H = diff::hessian_vforward(f, xs);
    benchmark::DoNotOptimize(H);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Ours_VForward_Hess)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

static void BM_Ours_Forward_Hess(benchmark::State &state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  auto x = vf_point(n);
  const std::span<const double> xs{x.data(), x.size()};
  auto f = [n](const auto *dof) { return vf_energy(dof, n); };
  for (auto _ : state) {
    benchmark::DoNotOptimize(xs);
    auto H = diff::detail::hessian_scalar(f, xs);
    benchmark::DoNotOptimize(H);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Ours_Forward_Hess)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

static void BM_AD_Forward_Hess(benchmark::State &state) {
  using autodiff::dual2nd;
  using autodiff::VectorXdual2nd;
  using autodiff::detail::at;
  using autodiff::detail::hessian;
  using autodiff::detail::wrt;
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  auto x0 = vf_point(n);
  VectorXdual2nd x(static_cast<Eigen::Index>(n));
  for (std::size_t k = 0; k < n; ++k) {
    x[static_cast<Eigen::Index>(k)] = x0[k];
  }
  auto f = [](const VectorXdual2nd &y) -> dual2nd {
    return vf_energy(y, static_cast<std::size_t>(y.size()));
  };
  dual2nd u;
  Eigen::VectorXd g;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    Eigen::MatrixXd H = hessian(f, wrt(x), at(x), u, g);
    benchmark::DoNotOptimize(H);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_AD_Forward_Hess)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

// ===========================================================================
// Sparse / cheap energy  f: R^n -> R, O(n) to evaluate, tridiagonal Hessian.
//
//   f(y) = Σ_i y_i·log(y_i)                    (separable -> diagonal)
//        + Σ_i c_i·(y_i - y_{i+1})^2           (nearest-neighbour -> tridiagonal)
//        + exp(y_0 · y_{n-1})                  (one dense corner)
//
// Unlike the dense quadratic above (O(n^2) to evaluate, which cancels the
// vector-forward sweep advantage), this energy costs O(n) per evaluation, so
// the O(n)-sweep vector-forward driver does O(n^2) lane-work vs the scalar
// driver's O(n^2) sweeps of O(n) work — same order, but the vector path's
// inner lane loops are contiguous and auto-vectorize.  This is the regime the
// vector-forward Hessian is built for.
// ===========================================================================

template <typename Vec>
static auto vf_energy_sparse(const Vec &y, std::size_t n) {
  using std::exp, std::log;
  using Scalar = std::remove_cvref_t<decltype(y[0])>;
  Scalar g{0};
  for (std::size_t i = 0; i < n; ++i) {
    g = g + y[i] * log(y[i]);
  }
  for (std::size_t i = 0; i + 1 < n; ++i) {
    const double c = 0.5 + 0.01 * static_cast<double>(i);
    const Scalar d = y[i] - y[i + 1];
    g = g + c * d * d;
  }
  g = g + exp(y[0] * y[n - 1]);
  return g;
}

static void BM_Ours_VForward_HessSparse(benchmark::State &state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  auto x = vf_point(n);
  const std::span<const double> xs{x.data(), x.size()};
  auto f = [n](const auto *dof) { return vf_energy_sparse(dof, n); };
  for (auto _ : state) {
    benchmark::DoNotOptimize(xs);
    auto H = diff::hessian_vforward(f, xs);
    benchmark::DoNotOptimize(H);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Ours_VForward_HessSparse)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

static void BM_Ours_Forward_HessSparse(benchmark::State &state) {
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  auto x = vf_point(n);
  const std::span<const double> xs{x.data(), x.size()};
  auto f = [n](const auto *dof) { return vf_energy_sparse(dof, n); };
  for (auto _ : state) {
    benchmark::DoNotOptimize(xs);
    auto H = diff::detail::hessian_scalar(f, xs);
    benchmark::DoNotOptimize(H);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_Ours_Forward_HessSparse)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

static void BM_AD_Forward_HessSparse(benchmark::State &state) {
  using autodiff::dual2nd;
  using autodiff::VectorXdual2nd;
  using autodiff::detail::at;
  using autodiff::detail::hessian;
  using autodiff::detail::wrt;
  const std::size_t n = static_cast<std::size_t>(state.range(0));
  auto x0 = vf_point(n);
  VectorXdual2nd x(static_cast<Eigen::Index>(n));
  for (std::size_t k = 0; k < n; ++k) {
    x[static_cast<Eigen::Index>(k)] = x0[k];
  }
  auto f = [](const VectorXdual2nd &y) -> dual2nd {
    return vf_energy_sparse(y, static_cast<std::size_t>(y.size()));
  };
  dual2nd u;
  Eigen::VectorXd g;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    Eigen::MatrixXd H = hessian(f, wrt(x), at(x), u, g);
    benchmark::DoNotOptimize(H);
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_AD_Forward_HessSparse)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

// ===========================================================================
// Expression-template energy  — same sparse-chain math, but the energy is now
// an actual compile-time expression *graph* (Variable / + / * / log / exp
// nodes) rather than a flat arithmetic loop.  This is the regime the
// vector-forward driver is built for: each Hessian row is one traversal of the
// graph, so O(n) sweeps vs the scalar driver's O(n^2) traversals.
//
//   E(x) = Σ_k x_k·log(x_k) + Σ_k c_k·(x_k - x_{k+1})^2 + exp(x_0·x_{n-1})
//          c_k = 0.5 + 0.01 k          (identical to vf_energy_sparse)
//
// The graph is bridged into the runtime driver via eval_seeded_as<T, Syms>:
// the driver hands us seeded dual dofs, we pack them in symbol order and
// traverse the graph once.  autodiff has no expression-template layer, so its
// side reuses vf_energy_sparse — the same closed form — for an honest compare.
//
// Labels are zero-padded ("x00","x01",...) so the symbol set, which the library
// sorts lexicographically, lands in index order.
// ===========================================================================

static auto make_chain_expr4() {
  using D = diff::Dual<double>;
  using diff::FixedString;
  diff::Variable<D, FixedString{"x00"}> a{D{1.0}};
  diff::Variable<D, FixedString{"x01"}> b{D{1.0}};
  diff::Variable<D, FixedString{"x02"}> c{D{1.0}};
  diff::Variable<D, FixedString{"x03"}> d{D{1.0}};
  return a * log(a) + b * log(b) + c * log(c) + d * log(d) +
         0.50 * (a - b) * (a - b) + 0.51 * (b - c) * (b - c) +
         0.52 * (c - d) * (c - d) + exp(a * d);
}

static auto make_chain_expr8() {
  using D = diff::Dual<double>;
  using diff::FixedString;
  diff::Variable<D, FixedString{"x00"}> a{D{1.0}};
  diff::Variable<D, FixedString{"x01"}> b{D{1.0}};
  diff::Variable<D, FixedString{"x02"}> c{D{1.0}};
  diff::Variable<D, FixedString{"x03"}> d{D{1.0}};
  diff::Variable<D, FixedString{"x04"}> e{D{1.0}};
  diff::Variable<D, FixedString{"x05"}> g{D{1.0}};
  diff::Variable<D, FixedString{"x06"}> h{D{1.0}};
  diff::Variable<D, FixedString{"x07"}> i{D{1.0}};
  return a * log(a) + b * log(b) + c * log(c) + d * log(d) + e * log(e) +
         g * log(g) + h * log(h) + i * log(i) + 0.50 * (a - b) * (a - b) +
         0.51 * (b - c) * (b - c) + 0.52 * (c - d) * (c - d) +
         0.53 * (d - e) * (d - e) + 0.54 * (e - g) * (e - g) +
         0.55 * (g - h) * (g - h) + 0.56 * (h - i) * (h - i) + exp(a * i);
}

// Build the runtime-driver energy lambda for an expression graph: pack the
// driver's seeded dofs (symbol order) and traverse the graph once.
template <std::size_t Nv, typename Expr>
static auto expr_energy(const Expr &E) {
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  return [&E](const auto *dof) {
    using T = std::remove_cvref_t<decltype(dof[0])>;
    std::array<T, Nv> s{};
    for (std::size_t k = 0; k < Nv; ++k) {
      s[k] = dof[k];
    }
    return E.template eval_seeded_as<T, Syms>(s);
  };
}

template <std::size_t Nv, typename MakeExpr>
static void ours_vforward_expr(benchmark::State &state, MakeExpr make) {
  auto E = make();
  auto f = expr_energy<Nv>(E);
  auto x = vf_point(Nv);
  const std::span<const double> xs{x.data(), x.size()};
  for (auto _ : state) {
    benchmark::DoNotOptimize(xs);
    auto H = diff::hessian_vforward(f, xs);
    benchmark::DoNotOptimize(H);
    benchmark::ClobberMemory();
  }
}

template <std::size_t Nv, typename MakeExpr>
static void ours_forward_expr(benchmark::State &state, MakeExpr make) {
  auto E = make();
  auto f = expr_energy<Nv>(E);
  auto x = vf_point(Nv);
  const std::span<const double> xs{x.data(), x.size()};
  for (auto _ : state) {
    benchmark::DoNotOptimize(xs);
    auto H = diff::detail::hessian_scalar(f, xs);
    benchmark::DoNotOptimize(H);
    benchmark::ClobberMemory();
  }
}

static void BM_Ours_VForward_HessExpr4(benchmark::State &state) {
  ours_vforward_expr<4>(state, make_chain_expr4);
}
BENCHMARK(BM_Ours_VForward_HessExpr4);
static void BM_Ours_Forward_HessExpr4(benchmark::State &state) {
  ours_forward_expr<4>(state, make_chain_expr4);
}
BENCHMARK(BM_Ours_Forward_HessExpr4);
// autodiff baseline for n=4/8 is BM_AD_Forward_HessSparse/{4,8} — identical
// closed form (autodiff has no expression-template layer to mirror).

static void BM_Ours_VForward_HessExpr8(benchmark::State &state) {
  ours_vforward_expr<8>(state, make_chain_expr8);
}
BENCHMARK(BM_Ours_VForward_HessExpr8);
static void BM_Ours_Forward_HessExpr8(benchmark::State &state) {
  ours_forward_expr<8>(state, make_chain_expr8);
}
BENCHMARK(BM_Ours_Forward_HessExpr8);

BENCHMARK_MAIN();
