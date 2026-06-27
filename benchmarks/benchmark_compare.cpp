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
#include "seeded_energy.hpp"
#include "values.hpp"
#include "vforward_driver.hpp"

#include <array>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <string_view>
#include <tuple>

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

// ---------------------------------------------------------------------------
// Reverse-mode helpers — two honest framings, both with inputs made opaque every
// iteration (DoNotOptimize) so neither side can hoist the result out of the loop:
//
//   Scratch — build the graph + backward, every iteration. The realistic
//             "gradient of a freshly-formed expression" call. Ours rebuilds a
//             stack aggregate (free, fully inlined); autodiff rebuilds its heap
//             shared_ptr tape (one make_shared per operation).
//   Reuse   — graph built once; only re-seed + backward is timed. Ours updates
//             leaves in place (Variable::update) then backward; autodiff updates
//             leaves, recomputes forward values (u.update(), no alloc), then
//             derivatives() (no alloc). Isolates the per-node backward cost.
//
// The autodiff Scratch-minus-Reuse delta is exactly the per-call tape-allocation
// cost; the Ours-Reuse vs AD-Reuse gap is the structural (no-alloc, no-vtable,
// inlined-recursion) residual.
// ---------------------------------------------------------------------------

// Ours, Scratch: `build` maps N doubles to an expression; rebuilt each iteration.
template <std::size_t N, typename Builder>
static void run_ours_reverse_scratch(benchmark::State &state,
                                     std::array<double, N> x0, Builder build) {
  auto xs = x0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(xs);
    auto expr = std::apply(build, xs);
    auto g = reverse_mode_grad(expr);
    benchmark::DoNotOptimize(g);
    benchmark::ClobberMemory();
  }
}

// Ours, Reuse: expr built once by the caller; leaves re-seeded in place.
// `vals` MUST be in canonical (alphabetical) symbol order — build it at the call
// site with make_values(named<...>(...)) so the order is explicit and checked.
template <CExpression Expr, std::size_t N>
static void run_ours_reverse_reuse(benchmark::State &state, Expr &expr,
                                   std::array<double, N> vals) {
  using Syms = extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  auto v = vals;
  for (auto _ : state) {
    benchmark::DoNotOptimize(v);
    expr.update(Syms{}, v);
    auto g = reverse_mode_grad(expr);
    benchmark::DoNotOptimize(g);
    benchmark::ClobberMemory();
  }
}

// autodiff, Scratch: leaves updated, then the var tape is rebuilt every iteration.
template <std::size_t N, typename LeavesTuple, typename MakeU>
static void run_ad_reverse_scratch(benchmark::State &state,
                                   std::array<double, N> x0, LeavesTuple leaves,
                                   MakeU make_u) {
  auto xs = x0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(xs);
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (std::get<I>(leaves).update(xs[I]), ...);
    }(std::make_index_sequence<N>{});
    autodiff::var u = make_u(leaves); // tape rebuilt here (heap nodes)
    auto g = std::apply(
        [&](auto &...ls) {
          return autodiff::reverse::detail::derivatives(
              u, autodiff::reverse::detail::wrt(ls...));
        },
        leaves);
    benchmark::DoNotOptimize(g);
    benchmark::ClobberMemory();
  }
}

// autodiff, Reuse: var tape built once; only leaf-update + forward recompute +
// backward is timed (no rebuild, no allocation).
template <std::size_t N, typename LeavesTuple, typename MakeU>
static void run_ad_reverse_reuse(benchmark::State &state,
                                 std::array<double, N> x0, LeavesTuple leaves,
                                 MakeU make_u) {
  autodiff::var u = make_u(leaves); // built ONCE
  auto xs = x0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(xs);
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (std::get<I>(leaves).update(xs[I]), ...);
    }(std::make_index_sequence<N>{});
    u.update(); // recompute forward values through the reused graph (no alloc)
    auto g = std::apply(
        [&](auto &...ls) {
          return autodiff::reverse::detail::derivatives(
              u, autodiff::reverse::detail::wrt(ls...));
        },
        leaves);
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

static void BM_Ours_Reverse_F1_Scratch(benchmark::State &state) {
  run_ours_reverse_scratch<1>(state, {1.25}, [](double xv) {
    auto x = PV(xv, "x");
    return exp(x) * sin(x) + x * x * x + 2.0 * x;
  });
}
BENCHMARK(BM_Ours_Reverse_F1_Scratch);

static void BM_Ours_Reverse_F1_Reuse(benchmark::State &state) {
  auto x = PV(1.25, "x");
  auto expr = exp(x) * sin(x) + x * x * x + 2.0 * x;
  run_ours_reverse_reuse(state, expr,
                         make_values<decltype(expr)>(named<"x">(1.25)));
}
BENCHMARK(BM_Ours_Reverse_F1_Reuse);

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

static auto make_u_F1 = [](auto &L) {
  auto &x = std::get<0>(L);
  return exp(x) * sin(x) + x * x * x + 2.0 * x;
};
// NOTE: autodiff's `var` copy ctor wraps the source in a *dependent* node (and
// it has no move ctor), so a `var` stored in a tuple becomes un-updatable. The
// leaves must be named locals, passed as a tuple of references via std::tie.
static void BM_AD_Reverse_F1_Scratch(benchmark::State &state) {
  autodiff::var x = 1.25;
  run_ad_reverse_scratch<1>(state, {1.25}, std::tie(x), make_u_F1);
}
BENCHMARK(BM_AD_Reverse_F1_Scratch);

static void BM_AD_Reverse_F1_Reuse(benchmark::State &state) {
  autodiff::var x = 1.25;
  run_ad_reverse_reuse<1>(state, {1.25}, std::tie(x), make_u_F1);
}
BENCHMARK(BM_AD_Reverse_F1_Reuse);

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

static void BM_Ours_Reverse_F2_Scratch(benchmark::State &state) {
  run_ours_reverse_scratch<2>(state, {1.3, 0.7}, [](double xv, double yv) {
    auto x = PV(xv, "x");
    auto y = PV(yv, "y");
    return x * y + sin(x) + y * y + exp(x + y);
  });
}
BENCHMARK(BM_Ours_Reverse_F2_Scratch);

static void BM_Ours_Reverse_F2_Reuse(benchmark::State &state) {
  auto x = PV(1.3, "x");
  auto y = PV(0.7, "y");
  auto expr = x * y + sin(x) + y * y + exp(x + y);
  run_ours_reverse_reuse(
      state, expr,
      make_values<decltype(expr)>(named<"x">(1.3), named<"y">(0.7)));
}
BENCHMARK(BM_Ours_Reverse_F2_Reuse);

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

static auto make_u_F2 = [](auto &L) {
  auto &x = std::get<0>(L);
  auto &y = std::get<1>(L);
  return x * y + sin(x) + y * y + exp(x + y);
};
static void BM_AD_Reverse_F2_Scratch(benchmark::State &state) {
  autodiff::var x = 1.3, y = 0.7;
  run_ad_reverse_scratch<2>(state, {1.3, 0.7}, std::tie(x, y), make_u_F2);
}
BENCHMARK(BM_AD_Reverse_F2_Scratch);

static void BM_AD_Reverse_F2_Reuse(benchmark::State &state) {
  autodiff::var x = 1.3, y = 0.7;
  run_ad_reverse_reuse<2>(state, {1.3, 0.7}, std::tie(x, y), make_u_F2);
}
BENCHMARK(BM_AD_Reverse_F2_Reuse);

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

static void BM_Ours_Reverse_F4_Scratch(benchmark::State &state) {
  run_ours_reverse_scratch<4>(
      state, {1.0, 0.5, 1.7, W0},
      [](double xv, double yv, double zv, double wv) {
        auto x = PV(xv, "x");
        auto y = PV(yv, "y");
        auto z = PV(zv, "z");
        auto w = PV(wv, "w");
        return (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
      });
}
BENCHMARK(BM_Ours_Reverse_F4_Scratch);

static void BM_Ours_Reverse_F4_Reuse(benchmark::State &state) {
  double wv = W0; // W0 is `const double`; PV(decltype(x)) must be non-const
  auto x = PV(1.0, "x");
  auto y = PV(0.5, "y");
  auto z = PV(1.7, "z");
  auto w = PV(wv, "w");
  auto expr = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
  // Symbols sort to {w,x,y,z}; make_values binds by name, not position.
  run_ours_reverse_reuse(state, expr,
                         make_values<decltype(expr)>(named<"x">(1.0),
                                                     named<"y">(0.5),
                                                     named<"z">(1.7),
                                                     named<"w">(W0)));
}
BENCHMARK(BM_Ours_Reverse_F4_Reuse);

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

static auto make_u_F4 = [](auto &L) {
  auto &x = std::get<0>(L);
  auto &y = std::get<1>(L);
  auto &z = std::get<2>(L);
  auto &w = std::get<3>(L);
  return (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
};
static void BM_AD_Reverse_F4_Scratch(benchmark::State &state) {
  autodiff::var x = 1.0, y = 0.5, z = 1.7, w = W0;
  run_ad_reverse_scratch<4>(state, {1.0, 0.5, 1.7, W0}, std::tie(x, y, z, w),
                            make_u_F4);
}
BENCHMARK(BM_AD_Reverse_F4_Scratch);

static void BM_AD_Reverse_F4_Reuse(benchmark::State &state) {
  autodiff::var x = 1.0, y = 0.5, z = 1.7, w = W0;
  run_ad_reverse_reuse<4>(state, {1.0, 0.5, 1.7, W0}, std::tie(x, y, z, w),
                          make_u_F4);
}
BENCHMARK(BM_AD_Reverse_F4_Reuse);

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

static void BM_Ours_Reverse_T1_Scratch(benchmark::State &state) {
  run_ours_reverse_scratch<1>(state, {2.0}, [](double xv) {
    auto x = PV(xv, "x");
    return PC(1.0) + x + x * x + PC(1.0) / x + log(x);
  });
}
BENCHMARK(BM_Ours_Reverse_T1_Scratch);

static void BM_Ours_Reverse_T1_Reuse(benchmark::State &state) {
  auto x = PV(2.0, "x");
  auto expr = PC(1.0) + x + x * x + PC(1.0) / x + log(x);
  run_ours_reverse_reuse(state, expr,
                         make_values<decltype(expr)>(named<"x">(2.0)));
}
BENCHMARK(BM_Ours_Reverse_T1_Reuse);

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

static auto make_u_T1 = [](auto &L) {
  auto &x = std::get<0>(L);
  return 1.0 + x + x * x + 1.0 / x + log(x);
};
static void BM_AD_Reverse_T1_Scratch(benchmark::State &state) {
  autodiff::var x = 2.0;
  run_ad_reverse_scratch<1>(state, {2.0}, std::tie(x), make_u_T1);
}
BENCHMARK(BM_AD_Reverse_T1_Scratch);

static void BM_AD_Reverse_T1_Reuse(benchmark::State &state) {
  autodiff::var x = 2.0;
  run_ad_reverse_reuse<1>(state, {2.0}, std::tie(x), make_u_T1);
}
BENCHMARK(BM_AD_Reverse_T1_Reuse);

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

static void BM_Ours_Reverse_TMulti3_Scratch(benchmark::State &state) {
  run_ours_reverse_scratch<3>(
      state, {1.0, 2.0, 3.0}, [](double xv, double yv, double zv) {
        auto x = PV(xv, "x");
        auto y = PV(yv, "y");
        auto z = PV(zv, "z");
        return PC(1.0) + x + y + z + x * y + y * z + x * z + x * y * z +
               exp(x / y + y / z);
      });
}
BENCHMARK(BM_Ours_Reverse_TMulti3_Scratch);

static void BM_Ours_Reverse_TMulti3_Reuse(benchmark::State &state) {
  auto x = PV(1.0, "x");
  auto y = PV(2.0, "y");
  auto z = PV(3.0, "z");
  auto expr = PC(1.0) + x + y + z + x * y + y * z + x * z + x * y * z +
              exp(x / y + y / z);
  run_ours_reverse_reuse(state, expr,
                         make_values<decltype(expr)>(
                             named<"x">(1.0), named<"y">(2.0), named<"z">(3.0)));
}
BENCHMARK(BM_Ours_Reverse_TMulti3_Reuse);

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

static auto make_u_TMulti3 = [](auto &L) {
  auto &x = std::get<0>(L);
  auto &y = std::get<1>(L);
  auto &z = std::get<2>(L);
  return 1.0 + x + y + z + x * y + y * z + x * z + x * y * z +
         exp(x / y + y / z);
};
static void BM_AD_Reverse_TMulti3_Scratch(benchmark::State &state) {
  autodiff::var x = 1.0, y = 2.0, z = 3.0;
  run_ad_reverse_scratch<3>(state, {1.0, 2.0, 3.0}, std::tie(x, y, z),
                            make_u_TMulti3);
}
BENCHMARK(BM_AD_Reverse_TMulti3_Scratch);

static void BM_AD_Reverse_TMulti3_Reuse(benchmark::State &state) {
  autodiff::var x = 1.0, y = 2.0, z = 3.0;
  run_ad_reverse_reuse<3>(state, {1.0, 2.0, 3.0}, std::tie(x, y, z),
                          make_u_TMulti3);
}
BENCHMARK(BM_AD_Reverse_TMulti3_Reuse);

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

static void BM_Ours_Reverse_TGrad2_Scratch(benchmark::State &state) {
  run_ours_reverse_scratch<2>(state, {1.0, 0.5}, [](double xv, double yv) {
    auto x = PV(xv, "x");
    auto y = PV(yv, "y");
    return sin(x) * cos(y) + exp(x * y);
  });
}
BENCHMARK(BM_Ours_Reverse_TGrad2_Scratch);

static void BM_Ours_Reverse_TGrad2_Reuse(benchmark::State &state) {
  auto x = PV(1.0, "x");
  auto y = PV(0.5, "y");
  auto expr = sin(x) * cos(y) + exp(x * y);
  run_ours_reverse_reuse(
      state, expr,
      make_values<decltype(expr)>(named<"x">(1.0), named<"y">(0.5)));
}
BENCHMARK(BM_Ours_Reverse_TGrad2_Reuse);

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

static auto make_u_TGrad2 = [](auto &L) {
  auto &x = std::get<0>(L);
  auto &y = std::get<1>(L);
  return sin(x) * cos(y) + exp(x * y);
};
static void BM_AD_Reverse_TGrad2_Scratch(benchmark::State &state) {
  autodiff::var x = 1.0, y = 0.5;
  run_ad_reverse_scratch<2>(state, {1.0, 0.5}, std::tie(x, y), make_u_TGrad2);
}
BENCHMARK(BM_AD_Reverse_TGrad2_Scratch);

static void BM_AD_Reverse_TGrad2_Reuse(benchmark::State &state) {
  autodiff::var x = 1.0, y = 0.5;
  run_ad_reverse_reuse<2>(state, {1.0, 0.5}, std::tie(x, y), make_u_TGrad2);
}
BENCHMARK(BM_AD_Reverse_TGrad2_Reuse);

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

// Reverse mode computes the full gradient; the directional derivative is a
// negligible dot product on top, so we time the gradient like the other cells.
static void BM_Ours_Reverse_TDir_Scratch(benchmark::State &state) {
  run_ours_reverse_scratch<2>(state, {1.0, 0.5}, [](double xv, double yv) {
    auto x = PV(xv, "x");
    auto y = PV(yv, "y");
    return exp(x) * sin(y);
  });
}
BENCHMARK(BM_Ours_Reverse_TDir_Scratch);

static void BM_Ours_Reverse_TDir_Reuse(benchmark::State &state) {
  auto x = PV(1.0, "x");
  auto y = PV(0.5, "y");
  auto expr = exp(x) * sin(y);
  run_ours_reverse_reuse(
      state, expr,
      make_values<decltype(expr)>(named<"x">(1.0), named<"y">(0.5)));
}
BENCHMARK(BM_Ours_Reverse_TDir_Reuse);

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

static auto make_u_TDir = [](auto &L) {
  auto &x = std::get<0>(L);
  auto &y = std::get<1>(L);
  return exp(x) * sin(y);
};
static void BM_AD_Reverse_TDir_Scratch(benchmark::State &state) {
  autodiff::var x = 1.0, y = 0.5;
  run_ad_reverse_scratch<2>(state, {1.0, 0.5}, std::tie(x, y), make_u_TDir);
}
BENCHMARK(BM_AD_Reverse_TDir_Scratch);

static void BM_AD_Reverse_TDir_Reuse(benchmark::State &state) {
  autodiff::var x = 1.0, y = 0.5;
  run_ad_reverse_reuse<2>(state, {1.0, 0.5}, std::tie(x, y), make_u_TDir);
}
BENCHMARK(BM_AD_Reverse_TDir_Reuse);

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

// Public router: handing the raw expression *graph* to diff::hessian() must
// auto-detect CExpression, bridge it internally, and pick the scalar driver —
// no seeded_energy() at the call site.  Should track the explicit Ours_Forward
// (scalar) numbers, not the slower Ours_VForward, proving the routing works
// end-to-end from the client's point of view.
template <std::size_t Nv, typename MakeExpr>
static void ours_hessian_expr(benchmark::State &state, MakeExpr make) {
  auto E = make();
  auto x = vf_point(Nv);
  const std::span<const double> xs{x.data(), x.size()};
  for (auto _ : state) {
    benchmark::DoNotOptimize(xs);
    auto H = diff::hessian(E, xs); // raw graph in, right driver out
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

// Public-router path (should match Ours_Forward above via the routing tag).
static void BM_Ours_Hessian_HessExpr4(benchmark::State &state) {
  ours_hessian_expr<4>(state, make_chain_expr4);
}
BENCHMARK(BM_Ours_Hessian_HessExpr4);
static void BM_Ours_Hessian_HessExpr8(benchmark::State &state) {
  ours_hessian_expr<8>(state, make_chain_expr8);
}
BENCHMARK(BM_Ours_Hessian_HessExpr8);

// ===========================================================================
// Reverse-mode GRADIENT over a compile-time expression graph (CExpression).
//
// Same sparse-chain energy as the HessExpr benchmarks, but the value type is
// plain double and we take a single reverse-mode gradient — one backward pass
// over the Variable/+/*/log/exp graph — instead of a Hessian. This is the regime
// where the structural advantage is starkest: our "graph" is a compile-time type
// the backward pass inlines into straight-line code, whereas autodiff builds the
// identical energy as a heap shared_ptr `var` tape (one node per operation) and
// walks it with virtual dispatch. Same Scratch/Reuse framing as the scalar
// reverse benchmarks above; symbols are zero-padded so canonical order == index
// order.
//   E(x) = Σ_k x_k·log(x_k) + Σ_k c_k·(x_k − x_{k+1})² + exp(x_0·x_{n-1})
// ===========================================================================

static const std::array<double, 4> GP4{0.27, 0.39, 0.51, 0.63};
static const std::array<double, 8> GP8{0.21666667, 0.28333333, 0.35, 0.41666667,
                                       0.48333333, 0.55,       0.61666667,
                                       0.68333333};

// Ours: double-valued chain graph (usable directly as the Scratch builder).
static auto grad_chain4(double x0, double x1, double x2, double x3) {
  auto a = PV(x0, "x00");
  auto b = PV(x1, "x01");
  auto c = PV(x2, "x02");
  auto d = PV(x3, "x03");
  return a * log(a) + b * log(b) + c * log(c) + d * log(d) +
         0.50 * (a - b) * (a - b) + 0.51 * (b - c) * (b - c) +
         0.52 * (c - d) * (c - d) + exp(a * d);
}
static auto grad_chain8(double x0, double x1, double x2, double x3, double x4,
                        double x5, double x6, double x7) {
  auto a = PV(x0, "x00");
  auto b = PV(x1, "x01");
  auto c = PV(x2, "x02");
  auto d = PV(x3, "x03");
  auto e = PV(x4, "x04");
  auto g = PV(x5, "x05");
  auto h = PV(x6, "x06");
  auto i = PV(x7, "x07");
  return a * log(a) + b * log(b) + c * log(c) + d * log(d) + e * log(e) +
         g * log(g) + h * log(h) + i * log(i) + 0.50 * (a - b) * (a - b) +
         0.51 * (b - c) * (b - c) + 0.52 * (c - d) * (c - d) +
         0.53 * (d - e) * (d - e) + 0.54 * (e - g) * (e - g) +
         0.55 * (g - h) * (g - h) + 0.56 * (h - i) * (h - i) + exp(a * i);
}

// autodiff: the identical energy built from a tuple of `var` leaf references.
static auto make_u_grad4 = [](auto &L) {
  auto &a = std::get<0>(L);
  auto &b = std::get<1>(L);
  auto &c = std::get<2>(L);
  auto &d = std::get<3>(L);
  return a * log(a) + b * log(b) + c * log(c) + d * log(d) +
         0.50 * (a - b) * (a - b) + 0.51 * (b - c) * (b - c) +
         0.52 * (c - d) * (c - d) + exp(a * d);
};
static auto make_u_grad8 = [](auto &L) {
  auto &a = std::get<0>(L);
  auto &b = std::get<1>(L);
  auto &c = std::get<2>(L);
  auto &d = std::get<3>(L);
  auto &e = std::get<4>(L);
  auto &g = std::get<5>(L);
  auto &h = std::get<6>(L);
  auto &i = std::get<7>(L);
  return a * log(a) + b * log(b) + c * log(c) + d * log(d) + e * log(e) +
         g * log(g) + h * log(h) + i * log(i) + 0.50 * (a - b) * (a - b) +
         0.51 * (b - c) * (b - c) + 0.52 * (c - d) * (c - d) +
         0.53 * (d - e) * (d - e) + 0.54 * (e - g) * (e - g) +
         0.55 * (g - h) * (g - h) + 0.56 * (h - i) * (h - i) + exp(a * i);
};

static void BM_Ours_ReverseGraph_G4_Scratch(benchmark::State &state) {
  run_ours_reverse_scratch<4>(state, GP4, grad_chain4);
}
BENCHMARK(BM_Ours_ReverseGraph_G4_Scratch);
static void BM_Ours_ReverseGraph_G4_Reuse(benchmark::State &state) {
  auto expr = grad_chain4(GP4[0], GP4[1], GP4[2], GP4[3]);
  run_ours_reverse_reuse(
      state, expr,
      make_values<decltype(expr)>(named<"x00">(GP4[0]), named<"x01">(GP4[1]),
                                  named<"x02">(GP4[2]), named<"x03">(GP4[3])));
}
BENCHMARK(BM_Ours_ReverseGraph_G4_Reuse);
static void BM_AD_ReverseGraph_G4_Scratch(benchmark::State &state) {
  autodiff::var a = GP4[0], b = GP4[1], c = GP4[2], d = GP4[3];
  run_ad_reverse_scratch<4>(state, GP4, std::tie(a, b, c, d), make_u_grad4);
}
BENCHMARK(BM_AD_ReverseGraph_G4_Scratch);
static void BM_AD_ReverseGraph_G4_Reuse(benchmark::State &state) {
  autodiff::var a = GP4[0], b = GP4[1], c = GP4[2], d = GP4[3];
  run_ad_reverse_reuse<4>(state, GP4, std::tie(a, b, c, d), make_u_grad4);
}
BENCHMARK(BM_AD_ReverseGraph_G4_Reuse);

static void BM_Ours_ReverseGraph_G8_Scratch(benchmark::State &state) {
  run_ours_reverse_scratch<8>(state, GP8, grad_chain8);
}
BENCHMARK(BM_Ours_ReverseGraph_G8_Scratch);
static void BM_Ours_ReverseGraph_G8_Reuse(benchmark::State &state) {
  auto expr = grad_chain8(GP8[0], GP8[1], GP8[2], GP8[3], GP8[4], GP8[5],
                          GP8[6], GP8[7]);
  run_ours_reverse_reuse(
      state, expr,
      make_values<decltype(expr)>(
          named<"x00">(GP8[0]), named<"x01">(GP8[1]), named<"x02">(GP8[2]),
          named<"x03">(GP8[3]), named<"x04">(GP8[4]), named<"x05">(GP8[5]),
          named<"x06">(GP8[6]), named<"x07">(GP8[7])));
}
BENCHMARK(BM_Ours_ReverseGraph_G8_Reuse);
static void BM_AD_ReverseGraph_G8_Scratch(benchmark::State &state) {
  autodiff::var a = GP8[0], b = GP8[1], c = GP8[2], d = GP8[3], e = GP8[4],
                g = GP8[5], h = GP8[6], i = GP8[7];
  run_ad_reverse_scratch<8>(state, GP8, std::tie(a, b, c, d, e, g, h, i),
                            make_u_grad8);
}
BENCHMARK(BM_AD_ReverseGraph_G8_Scratch);
static void BM_AD_ReverseGraph_G8_Reuse(benchmark::State &state) {
  autodiff::var a = GP8[0], b = GP8[1], c = GP8[2], d = GP8[3], e = GP8[4],
                g = GP8[5], h = GP8[6], i = GP8[7];
  run_ad_reverse_reuse<8>(state, GP8, std::tie(a, b, c, d, e, g, h, i),
                          make_u_grad8);
}
BENCHMARK(BM_AD_ReverseGraph_G8_Reuse);

// ---------------------------------------------------------------------------
// One-time correctness cross-check (runs at static-init, before any benchmark).
// Guards the canonical-vs-source symbol-order footgun (esp. F4, whose symbols
// sort to {w,x,y,z}) and the Reuse/update path. assert() is a no-op under
// NDEBUG, so this aborts manually to stay effective in Release builds.
// ---------------------------------------------------------------------------
namespace {

bool reverse_close(double a, double b) {
  const double d = std::fabs(a - b);
  const double m = std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
  return d <= 1e-9 * m;
}

void reverse_check(const char *what, double ours, double ad) {
  if (!reverse_close(ours, ad)) {
    std::fprintf(stderr,
                 "reverse cross-check FAILED: %s  ours=%.12g  autodiff=%.12g\n",
                 what, ours, ad);
    std::abort();
  }
}

// Ours reverse partial w.r.t. a named symbol (handles canonical ordering).
template <CExpression Expr>
double ours_partial(const Expr &expr, std::string_view sym) {
  const auto g = reverse_mode_grad(expr);
  const auto order = symbol_order<std::remove_cvref_t<Expr>>();
  for (std::size_t i = 0; i < order.size(); ++i)
    if (order[i] == sym)
      return g[i];
  return std::numeric_limits<double>::quiet_NaN();
}

[[maybe_unused]] const bool reverse_crosscheck_ran = [] {
  namespace ad = autodiff::reverse::detail;
  // F1: f(x) = exp(x)sin(x) + x^3 + 2x
  {
    auto x = PV(1.25, "x");
    auto e = exp(x) * sin(x) + x * x * x + 2.0 * x;
    autodiff::var X = 1.25;
    autodiff::var u = exp(X) * sin(X) + X * X * X + 2.0 * X;
    auto [dx] = ad::derivatives(u, ad::wrt(X));
    reverse_check("F1 d/dx", ours_partial(e, "x"), dx);
  }
  // F2: f(x,y) = xy + sin(x) + y^2 + exp(x+y)
  {
    auto x = PV(1.3, "x");
    auto y = PV(0.7, "y");
    auto e = x * y + sin(x) + y * y + exp(x + y);
    autodiff::var X = 1.3, Y = 0.7;
    autodiff::var u = X * Y + sin(X) + Y * Y + exp(X + Y);
    auto [dx, dy] = ad::derivatives(u, ad::wrt(X, Y));
    reverse_check("F2 d/dx", ours_partial(e, "x"), dx);
    reverse_check("F2 d/dy", ours_partial(e, "y"), dy);
  }
  // F4: symbols sort to {w,x,y,z} — the ordering trap.
  {
    double wv = W0;
    auto x = PV(1.0, "x");
    auto y = PV(0.5, "y");
    auto z = PV(1.7, "z");
    auto w = PV(wv, "w");
    auto e = (x + y) * (z - w) + exp(x * z) + sin(y * w) + x * y * z * w;
    autodiff::var X = 1.0, Y = 0.5, Z = 1.7, Wv = W0;
    autodiff::var u =
        (X + Y) * (Z - Wv) + exp(X * Z) + sin(Y * Wv) + X * Y * Z * Wv;
    auto [dx, dy, dz, dw] = ad::derivatives(u, ad::wrt(X, Y, Z, Wv));
    reverse_check("F4 d/dx", ours_partial(e, "x"), dx);
    reverse_check("F4 d/dy", ours_partial(e, "y"), dy);
    reverse_check("F4 d/dz", ours_partial(e, "z"), dz);
    reverse_check("F4 d/dw", ours_partial(e, "w"), dw);
  }
  // G4: reverse gradient over a compile-time expression graph vs autodiff var.
  {
    auto e = grad_chain4(GP4[0], GP4[1], GP4[2], GP4[3]);
    autodiff::var a = GP4[0], b = GP4[1], c = GP4[2], d = GP4[3];
    auto leaves = std::tie(a, b, c, d);
    autodiff::var u = make_u_grad4(leaves);
    auto [da, db, dc, dd] = ad::derivatives(u, ad::wrt(a, b, c, d));
    reverse_check("G4 d/dx00", ours_partial(e, "x00"), da);
    reverse_check("G4 d/dx01", ours_partial(e, "x01"), db);
    reverse_check("G4 d/dx02", ours_partial(e, "x02"), dc);
    reverse_check("G4 d/dx03", ours_partial(e, "x03"), dd);
  }
  return true;
}();

} // namespace

BENCHMARK_MAIN();
