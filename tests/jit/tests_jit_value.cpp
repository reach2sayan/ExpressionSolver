#include "ddx.hpp"
#include "jit/kernel.hpp"
#include "rt/bridge.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

// The interpreter in rt/interpret.hpp is the reference: it and the kernel walk
// the same graph, so a disagreement is a codegen bug.  Batch shape throughout.

namespace {
using ddx::rt::Builder;
using ddx::rt::Graph;

// No native target means no JIT to test; bring it up once and name the reason.
ddx::jit::Compiler &compiler() {
  static ddx::jit::result<ddx::jit::Compiler> c = ddx::jit::Compiler::create();
  EXPECT_TRUE(c.has_value()) << (c ? "" : c.error().detail);
  return *c;
}

// compile() answers with result<Kernel>; name LLVM's reason here rather than
// meeting an empty Kernel three lines later.
ddx::jit::Kernel must_compile(auto &&...args) {
  auto k = compiler().compile(static_cast<decltype(args) &&>(args)...);
  EXPECT_TRUE(k.has_value()) << (k ? std::string{} : k.error().detail);
  return k ? std::move(*k) : ddx::jit::Kernel{};
}

// Compile `build` and compare a batch against the interpreter.
void expect_matches_interpreter(auto build, std::size_t nvars,
                                std::size_t n = 64) {
  Builder<> b;
  std::vector<ddx::rt::RTExpression<>> vars;
  static constexpr const char *names[] = {"x", "y", "z", "w"};
  for (std::size_t i = 0; i < nvars; ++i) {
    vars.push_back(var(b, names[i]));
  }
  const auto root = build(b, vars);
  const auto graph = Graph<>::freeze(b, std::array{root.id(b)});
  const auto kernel = must_compile(graph);
  ASSERT_TRUE(static_cast<bool>(kernel));
  ASSERT_EQ(kernel.arity(), nvars);

  // Spread the sample points over a range every op in the suite accepts.
  std::vector<std::vector<double>> columns(nvars, std::vector<double>(n));
  for (std::size_t j = 0; j < nvars; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      columns[j][i] = 0.15 + 0.6 * static_cast<double>((i + 3 * j) % 11) / 11.0;
    }
  }
  std::vector<const double *> xs(nvars);
  for (std::size_t j = 0; j < nvars; ++j) {
    xs[j] = columns[j].data();
  }

  std::vector<double> got(n, std::numeric_limits<double>::quiet_NaN());
  double *const value_columns[]{got.data()};
  kernel(xs, value_columns, {}, {}, n);

  for (std::size_t i = 0; i < n; ++i) {
    std::vector<double> point(nvars);
    for (std::size_t j = 0; j < nvars; ++j) {
      point[j] = columns[j][i];
    }
    const double want = ddx::rt::evaluate(b, root.id(b), point);
    ASSERT_NEAR(got[i], want, 1e-12 * std::max(1.0, std::abs(want)))
        << "at point " << i;
  }
}

// A Compiler is the LLJIT and a Kernel points into code it owns: one owner,
// but it must move -- Equation keeps one in a static.
TEST(JitValue, CompilerIsMovableButNotCopyable) {
  static_assert(!std::is_copy_constructible_v<ddx::jit::Compiler>);
  static_assert(!std::is_copy_assignable_v<ddx::jit::Compiler>);
  static_assert(std::is_move_constructible_v<ddx::jit::Compiler>);
  static_assert(std::is_move_assignable_v<ddx::jit::Compiler>);
  SUCCEED();
}

TEST(JitValue, Arithmetic) {
  expect_matches_interpreter([](Builder<> &, auto &v) { return v[0] + v[1]; },
                             2);
  expect_matches_interpreter([](Builder<> &, auto &v) { return v[0] - v[1]; },
                             2);
  expect_matches_interpreter([](Builder<> &, auto &v) { return v[0] * v[1]; },
                             2);
  expect_matches_interpreter([](Builder<> &, auto &v) { return v[0] / v[1]; },
                             2);
  expect_matches_interpreter([](Builder<> &, auto &v) { return -v[0] * v[1]; },
                             2);
}

// Split across tests so a failure names the op that broke.
TEST(JitValue, IntrinsicBackedFunctions) {
  expect_matches_interpreter(
      [](Builder<> &, auto &v) { return sin(v[0]) + cos(v[1]); }, 2);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) { return tan(v[0]) * exp(v[1]); }, 2);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) { return log(v[0]) - log10(v[1]); }, 2);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) { return sqrt(v[0]) + asin(v[1]); }, 2);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) { return acos(v[0]) + atan(v[1]); }, 2);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) { return sinh(v[0]) + cosh(v[1]); }, 2);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) { return tanh(v[0]) * abs(v[1]); }, 2);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) { return pow(v[0], v[1]); }, 2);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) { return atan2(v[0], v[1]); }, 2);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) { return max(v[0], v[1]); }, 2);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) { return min(v[0], v[1]); }, 2);
}

// No LLVM intrinsic: these go out as libm calls, a different emitter path.
TEST(JitValue, LibmBackedFunctions) {
  expect_matches_interpreter([](Builder<> &, auto &v) { return cbrt(v[0]); },
                             1);
  expect_matches_interpreter([](Builder<> &, auto &v) { return asinh(v[0]); },
                             1);
  expect_matches_interpreter([](Builder<> &, auto &v) { return atanh(v[0]); },
                             1);
  expect_matches_interpreter([](Builder<> &, auto &v) { return erf(v[0]); }, 1);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) { return hypot(v[0], v[1]); }, 2);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) { return acosh(v[0] + 1.0); }, 1);
}

TEST(JitValue, MaxMinPropagateNaNSymmetrically) {
  // llvm.maximum/minimum propagate a NaN operand from either side, as
  // max_impl does; maxnum/minnum would return the other operand.
  Builder<> b;
  const auto root =
      max(var(b, "x"), var(b, "y")) + min(var(b, "x"), var(b, "y"));
  const auto kernel = must_compile(Graph<>::freeze(b, std::array{root.id(b)}));
  ASSERT_TRUE(static_cast<bool>(kernel));

  const double qnan = std::numeric_limits<double>::quiet_NaN();
  const std::array cx{qnan, 2.0, 1.0};
  const std::array cy{2.0, qnan, 3.0};
  const std::array<const double *, 2> xs{cx.data(), cy.data()};
  std::array<double, 3> got{};
  double *const value_columns[]{got.data()};
  kernel(xs, value_columns, {}, {}, got.size());

  EXPECT_TRUE(std::isnan(got[0]));
  EXPECT_TRUE(std::isnan(got[1]));
  EXPECT_DOUBLE_EQ(got[2], 4.0);
}

// IEEE 754-2019 maximum/minimum order -0 below +0, which is what llvm.maximum
// and llvm.minimum compute; the interpreter has to say the same, or 1/min(x, y)
// changes sign between the two.
TEST(JitValue, MaxMinSignedZeroTiesMatchTheInterpreter) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto graph = Graph<>::freeze(b, std::array{max(x, y).id(b), min(x, y).id(b)});
  const auto kernel = must_compile(graph);
  ASSERT_TRUE(static_cast<bool>(kernel));

  const std::array cx{-0.0, 0.0, -0.0, 0.0};
  const std::array cy{0.0, -0.0, -0.0, 0.0};
  const std::array<const double *, 2> xs{cx.data(), cy.data()};
  std::array<double, 4> got_max{};
  std::array<double, 4> got_min{};
  double *const value_columns[]{got_max.data(), got_min.data()};
  kernel(xs, value_columns, {}, {}, cx.size());

  for (std::size_t i = 0; i < cx.size(); ++i) {
    const auto want = ddx::rt::evaluate_all(b, std::array{cx[i], cy[i]});
    EXPECT_EQ(std::signbit(got_max[i]), std::signbit(want[graph.outputs()[0]]))
        << "max at " << i;
    EXPECT_EQ(std::signbit(got_min[i]), std::signbit(want[graph.outputs()[1]]))
        << "min at " << i;
  }
  EXPECT_FALSE(std::signbit(got_max[0])) << "max(-0, +0) is +0";
  EXPECT_TRUE(std::signbit(got_min[0])) << "min(-0, +0) is -0";
}

// Every width computes the same bits: a lane is an IEEE operation on its own,
// contraction happens per lane the same way it does for one, and a
// transcendental is the same scalar libm call per lane.  Checked over every op
// the emitter has, on a batch that is not a multiple of any width.
TEST(JitValue, EveryLaneWidthAgreesToTheBit) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto f = x / (y + 1.0) - x * y;
#define DDX_TEST_OP(fn, Op, label, ...) f = f + fn(x * 0.5);
  DDX_UNARY_MATH_TABLE(DDX_TEST_OP)
#undef DDX_TEST_OP
  f = f + pow(x, y) + atan2(x, y) + hypot(x, y) + abs(x - y) + max(x, y) +
      min(x, y) + sign(x - y);
  const auto graph = ddx::rt::GraphBuilder{b}.value(f).jacobian().build();

  constexpr std::size_t n = 13;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.2 + 0.05 * static_cast<double>(i);
    cy[i] = 0.9 - 0.03 * static_cast<double>(i);
  }
  const std::array<const double *, 2> xs{cx.data(), cy.data()};

  const auto run = [&](unsigned lanes) {
    ddx::jit::Options o;
    o.lanes = lanes;
    const auto kernel = must_compile(graph, o);
    std::vector<double> value(n), dx(n), dy(n);
    double *const values[]{value.data()};
    const std::array<double *, 2> partials{dx.data(), dy.data()};
    kernel(xs, values, partials, {}, n);
    return std::tuple{value, dx, dy};
  };

  const auto [v1, dx1, dy1] = run(1);
  for (const unsigned lanes : {2u, 4u, 8u, 0u}) {
    const auto [v, dx, dy] = run(lanes);
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_EQ(std::bit_cast<std::uint64_t>(v[i]),
                std::bit_cast<std::uint64_t>(v1[i]))
          << "value, lanes " << lanes << " at " << i;
      EXPECT_EQ(std::bit_cast<std::uint64_t>(dx[i]),
                std::bit_cast<std::uint64_t>(dx1[i]))
          << "d/dx, lanes " << lanes << " at " << i;
      EXPECT_EQ(std::bit_cast<std::uint64_t>(dy[i]),
                std::bit_cast<std::uint64_t>(dy1[i]))
          << "d/dy, lanes " << lanes << " at " << i;
    }
  }
}

// The codegen level rides on the module, so one JIT serves compiles at
// several.  1, 2 and 3 share an instruction selector and a register allocator
// and must agree to the bit -- which is what lets the default be the cheapest
// of them.  Level 0 takes FastISel, forms no FMAs, and is not held to it.
TEST(JitValue, CodegenLevelsOneToThreeAgreeToTheBit) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  // Multiply-add shapes throughout, so there is an FMA at every level that
  // forms one -- the thing the levels could have disagreed about.
  const auto f = x * y + x * x - y + exp(x * y) * (y + 1.0) + log(x) * y;
  const auto graph = ddx::rt::GraphBuilder{b}.value(f).jacobian().build();

  const std::array cx{0.25, 1.5, 2.75, 3.0};
  const std::array cy{1.25, 0.5, 2.0, 0.75};
  const std::array<const double *, 2> xs{cx.data(), cy.data()};

  const auto run = [&](unsigned level) {
    ddx::jit::Options o;
    o.codegen_level = level;
    const auto kernel = must_compile(graph, o);
    std::vector<double> value(cx.size()), dx(cx.size()), dy(cx.size());
    double *const values[]{value.data()};
    const std::array<double *, 2> partials{dx.data(), dy.data()};
    kernel(xs, values, partials, {}, cx.size());
    return std::tuple{value, dx, dy};
  };

  const auto [v1, dx1, dy1] = run(1);
  for (const unsigned level : {2u, 3u}) {
    const auto [v, dx, dy] = run(level);
    for (std::size_t i = 0; i < cx.size(); ++i) {
      EXPECT_EQ(std::bit_cast<std::uint64_t>(v[i]),
                std::bit_cast<std::uint64_t>(v1[i]))
          << "value, codegen " << level << " at " << i;
      EXPECT_EQ(std::bit_cast<std::uint64_t>(dx[i]),
                std::bit_cast<std::uint64_t>(dx1[i]))
          << "d/dx, codegen " << level << " at " << i;
      EXPECT_EQ(std::bit_cast<std::uint64_t>(dy[i]),
                std::bit_cast<std::uint64_t>(dy1[i]))
          << "d/dy, codegen " << level << " at " << i;
    }
  }

  // 0 is still a correct kernel, just not a bit-identical one.
  const auto [v0, dx0, dy0] = run(0);
  for (std::size_t i = 0; i < cx.size(); ++i) {
    EXPECT_NEAR(v0[i], v1[i], 1e-12 * std::max(1.0, std::abs(v1[i])));
  }
}

// Both vectorisers pack operations that were already independent, and neither
// is given `reassoc`, so neither may move a bit.  This is the gate on turning
// either on: a kernel that is 14% quicker and answers differently is not the
// same kernel.
TEST(JitValue, TheVectorisersDoNotMoveABit) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto f = x * log(x) + y * exp(x * y) + sqrt(x + y) + x * y * (x + y);
  const auto graph = ddx::rt::GraphBuilder{b}.value(f).jacobian().build();

  constexpr std::size_t n = 9;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.05 * static_cast<double>(i);
    cy[i] = 0.9 - 0.03 * static_cast<double>(i);
  }
  const std::array<const double *, 2> xs{cx.data(), cy.data()};

  const auto run = [&](ddx::jit::Options o) {
    const auto kernel = must_compile(graph, o);
    std::vector<double> value(n), dx(n), dy(n);
    double *const values[]{value.data()};
    const std::array<double *, 2> partials{dx.data(), dy.data()};
    kernel(xs, values, partials, {}, n);
    return std::tuple{value, dx, dy};
  };

  for (const unsigned lanes : {1u, 4u}) {
    const auto [v0, dx0, dy0] = run({.lanes = lanes});
    for (const auto &[what, o] :
         {std::pair{"slp", ddx::jit::Options{.lanes = lanes, .slp = true}},
          std::pair{"loop_vectorize",
                    ddx::jit::Options{.lanes = lanes, .loop_vectorize = true}}}) {
      const auto [v, dx, dy] = run(o);
      for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(v[i]),
                  std::bit_cast<std::uint64_t>(v0[i]))
            << what << ", lanes " << lanes << ", value at " << i;
        EXPECT_EQ(std::bit_cast<std::uint64_t>(dx[i]),
                  std::bit_cast<std::uint64_t>(dx0[i]))
            << what << ", lanes " << lanes << ", d/dx at " << i;
        EXPECT_EQ(std::bit_cast<std::uint64_t>(dy[i]),
                  std::bit_cast<std::uint64_t>(dy0[i]))
            << what << ", lanes " << lanes << ", d/dy at " << i;
      }
    }
  }
}

TEST(JitValue, SharedAndNested) {
  expect_matches_interpreter(
      [](Builder<> &, auto &v) {
        return (v[0] * v[1]) * (v[0] * v[1]) + sin(v[0] * v[1]);
      },
      2);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) { return sin(cos(exp(v[0] * v[1]))); }, 2);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) {
        auto t = v[0];
        for (int i = 0; i < 6; ++i) {
          t = sin(t * v[1]) + exp(t / 3.0);
        }
        return t;
      },
      2);
  expect_matches_interpreter(
      [](Builder<> &, auto &v) {
        return exp(v[0] * v[1]) + log(v[2]) / tanh(v[3]);
      },
      4);
}

TEST(JitValue, AgreesWithDdxThroughTheBridge) {
  constexpr auto x = ddx::var<"x">;
  constexpr auto y = ddx::var<"y">;
  Builder<> b;
  const auto root = ddx::rt::to_graph(b, exp(x) * sin(y));
  const auto graph = Graph<>::freeze(b, std::array{root.id(b)});
  const auto kernel = must_compile(graph);

  const std::array cx{1.0, 0.25, 2.5};
  const std::array cy{2.0, 1.75, 0.5};
  const std::array<const double *, 2> xs{cx.data(), cy.data()};
  std::array<double, 3> got{};
  double *const value_columns[]{got.data()};
  kernel(xs, value_columns, {}, {}, got.size());

  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], ddx::Equation{exp(x) * sin(y)}.evaluate(cx[i], cy[i]),
                1e-12);
  }
}

// rt::Equation::run() interprets when there is no kernel, so an empty Kernel
// must be falsy and bring-up must report its refusal rather than throw.
TEST(JitValue, AnUnbuiltKernelIsFalsyAndCreateAnswersWithAResult) {
  const ddx::jit::Kernel none;
  EXPECT_FALSE(static_cast<bool>(none));
  EXPECT_EQ(none.arity(), 0u);
  EXPECT_EQ(none.outputs(), 0u);

  auto c = ddx::jit::Compiler::create();
  ASSERT_TRUE(c.has_value()) << c.error().detail;

  Builder<> b;
  const auto x = var(b, "x");
  auto k = c->compile(Graph<>::freeze(b, std::array{(x + x).id(b)}));
  ASSERT_TRUE(k.has_value()) << k.error().detail;
  EXPECT_TRUE(static_cast<bool>(*k));
}

TEST(JitValue, EmptyBatchWritesNothing) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto graph = Graph<>::freeze(b, std::array{sin(x).id(b)});
  const auto kernel = must_compile(graph);

  const double *none = nullptr;
  const std::array<const double *, 1> xs{none};
  double sentinel = 42.0;
  double *const value_columns[]{&sentinel};
  kernel(xs, value_columns, {}, {}, 0);
  EXPECT_DOUBLE_EQ(sentinel, 42.0);
}

TEST(JitValue, ConstantFoldsToAStore) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto f = x * ddx::rt::RTExpression<>{0} + ddx::rt::RTExpression<>{7};
  const auto graph = Graph<>::freeze(b, std::array{f.id(b)});
  const auto kernel = must_compile(graph);

  const std::array cx{1.0, 2.0};
  const std::array<const double *, 1> xs{cx.data()};
  std::array<double, 2> got{};
  double *const value_columns[]{got.data()};
  kernel(xs, value_columns, {}, {}, got.size());
  EXPECT_DOUBLE_EQ(got[0], 7.0);
  EXPECT_DOUBLE_EQ(got[1], 7.0);
}

TEST(JitValue, SeparateCompilesCoexist) {
  Builder<> b1;
  const auto x1 = var(b1, "x");
  const auto k1 = must_compile(Graph<>::freeze(b1, std::array{sin(x1).id(b1)}));

  Builder<> b2;
  const auto x2 = var(b2, "x");
  const auto k2 = must_compile(Graph<>::freeze(b2, std::array{cos(x2).id(b2)}));

  const std::array c{0.5};
  const std::array<const double *, 1> xs{c.data()};
  double a = 0, bb = 0;
  double *const first[]{&a};
  double *const second[]{&bb};
  k1(xs, first, {}, {}, 1);
  k2(xs, second, {}, {}, 1);
  EXPECT_NEAR(a, std::sin(0.5), 1e-15);
  EXPECT_NEAR(bb, std::cos(0.5), 1e-15);
}

// A Kernel shares ownership of the JIT its code lives in; without that share
// this call lands in an unmapped page.
TEST(JitValue, KernelOutlivesItsCompiler) {
  ddx::jit::Kernel kernel;
  {
    auto local = *ddx::jit::Compiler::create();
    Builder<> b;
    const auto x = var(b, "x");
    auto k = local.compile(Graph<>::freeze(b, std::array{(x * x).id(b)}));
    ASSERT_TRUE(k.has_value()) << k.error().detail;
    kernel = std::move(*k);
  }
  ASSERT_TRUE(static_cast<bool>(kernel));

  const std::array c{3.0};
  const std::array<const double *, 1> xs{c.data()};
  double got = 0;
  double *const value_columns[]{&got};
  kernel(xs, value_columns, {}, {}, 1);
  EXPECT_DOUBLE_EQ(got, 9.0);
}

} // namespace
