#include "ddx.hpp"
#include "jit/kernel.hpp"
#include "rt/archive.hpp"
#include "rt/bridge.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <bit>
#include <format>
#include <latch>
#include <random>
#include <filesystem>
#include <fstream>
#include <thread>
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

// A cache directory of this process's own.  A fixed name under
// temp_directory_path() is shared mutable state between processes however
// private it looks, and it stays hidden precisely because ctest runs different
// binaries in parallel and never the same one twice -- so two copies of this
// binary, which is what a stress run is, would delete each other's entries
// through remove_all() and fail for reasons that look like the library.
[[nodiscard]] std::filesystem::path cache_dir(std::string_view what) {
  static const auto token = std::random_device{}();
  static std::atomic<unsigned> seq{0};
  return std::filesystem::temp_directory_path() /
         std::format("ddx_jit_{}_{:08x}_{}", what, token, seq.fetch_add(1));
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
  const auto graph =
      Graph<>::freeze(b, std::array{max(x, y).id(b), min(x, y).id(b)});
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
// A graph with no Jacobian block at all, which is what Equation::evaluate()
// freezes: the kernel takes an empty partials span, and the emitter has to
// produce a function that stores value columns and nothing else.
TEST(JitValue, AValuesOnlyGraphCompilesAndStoresNoPartials) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto graph =
      ddx::rt::GraphBuilder{b}.value(x * log(x) + exp(x * y)).build();
  ASSERT_EQ(graph.layout().jacobian, 0u);

  constexpr std::size_t n = 5;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.1 * static_cast<double>(i);
    cy[i] = 1.1 - 0.05 * static_cast<double>(i);
  }
  const std::array<const double *, 2> xs{cx.data(), cy.data()};

  const auto kernel = must_compile(graph);
  std::vector<double> value(n);
  double *const values[]{value.data()};
  kernel(xs, values, {}, {}, n);

  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(value[i], cx[i] * std::log(cx[i]) + std::exp(cx[i] * cy[i]),
                1e-12)
        << "value at " << i;
  }
}

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
  const auto graph = ddx::rt::GraphBuilder{b}.value(f).build_jacobian().build();

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
// several.  All four must agree to the bit -- which is what lets the default be
// the cheapest of them.  Level 0 is in that list only because contraction moved
// into the graph: it takes FastISel, which forms no FMAs of its own, and used to
// answer differently for exactly that reason.  An llvm.fma it is handed is an
// operation like any other.
TEST(JitValue, EveryCodegenLevelAgreesToTheBit) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  // Multiply-add shapes throughout, so there is an FMA at every level that
  // forms one -- the thing the levels could have disagreed about.
  const auto f = x * y + x * x - y + exp(x * y) * (y + 1.0) + log(x) * y;
  const auto graph = ddx::rt::GraphBuilder{b}.value(f).build_jacobian().build();

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
  for (const unsigned level : {0u, 2u, 3u}) {
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
  const auto graph = ddx::rt::GraphBuilder{b}.value(f).build_jacobian().build();

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
          std::pair{
              "loop_vectorize",
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

// The object a compile produced links back into the same JIT and answers the
// same bits.  This is the whole of what makes a compile storable: the bytes are
// the kernel, and adopting them skips emission, the pass pipeline and codegen
// alike.
TEST(JitValue, AnObjectAdoptsBackIntoTheSameKernel) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto root = (x * log(x) + exp(x * y) + sqrt(y)).id(b);
  const auto graph = ddx::rt::GraphBuilder<double>{b}
                         .values_from(std::array{root})
                         .jacobian_from(
                             ddx::rt::build_jacobian_impl<
                                 ddx::impl::DiffMode::Reverse>(b, root).partial)
                         .build();

  const auto compiled = must_compile(graph, ddx::jit::Options{.retain_object = true});
  ASSERT_TRUE(compiled);
  ASSERT_FALSE(compiled.object().empty()) << "retain_object kept nothing";

  // The name the compile used is not something a caller can guess, which is why
  // the kernel carries it: whoever stores the bytes stores the symbol beside
  // them.
  ASSERT_FALSE(compiled.symbol().empty());
  const auto adopted = compiler().adopt(
      compiled.object(), compiled.symbol(), compiled.arity(), compiled.values(),
      compiled.jacobian_columns(), compiled.hessian_columns());
  ASSERT_TRUE(adopted.has_value()) << adopted.error().detail;

  const std::array cx{0.4, 0.9}, cy{1.3, 0.7};
  const std::array<const double *, 2> xs{cx.data(), cy.data()};
  std::array<double, 2> f1{}, dx1{}, dy1{}, f2{}, dx2{}, dy2{};
  double *const v1[]{f1.data()};
  double *const g1[]{dx1.data(), dy1.data()};
  double *const v2[]{f2.data()};
  double *const g2[]{dx2.data(), dy2.data()};
  compiled(xs, v1, g1, {}, 2);
  (*adopted)(xs, v2, g2, {}, 2);

  EXPECT_EQ(f1, f2) << "the adopted kernel moved a value";
  EXPECT_EQ(dx1, dx2) << "the adopted kernel moved d/dx";
  EXPECT_EQ(dy1, dy2) << "the adopted kernel moved d/dy";

  // And it carries its own bytes, so a kernel that came off disk can go back.
  EXPECT_EQ(adopted->object().size(), compiled.object().size());
}

// The point of the cache: the second compile of the same graph does no codegen
// at all, and answers the same bits.  Timing is not the assertion -- the report
// is, because a cache hit does not run the phases rather than running them
// quickly.
TEST(JitValue, ACachedObjectSkipsTheWholeCompile) {
  const auto dir = cache_dir("skips");
  std::filesystem::remove_all(dir);

  const auto build = [] {
    Builder<> b;
    const auto x = var(b, "x");
    const auto y = var(b, "y");
    const auto root = (x * log(x) + exp(x * y) + sqrt(y)).id(b);
    return std::pair{
        std::move(b),
        root}; // the arena has to outlive the freeze that reads it
  };

  ddx::jit::Options opt;
  opt.cache_dir = dir.string();

  ddx::jit::CompileReport cold, warm;
  double cold_f = 0, warm_f = 0;
  const std::array c{0.7};
  const std::array<const double *, 2> xs{c.data(), c.data()};

  for (const auto &[report, out] :
       {std::pair{&cold, &cold_f}, std::pair{&warm, &warm_f}}) {
    auto [b, root] = build();
    const auto graph = Graph<>::freeze(b, std::array{root});
    auto k = compiler().compile(graph, opt, report);
    ASSERT_TRUE(k.has_value()) << k.error().detail;
    double *const values[]{out};
    (*k)(xs, values, {}, {}, 1);
  }

  EXPECT_GT(cold.codegen.count(), 0) << "the cold compile did no codegen";
  EXPECT_EQ(warm.codegen.count(), 0) << "the warm compile ran the backend";
  EXPECT_EQ(warm.emit.count(), 0) << "the warm compile emitted IR";
  EXPECT_EQ(warm.optimize.count(), 0) << "the warm compile ran the passes";
  EXPECT_EQ(cold_f, warm_f) << "the cached kernel moved a bit";

  std::filesystem::remove_all(dir);
}

// A cache entry with a byte flipped in it must not be run.  The checksum is
// what stands between a corrupt file and machine code being jumped into.
TEST(JitValue, ACorruptCacheEntryIsAMissNotACrash) {
  const auto dir = cache_dir("corrupt");
  std::filesystem::remove_all(dir);

  ddx::jit::Options opt;
  opt.cache_dir = dir.string();

  Builder<> b;
  const auto x = var(b, "x");
  const auto graph = Graph<>::freeze(b, std::array{(x * x + log(x)).id(b)});
  ASSERT_TRUE(compiler().compile(graph, opt).has_value());

  // Every entry the cold compile wrote, damaged in the payload.
  std::size_t damaged = 0;
  for (const auto &e : std::filesystem::directory_iterator{dir}) {
    std::fstream f{e.path(), std::ios::in | std::ios::out | std::ios::binary};
    ASSERT_TRUE(f.is_open());
    f.seekp(static_cast<std::streamoff>(ddx::rt::header_bytes + 8));
    f.put('\xff');
    ++damaged;
  }
  ASSERT_GT(damaged, 0u) << "the cold compile cached nothing";

  ddx::jit::CompileReport after;
  const auto k = compiler().compile(graph, opt, &after);
  ASSERT_TRUE(k.has_value()) << "a corrupt entry failed the compile";
  EXPECT_GT(after.codegen.count(), 0) << "a corrupt entry was adopted";

  double got = 0;
  const std::array c{0.7};
  const std::array<const double *, 1> xs{c.data()};
  double *const values[]{&got};
  (*k)(xs, values, {}, {}, 1);
  EXPECT_NEAR(got, 0.7 * 0.7 + std::log(0.7), 1e-15);

  std::filesystem::remove_all(dir);
}

// A cache entry is parsed bytes, and it carries a length read out of its own
// payload -- the shape where a hand-picked corrupt case proves nothing.  Every
// byte of the prologue and a stride through the payload, each flipped, and every
// truncation: the assertion is only that a compile still *returns a correct
// kernel*, because every one of these must degrade to a miss.
TEST(JitValue, ACacheEntrySurvivesCorruptionAndTruncation) {
  const auto dir =
      cache_dir("fuzz");
  std::filesystem::remove_all(dir);

  ddx::jit::Options opt;
  opt.cache_dir = dir.string();

  Builder<> b;
  const auto x = var(b, "x");
  const auto graph = Graph<>::freeze(b, std::array{(x * x + log(x)).id(b)});
  ASSERT_TRUE(compiler().compile(graph, opt).has_value());

  std::filesystem::path entry;
  for (const auto &e : std::filesystem::directory_iterator{dir}) {
    entry = e.path();
  }
  ASSERT_FALSE(entry.empty()) << "the cold compile cached nothing";
  const auto pristine = *ddx::rt::read_file(entry);
  ASSERT_GT(pristine.size(), ddx::rt::header_bytes);

  const std::array c{0.7};
  const std::array<const double *, 1> xs{c.data()};
  const double want = 0.7 * 0.7 + std::log(0.7);

  // Two assertions, and the second is the one with teeth.  "Still returns a
  // correct kernel" is satisfied by an entry that was adopted *and happened to
  // be harmless*; "the backend ran" says the entry was refused outright, which
  // is the property that has to hold for every damaged byte.  Reporting only
  // the first would be a test that passes while corrupt entries are executed.
  const auto rejected = [&](const char *what, std::size_t at) {
    ddx::jit::CompileReport rep;
    const auto k = compiler().compile(graph, opt, &rep);
    ASSERT_TRUE(k.has_value()) << what << " at " << at << " failed the compile";
    EXPECT_GT(rep.codegen.count(), 0)
        << what << " at " << at << " was adopted rather than refused";
    double got = 0;
    double *const values[]{&got};
    (*k)(xs, values, {}, {}, 1);
    EXPECT_NEAR(got, want, 1e-15) << what << " at " << at << " ran wrong code";
  };

  // Every prologue byte under three masks, then a stride through the payload.
  // The prologue gets no stride because it is the one region no checksum can
  // cover.  A sweep rather than a spot check on the bytes that matter today,
  // because what it is really for is the *next* header field somebody adds and
  // forgets to verify -- both halves of this format have gained fields recently.
  //
  // Three masks and not one because one only detects a field that is *entirely*
  // unverified: relax `scalar_size` to a range check and `~8 = 247` is still
  // refused while `8 -> 9` is adopted, so `~b` alone would call that field
  // verified.  Whole-byte, low-bit, high-bit.
  //
  // Three and not all 255, which is the honest trade rather than a claim that
  // three suffices.  Exhaustive was run once -- **0 of 14280 accepted** -- and
  // costs 98 s against an 18 s suite, because every rejection here provokes a
  // real recompile where the archive's equivalent sweep only reparses a file.
  // What makes three adequate *today* is that every field is checked with `!=`
  // against a value derived elsewhere, and an exact check catches any change at
  // all.  That is a property of the current fields, not of the sampling: **a
  // field added with a range check needs this exhaustive again**, and the
  // scalar_size demonstration above is exactly what that looks like.
  const std::size_t stride = std::max<std::size_t>(1, pristine.size() / 64);
  for (std::size_t i = 0; i < ddx::rt::header_bytes; ++i) {
    for (const std::byte mask :
         {std::byte{0xff}, std::byte{0x01}, std::byte{0x80}}) {
      auto damaged = pristine;
      damaged[i] ^= mask;
      if (damaged[i] == pristine[i]) {
        continue;
      }
      ASSERT_TRUE(ddx::rt::write_file(entry, damaged).has_value());
      rejected("a flipped prologue byte", i);
    }
  }
  for (std::size_t i = ddx::rt::header_bytes; i < pristine.size(); i += stride) {
    auto damaged = pristine;
    damaged[i] = ~damaged[i];
    ASSERT_TRUE(ddx::rt::write_file(entry, damaged).has_value());
    rejected("a flipped payload byte", i);
  }

  for (std::size_t n = 0; n < pristine.size(); n += stride) {
    ASSERT_TRUE(ddx::rt::write_file(
                    entry, std::span<const std::byte>{pristine}.first(n))
                    .has_value());
    rejected("a truncation to", n);
  }

  // The case the sweep above cannot reach: a symbol length near 2^32, where
  // `4 + length` in 32-bit arithmetic wraps past the bounds check it is meant
  // to fail.  Damaging the file cannot get there -- the checksum fails first,
  // every time -- so the entry is *rebuilt* with the wrapping length and a CRC
  // that agrees with it, which is what makes this test not vacuous.  CRC32
  // detects accidents; it does not sign anything.
  {
    const auto head = ddx::rt::get_header(pristine, "ddxjitob");
    ASSERT_TRUE(head.has_value());
    std::vector<std::byte> payload(
        pristine.begin() + ddx::rt::header_bytes, pristine.end());
    ASSERT_GT(payload.size(), 4u);
    std::ranges::fill_n(payload.begin(), 4, std::byte{0xff}); // 0xffffffff

    std::vector<std::byte> forged(ddx::rt::header_bytes + payload.size());
    ddx::rt::put_header({.magic = "ddxjitob",
                         .format = head->format,
                         .schema = head->schema,
                         .scalar_size = head->scalar_size,
                         .scalar_kind = head->scalar_kind,
                         .model_digest = head->model_digest,
                         .payload_bytes = payload.size(),
                         .payload_crc = ddx::rt::checksum(payload)},
                        forged);
    std::ranges::copy(payload, forged.begin() + ddx::rt::header_bytes);
    ASSERT_TRUE(ddx::rt::write_file(entry, forged).has_value());
    rejected("a wrapping symbol length whose CRC agrees", 0);
  }

  std::filesystem::remove_all(dir);
}

// The cache is the multi-writer case: a parallel build, or two workers warming
// the same key, all miss together, all compile, and all write one path at once.
// Every one of them must get a correct kernel, exactly one whole entry must be
// left, and nothing may be left staged.
//
// This is *not* a regression test for rt::write_file's per-writer staging name,
// and the reason is worth stating because two plausible ones are wrong.
//
// Measured against the shared ".tmp" it replaced: this test passes 0/10, while
// **120 of 160 cache writes fail**.  So the writers do collide, constantly --
// spacing is not what hides it, and neither is payload size (checked to 40
// variables).  What hides it is that a losing writer's rename finds nothing to
// move and `write_entry` discards that error by design: a cache that cannot be
// written is a cache that misses, which is exactly the degradation this test
// asserts is survivable.  One writer always wins, so one whole entry always
// lands, and every assertion below holds either way.
//
// There is also no torn file to find, at any size: rename is atomic, so the
// content race resolves before anything a reader could see, and RtArchive's
// concurrent-writer test measured zero torn files from 1 KiB to 1 MiB over forty
// rounds with the bug fully present.  Neither suite has a tearing test because
// on this platform there is nothing there to test.
//
// What this covers is the end of the pipe: whatever the writers did, one whole
// entry loads afterwards.
TEST(JitValue, ConcurrentCompilesLeaveOneWholeCacheEntry) {
  const auto dir =
      cache_dir("writers");
  std::filesystem::remove_all(dir);

  ddx::jit::Options opt;
  opt.cache_dir = dir.string();

  constexpr std::size_t writers = 8;
  const double want = 0.7 * 0.7 + std::log(0.7);
  std::array<double, writers> got{};
  std::array<bool, writers> ok{};
  std::array<bool, writers> compiled{};

  // The graphs are built *before* the rendezvous and the compiles start after
  // it, so every writer reads the cache before any of them has written.  That
  // makes "all eight miss" structural rather than lucky -- without it, a
  // straggler thread on a loaded box starts after the first writer has landed
  // its entry, hits, and the overlap assertion below becomes the same
  // scheduling-dependent coin flip that the ladder turned
  // DroppingAMidFlightCompileDoesNotBlock into.
  std::latch start{writers};

  {
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < writers; ++t) {
      threads.emplace_back([&, t] {
        // Its own arena: one Builder is not shared between threads.  Two
        // independently built graphs hashing alike is also the cache's premise,
        // so this tests that too.
        Builder<> b;
        const auto x = var(b, "x");
        const auto graph =
            Graph<>::freeze(b, std::array{(x * x + log(x)).id(b)});
        start.arrive_and_wait();
        ddx::jit::CompileReport rep;
        const auto k = compiler().compile(graph, opt, &rep);
        if (!k) {
          return;
        }
        compiled[t] = rep.codegen.count() > 0;
        const std::array c{0.7};
        const std::array<const double *, 1> xs{c.data()};
        double *const values[]{&got[t]};
        (*k)(xs, values, {}, {}, 1);
        ok[t] = true;
      });
    }
    for (auto &th : threads) {
      th.join();
    }
  }

  // Without this the test can quietly stop being about concurrency: if the
  // writers ever serialise into one compile and seven cache hits, every
  // assertion below still passes and nothing races at all.  It is an assertion
  // about *work done*, not about timing, and the latch above is what keeps it
  // that way -- assert overlap without one and this is a flake, not a guard.
  EXPECT_EQ(std::ranges::count(compiled, true), writers)
      << "a writer hit the cache, so the writers did not all overlap";

  for (std::size_t t = 0; t < writers; ++t) {
    EXPECT_TRUE(ok[t]) << "writer " << t << " got no kernel";
    EXPECT_NEAR(got[t], want, 1e-15) << "writer " << t << " ran wrong code";
  }

  // One key, so one entry -- and no staging file left behind by a writer that
  // lost the rename.
  std::size_t entries = 0;
  for (const auto &e : std::filesystem::directory_iterator{dir}) {
    EXPECT_EQ(e.path().extension(), ".ddxjit")
        << "left staged: " << e.path().filename();
    ++entries;
  }
  EXPECT_EQ(entries, 1u) << "one graph and one option set should be one entry";

  // And what landed is a whole entry, not a mixture of eight.
  Builder<> b;
  const auto x = var(b, "x");
  const auto graph = Graph<>::freeze(b, std::array{(x * x + log(x)).id(b)});
  ddx::jit::CompileReport warm;
  const auto k = compiler().compile(graph, opt, &warm);
  ASSERT_TRUE(k.has_value()) << k.error().detail;
  EXPECT_EQ(warm.codegen.count(), 0) << "what the writers left did not load";

  std::filesystem::remove_all(dir);
}

// No directory named, nothing written: a library does not put files on a
// caller's disk because it would be faster.
TEST(JitValue, NoCacheDirectoryWritesNothing) {
  const auto dir =
      cache_dir("absent");
  std::filesystem::remove_all(dir);

  Builder<> b;
  const auto x = var(b, "x");
  ASSERT_TRUE(
      compiler().compile(Graph<>::freeze(b, std::array{(x * x).id(b)}))
          .has_value());
  EXPECT_FALSE(std::filesystem::exists(dir));
}

// Nothing is kept unless it was asked for: machine code held for the life of
// every kernel is not a default anyone should pay.
TEST(JitValue, AnObjectIsNotKeptUnlessAsked) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto k = must_compile(Graph<>::freeze(b, std::array{(x * x).id(b)}));
  EXPECT_TRUE(k.object().empty());
}

// The symbol is the one thing adopt() takes from the caller and cannot check
// against the object, so what stops a forged one is that the lookup is scoped to
// the dylib the object went into.  Resolved through the link order instead, a
// forged name would find some *other* function and `toPtr<function_type>()`
// would hand it back to be called under the five-pointer kernel ABI -- which on
// SysV neither crashes nor writes anything, so the caller gets untouched output
// buffers and no diagnostic.  Silent wrong answers, which is worse than a fault.
//
// The scoping already closes it.  This test exists because widening that lookup
// would look like a simplification -- and it does bite: swapping
// `lookup(*jd, symbol)` for `lookup(symbol)` fails this test on `memcpy`, which
// the process generator resolves.  `memcpy` is the case doing the work: measured
// under that widened lookup, the libm names do *not* resolve by bare name and
// would not have caught it.  They stay as cheap coverage of the intent, not
// because they carry the test.
TEST(JitValue, AForgedSymbolCannotReachThroughTheLinkOrder) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto graph = Graph<>::freeze(b, std::array{(x * x + exp(x)).id(b)});
  const auto compiled =
      must_compile(graph, ddx::jit::Options{.retain_object = true});
  ASSERT_TRUE(compiled);
  ASSERT_FALSE(compiled.object().empty());

  // Its own symbol links, so the negatives below are not vacuous.
  EXPECT_TRUE(compiler()
                  .adopt(compiled.object(), compiled.symbol(), 1, 1, 0, 0)
                  .has_value());

  // Every one of these resolves in the main dylib -- libm through define_libm,
  // memcpy through the process generator -- and the object itself defines none
  // of them.
  for (const std::string_view forged :
       {"sin", "exp", "log", "memcpy", "ddx_kernel_999"}) {
    EXPECT_FALSE(
        compiler().adopt(compiled.object(), forged, 1, 1, 0, 0).has_value())
        << forged << " resolved through the link order";
  }
}

// A cache entry that will not link is a miss, not a crash and not an abort.
TEST(JitValue, AdoptingRubbishIsAnErrorNotACrash) {
  const std::array<std::byte, 8> rubbish{std::byte{0x7f}, std::byte{'E'},
                                         std::byte{'L'},  std::byte{'F'},
                                         std::byte{0},    std::byte{0},
                                         std::byte{0},    std::byte{0}};
  const auto k = compiler().adopt(rubbish, "ddx_kernel_0", 1, 1, 0, 0);
  EXPECT_FALSE(k.has_value()) << "a truncated object linked";
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
