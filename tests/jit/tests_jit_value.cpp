#include "ddx.hpp"
#include "jit/kernel.hpp"
#include "rt/bridge.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <numeric>
#include <type_traits>
#include <vector>

// ===========================================================================
// JIT'd values (src/jit/codegen.cpp)
//
// The interpreter in rt/interpret.hpp is the reference: it and the compiled
// kernel walk the same graph, so any disagreement is a codegen bug rather than
// a question about what the expression means.  Batch shape throughout, because
// that is the shape the kernel exists for.
// ===========================================================================

namespace {
using ddx::rt::Builder;
using ddx::rt::Graph;

// A host with no native target has no JIT to test; every test here needs one,
// so bring it up once and let the assertion name the reason if it will not.
ddx::jit::Compiler &compiler() {
  static ddx::jit::result<ddx::jit::Compiler> c = ddx::jit::Compiler::create();
  EXPECT_TRUE(c.has_value()) << (c ? "" : c.error().detail);
  return *c;
}

// A compile that has to succeed for the test to mean anything.  compile()
// answers with result<Kernel>; the assertion names LLVM's own reason when it
// does not, instead of an empty Kernel three lines later.
ddx::jit::Kernel must_compile(auto &&...args) {
  auto k = compiler().compile(static_cast<decltype(args) &&>(args)...);
  EXPECT_TRUE(k.has_value()) << (k ? std::string{} : k.error().detail);
  return k ? std::move(*k) : ddx::jit::Kernel{};
}

// Compile the value of `build` and compare a whole batch against the
// interpreter, point by point.
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

// A Compiler is the LLJIT, and a Kernel is a pointer into code it owns: there
// is one owner, but it has to be able to move -- Equation keeps one in a static
// and hands compilers around.
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

// These six have no LLVM intrinsic and go out as libm calls, so they exercise
// a different path through the emitter.
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
  // max/min are llvm.maximum/minimum, the IEEE-754 pair that propagates a NaN
  // operand from either side; maxnum/minnum would return the other operand
  // instead, and the interpreter's max_impl does not.
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

// A host the JIT will not come up on is not a failure a caller has to handle:
// rt::Equation::run() asks the kernel whether it exists and interprets if it
// does not.  That branch is gated on exactly this, so an empty Kernel has to be
// falsy, and bring-up has to report its refusal rather than throw it.
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

// A Kernel shares ownership of the JIT its code lives in, so a Compiler going
// out of scope does not take the code with it.  Without that share this call
// lands in an unmapped page, which is why it is asserted rather than left to
// the lifetime comment in kernel.hpp.
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
