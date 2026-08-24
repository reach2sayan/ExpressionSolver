#include "ddx.hpp"
#include "rt/bridge.hpp"
#include "rt/equation.hpp"
#include "util/ranges.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <tuple>
#include <iterator>
#include <numbers>
#include <ranges>
#include <span>
#include <vector>

// One name for both kinds of expression.  The specialisation is keyed on the
// RTExpression<T> *pattern*, not a constraint: a constraint over
// <TFirst, TRest...> has the same pattern as the compile-time one and neither
// subsumes the other.  to_graph lowers a typed expression, so the same function
// can be asked the same question both ways.

namespace {
constexpr auto sx = ddx::var<"x">;
constexpr auto sy = ddx::var<"y">;

// constexpr, never consteval (NOTES.md): one implementation serves both a
// static_assert and a value read at run time.
consteval double slope_at_two() {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto eq = ddx::rt::equation(x * x + 3.0 * x);
  return (*eq.jacobian(2.0))[0];
}
static_assert(slope_at_two() == 7.0);

TEST(RtEquation, MatchesTheCompileTimeEquation) {
  const auto f = exp(sx) * sin(sy) + sx * sy;
  const ddx::Equation compile_time{f};

  ddx::rt::Builder<> b;
  const auto runtime = ddx::rt::equation(ddx::rt::to_graph(b, f));

  EXPECT_EQ(runtime.arity(), 2u);
  EXPECT_NEAR(*runtime.evaluate(1.3, 0.7), compile_time.evaluate(1.3, 0.7),
              1e-12);

  const auto want = compile_time.jacobian(1.3, 0.7);
  const auto got = *runtime.jacobian(1.3, 0.7);
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], want[i], 1e-12) << "partial " << i;
  }
}

// CEvalArg admits four spellings; the runtime side resolves names against a
// run-time symbol list where make_point uses a compile-time one.
TEST(RtEquation, EveryCallSpellingAgrees) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(log(x) * sqrt(y));

  const auto positional = *eq.jacobian(1.3, 0.7);
  const auto by_range = *eq.jacobian(std::array{1.3, 0.7});
  const auto by_name = *eq.jacobian(ddx::named<"y">(0.7), ddx::named<"x">(1.3));

  ASSERT_EQ(positional.size(), 2u);
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_DOUBLE_EQ(by_range[i], positional[i]);
    EXPECT_DOUBLE_EQ(by_name[i], positional[i]);
  }
}

TEST(RtEquation, SystemsGiveAJacobian) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x * y + sin(x), exp(x) - y * y);

  const auto J = *eq.jacobian(1.3, 0.7);
  ASSERT_EQ(J.size(), 4u); // 2 functions x 2 symbols, row-major

  // Row k is the Jacobian row of function k, which the compile-time side agrees
  // on.
  const auto row0 = ddx::Equation{sx * sy + sin(sx)}.jacobian(1.3, 0.7);
  const auto row1 = ddx::Equation{exp(sx) - sy * sy}.jacobian(1.3, 0.7);
  EXPECT_NEAR(J[0], row0[0], 1e-12);
  EXPECT_NEAR(J[1], row0[1], 1e-12);
  EXPECT_NEAR(J[2], row1[0], 1e-12);
  EXPECT_NEAR(J[3], row1[1], 1e-12);
}

TEST(RtEquation, MultipleOutputsEvaluateToAVector) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto eq = ddx::rt::equation(x * x, sin(x));
  const auto v = *eq.evaluate(0.5);
  ASSERT_EQ(v.size(), 2u);
  EXPECT_DOUBLE_EQ(v[0], 0.25);
  EXPECT_NEAR(v[1], std::sin(0.5), 1e-15);
}

TEST(RtEquation, AWrongSizedPointIsRejected) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x + y);
  EXPECT_EQ(eq.jacobian(std::array{1.0}).error().code, ddx::errc::wrong_arity);
  EXPECT_EQ(eq.jacobian(ddx::named<"z">(1.0)).error().code,
            ddx::errc::unknown_symbol);
}

} // namespace

namespace {

// dual2nd on the compile-time side, because that is what its hessian needs;
// the runtime graph is over plain double either way.
TEST(RtEquation, HessianMatchesTheCompileTimeEquation) {
  using D = ddx::dual2nd;
  constexpr auto dx = ddx::var<"x", D>;
  constexpr auto dy = ddx::var<"y", D>;
  const auto f = exp(dx) * sin(dy) + dx * dx * dy;

  ddx::rt::Builder<> b;
  const auto eq = ddx::rt::equation(ddx::rt::to_graph(b, f));
  const auto got = *eq.hessian(0.7, 1.1);
  const auto want = ddx::Equation{f}.hessian(0.7, 1.1);

  ASSERT_EQ(got.size(), 4u);
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      EXPECT_NEAR(got[i * 2 + j], ddx::impl::get_real_part<1>(want[i, j]),
                  1e-10)
          << "H[" << i << "][" << j << "]";
    }
  }
  EXPECT_TRUE(std::isfinite(got[1]));
  EXPECT_DOUBLE_EQ(got[1], got[2]) << "a Hessian is symmetric";
}

// The colour count is a property of the expression, and the facade exposes it
// because whether colouring saves anything is not something a caller can guess.
TEST(RtEquation, HessianColoursReflectTheCoupling) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto z = var(b, "z");

  const auto coupled = ddx::rt::equation(x * y + y * z + x * z);
  EXPECT_EQ(coupled.hessian_colors(), 3u) << "every pair couples";

  ddx::rt::Builder<> b2;
  const auto a = var(b2, "a");
  const auto c = var(b2, "c");
  const auto d = var(b2, "d");
  const auto separable = ddx::rt::equation(sin(a) + exp(c) + d * d);
  EXPECT_EQ(separable.hessian_colors(), 1u) << "nothing couples: one sweep";
}

} // namespace

namespace {

// equation() makes the arena, hands out symbols from it and moves it into the
// result, so a caller never names or keeps a Builder alive.
TEST(RtEquation, TheFactoryHidesTheArena) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return exp(x) * sin(y);
  });

  EXPECT_EQ(eq.arity(), 2u);
  const auto g = *eq.jacobian(1.0, 2.0);
  ASSERT_EQ(g.size(), 2u);
  EXPECT_NEAR(*eq.evaluate(1.0, 2.0), std::exp(1.0) * std::sin(2.0), 1e-12);
  EXPECT_NEAR(g[0], std::exp(1.0) * std::sin(2.0), 1e-12);
  EXPECT_NEAR(g[1], std::exp(1.0) * std::cos(2.0), 1e-12);
}

// Owning the arena is what lets an Equation be returned, stored and passed.
TEST(RtEquation, AnOwningEquationOutlivesItsScope) {
  const auto make = [](double c) {
    return ddx::rt::equation([c] {
      const auto x = ddx::rt::var("x");
      return c * x * x;
    });
  };

  const auto eq = make(3.0);
  EXPECT_NEAR(*eq.evaluate(2.0), 12.0, 1e-12);
  EXPECT_NEAR((*eq.jacobian(2.0))[0], 12.0, 1e-12); // d(3x^2)/dx = 6x
  EXPECT_NEAR((*eq.hessian(2.0))[0], 6.0, 1e-12);
}

// Same answers through the compiled kernel and the interpreter alike.
TEST(RtEquation, BatchAgreesWithPerPoint) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(exp(x) * sin(y) + x * y);

  constexpr std::size_t n = 8;
  std::vector<double> cx(n), cy(n), f(n), dx(n), dy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.1 * static_cast<double>(i);
    cy[i] = 1.0 + 0.2 * static_cast<double>(i);
  }
  const double *const columns[]{cx.data(), cy.data()};
  double *const values[]{f.data()};
  double *const partials[]{dx.data(), dy.data()};

  *eq.jacobian(columns, values, partials, n);

  for (std::size_t i = 0; i < n; ++i) {
    const auto want = *eq.jacobian(cx[i], cy[i]);
    EXPECT_NEAR(f[i], *eq.evaluate(cx[i], cy[i]), 1e-12) << "point " << i;
    EXPECT_NEAR(dx[i], want[0], 1e-12) << "point " << i;
    EXPECT_NEAR(dy[i], want[1], 1e-12) << "point " << i;
  }
}

TEST(RtEquation, BatchHessianFillsTheCompressedColumns) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(exp(x) * sin(y) + x * x * y);

  constexpr std::size_t n = 4;
  const std::size_t columns = *eq.hessian_columns();
  ASSERT_GT(columns, 0u);

  std::vector<double> cx(n, 0.7), cy(n, 1.1), f(n), dx(n), dy(n);
  std::vector<std::vector<double>> blocks(columns, std::vector<double>(n));
  const auto block_ptrs =
      blocks | std::views::transform([](auto &c) { return c.data(); }) |
      ddx::impl::to<std::vector<double *>>();
  const double *const inputs[]{cx.data(), cy.data()};
  double *const values[]{f.data()};
  double *const partials[]{dx.data(), dy.data()};

  *eq.hessian(inputs, values, partials, block_ptrs, n);

  // The dense per-point Hessian is the scatter of those columns.
  const auto dense = *eq.hessian(0.7, 1.1);
  EXPECT_NEAR(dense[1], dense[2], 1e-12) << "symmetric";
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(f[i], *eq.evaluate(0.7, 1.1), 1e-12);
  }
}

} // namespace

namespace {

// Any contiguous range of columns, so a caller reaches for whatever it already
// has rather than building a span by hand.
TEST(RtEquation, BatchTakesAnyContiguousRange) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x * y);

  constexpr std::size_t n = 3;
  std::vector<double> cx(n, 2.0), cy(n, 3.0), f(n), dx(n), dy(n);

  const std::vector<const double *> as_vector{cx.data(), cy.data()};
  const double *const as_array[]{cx.data(), cy.data()};
  std::vector<double *> outputs{f.data()};
  std::vector<double *> partials{dx.data(), dy.data()};

  *eq.jacobian(as_vector, outputs, partials, n);
  EXPECT_DOUBLE_EQ(f[0], 6.0);
  EXPECT_DOUBLE_EQ(dx[0], 3.0);

  std::ranges::fill(f, 0.0);
  *eq.jacobian(as_array, outputs, partials, n);
  EXPECT_DOUBLE_EQ(f[0], 6.0);

  *eq.jacobian(std::span{as_vector}, std::span{outputs}, std::span{partials},
               n);
  EXPECT_DOUBLE_EQ(f[0], 6.0);
}

// A wrong column count is silent corruption once it reaches the kernel; the
// ABI cannot notice, so this layer has to.
TEST(RtEquation, AMismatchedColumnCountIsRejected) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = ddx::rt::equation(x * y);

  std::vector<double> cx(2, 1.0), cy(2, 1.0), f(2), dx(2), dy(2);
  const std::vector<const double *> xs{cx.data(), cy.data()};
  std::vector<double *> values{f.data()};
  std::vector<double *> partials{dx.data(), dy.data()};
  EXPECT_TRUE(eq.jacobian(xs, values, partials, 2).has_value());

  // One partial per symbol, so one column is one too few.
  const std::vector<double *> one_partial{dx.data()};
  EXPECT_EQ(eq.jacobian(xs, values, one_partial, 2).error().code,
            ddx::errc::wrong_column_count);

  // One value column for one function, so two is one too many.
  const std::vector<double *> two_values{f.data(), dy.data()};
  EXPECT_EQ(eq.jacobian(xs, two_values, partials, 2).error().code,
            ddx::errc::wrong_column_count);

  // And the point itself has to name every symbol.
  const std::vector<const double *> short_point{cx.data()};
  EXPECT_EQ(eq.jacobian(short_point, values, partials, 2).error().code,
            ddx::errc::wrong_column_count);
}

} // namespace

namespace {

// Symbols come from equation(); no arena is ever named.
TEST(RtEquation, TheArenaIsInvisible) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return x * y + sin(x);
  });
  EXPECT_EQ(eq.arity(), 2u);
  EXPECT_NEAR(*eq.evaluate(2.0, 3.0), 6.0 + std::sin(2.0), 1e-12);

  // var() poisons the expression rather than answering with an error, which
  // would cost every model lambda a dereference.  The poison survives every
  // operator and reaches the Equation, which carries it.
  const auto stray = ddx::rt::var("stray");
  EXPECT_TRUE(stray.poisoned());
  EXPECT_TRUE((stray * 2.0 + 1.0).poisoned());

  const auto poisoned = ddx::rt::equation(stray * 2.0);
  EXPECT_TRUE(poisoned.poisoned());
  ASSERT_TRUE(poisoned.status().has_value());
  EXPECT_EQ(poisoned.status()->code, ddx::errc::no_arena);

  // Every call spends the error rather than reading a graph that is not there.
  EXPECT_EQ(poisoned.evaluate(1.0).error().code, ddx::errc::no_arena);
  EXPECT_EQ(poisoned.jacobian(1.0).error().code, ddx::errc::no_arena);
  EXPECT_EQ(poisoned.point(1.0).error().code, ddx::errc::no_arena);
  // Every count is nullopt, never a 0 a caller could loop over by accident.
  EXPECT_FALSE(poisoned.arity().has_value());
  EXPECT_FALSE(poisoned.symbols().has_value());
  EXPECT_FALSE(poisoned.jacobian_columns().has_value());
  EXPECT_FALSE(poisoned.value_columns().has_value());
  EXPECT_FALSE(poisoned.hessian_columns().has_value());
  EXPECT_FALSE(poisoned.hessian_colors().has_value());
}

// A bare literal reaches no graph, so it names no builder -- distinct from
// no_arena, and surfacing on the poisoned Equation.
TEST(RtEquation, ALiteralNamesNoGraph) {
  const auto eq = ddx::rt::equation(ddx::rt::RTExpression<double>{2.0});
  ASSERT_TRUE(eq.status().has_value());
  EXPECT_EQ(eq.status()->code, ddx::errc::no_graph);
  EXPECT_EQ(eq.evaluate(1.0).error().code, ddx::errc::no_graph);
}

// A system has one Hessian per output -- the shape Equation::hessian returns as
// nd_stack_t<S, m, n, 2> on the compile-time side.
TEST(RtEquation, HessianOfASystemIsOneBlockPerOutput) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return std::array{x * x * y, exp(x) * y};
  });

  constexpr std::size_t n = 2;
  const auto H = *eq.hessian(0.7, 1.3);
  ASSERT_EQ(H.size(), 2 * n * n);

  const auto at = [&](std::size_t k, std::size_t i, std::size_t j) {
    return H[(k * n + i) * n + j];
  };

  EXPECT_NEAR(at(0, 0, 0), 2 * 1.3, 1e-10);
  EXPECT_NEAR(at(0, 0, 1), 2 * 0.7, 1e-10);
  EXPECT_NEAR(at(0, 1, 1), 0.0, 1e-10);
  EXPECT_NEAR(at(1, 0, 0), std::exp(0.7) * 1.3, 1e-10);
  EXPECT_NEAR(at(1, 0, 1), std::exp(0.7), 1e-10);
  EXPECT_NEAR(at(1, 1, 1), 0.0, 1e-10);

  for (std::size_t k = 0; k < 2; ++k) {
    EXPECT_NEAR(at(k, 0, 1), at(k, 1, 0), 1e-12);
  }
}

// One Taylor sweep rather than K nested duals, matching
// univariate_derivative_impl.
TEST(RtEquation, UnivariateDerivativesToArbitraryOrder) {
  const auto eq = ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    return exp(x) * sin(x);
  });

  // d^k/dx^k [e^x sin x] = 2^(k/2) e^x sin(x + k*pi/4)
  constexpr double at = 0.7;
  const auto expected = [](std::size_t k) {
    return std::pow(std::numbers::sqrt2, static_cast<double>(k)) *
           std::exp(at) *
           std::sin(at + static_cast<double>(k) * std::numbers::pi / 4);
  };
  EXPECT_NEAR(*eq.univariate_derivative<1>(at), expected(1), 1e-10);
  EXPECT_NEAR(*eq.univariate_derivative<2>(at), expected(2), 1e-10);
  EXPECT_NEAR(*eq.univariate_derivative<4>(at), expected(4), 1e-10);

  // It is a one-variable question; more than one symbol has no answer.
  const auto two =
      ddx::rt::equation([] { return ddx::rt::var("x") * ddx::rt::var("y"); });
  EXPECT_EQ(two.univariate_derivative<1>(1.0).error().code,
            ddx::errc::not_univariate);
}

} // namespace

// --- choosing the backend
// ----------------------------------------------------- Compiling is not free
// and its cost grows faster than the graph, so a caller has to be able to
// decline it on a build that has the backend.

#ifdef DDX_HAS_JIT
TEST(RtEquation, InterpretDeclinesTheBackendAndAgreesWithIt) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto eq = ddx::rt::equation(x * log(x) + y * exp(x * y) + sqrt(x + y));

  constexpr std::size_t n = 16;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.02 * static_cast<double>(i);
    cy[i] = 0.7 - 0.01 * static_cast<double>(i);
  }
  const double *const columns[]{cx.data(), cy.data()};

  const auto run = [&](ddx::rt::Backend which) {
    std::vector<double> f(n), dx(n), dy(n);
    double *const values[]{f.data()};
    double *const partials[]{dx.data(), dy.data()};
    eq.options({.backend = which});
    const bool kernel = eq.uses_kernel();
    *eq.jacobian(columns, values, partials, n);
    return std::tuple{f, dx, dy, kernel};
  };

  const auto [f_jit, dx_jit, dy_jit, used_kernel] =
      run(ddx::rt::Backend::Compile);
  const auto [f_int, dx_int, dy_int, used_none] =
      run(ddx::rt::Backend::Interpret);

  EXPECT_EQ(eq.options().backend, ddx::rt::Backend::Interpret);
  EXPECT_FALSE(used_none) << "Interpret still reached for a kernel";
  EXPECT_TRUE(used_kernel) << "Compile did not compile on a JIT build";

  // Near, not equal: the kernel contracts a multiply and an add into an FMA
  // where the interpreter evaluates them separately, so the two paths agree to
  // rounding rather than to the bit.  Declining the backend changes which
  // arithmetic runs, which is the whole point of being able to decline it.
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(f_jit[i], f_int[i], 1e-12) << "value at " << i;
    EXPECT_NEAR(dx_jit[i], dx_int[i], 1e-12) << "d/dx at " << i;
    EXPECT_NEAR(dy_jit[i], dy_int[i], 1e-12) << "d/dy at " << i;
  }
}

// Compiling at all is one question and how to compile is another, so the lane
// width has to be reachable from the Equation rather than only from
// jit::Compiler -- and choosing it must not change a bit of the answer.
TEST(RtEquation, OptionsReachTheCompilerAndDoNotChangeTheAnswer) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto eq = ddx::rt::equation(x * log(x) + y * exp(x * y) + sqrt(x + y));

  constexpr std::size_t n = 16;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.02 * static_cast<double>(i);
    cy[i] = 0.7 - 0.01 * static_cast<double>(i);
  }
  const double *const columns[]{cx.data(), cy.data()};

  const auto run = [&](const ddx::jit::Options &opt) {
    std::vector<double> f(n), dx(n), dy(n);
    double *const values[]{f.data()};
    double *const partials[]{dx.data(), dy.data()};
    eq.options(opt);
    const bool kernel = eq.uses_kernel();
    *eq.jacobian(columns, values, partials, n);
    return std::tuple{f, dx, dy, kernel};
  };

  const ddx::jit::Options scalar{.lanes = 1};

  const auto [f0, dx0, dy0, k0] = run({});
  const auto [f1, dx1, dy1, k1] = run(scalar);

  EXPECT_EQ(eq.options().lanes, 1u) << "the setter did not take";
  EXPECT_TRUE(k0 && k1) << "an option refused the compile outright";

  // Bit-exact: a lane is its own IEEE operation, so the width changes what is
  // scheduled, never what is contracted.
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(f1[i], f0[i]) << "scalar value at " << i;
    EXPECT_EQ(dx1[i], dx0[i]) << "scalar d/dx at " << i;
    EXPECT_EQ(dy1[i], dy0[i]) << "scalar d/dy at " << i;
  }
}

// Background takes the compile off the critical path: the block sweep answers
// until the kernel lands, and what lands is what Compile would have blocked for.
TEST(RtEquation, BackgroundCompilesBehindTheInterpreter) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto eq = ddx::rt::equation(x * log(x) + y * exp(x * y) + sqrt(x + y));

  constexpr std::size_t n = 16;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.02 * static_cast<double>(i);
    cy[i] = 0.7 - 0.01 * static_cast<double>(i);
  }
  const double *const columns[]{cx.data(), cy.data()};

  std::vector<double> f0(n), dx0(n), dy0(n), f1(n), dx1(n), dy1(n), f2(n),
      dx2(n), dy2(n);
  const auto into = [&](std::vector<double> &f, std::vector<double> &dx,
                        std::vector<double> &dy) {
    double *const values[]{f.data()};
    double *const partials[]{dx.data(), dy.data()};
    ASSERT_TRUE(eq.jacobian(columns, values, partials, n).has_value());
  };

  // Whichever path is ready at this instant -- the point is that it answers.
  eq.options({.backend = ddx::rt::Backend::Background});
  into(f0, dx0, dy0);

  ASSERT_TRUE(eq.wait_for_kernel()) << "the background compile never landed";
  EXPECT_TRUE(eq.uses_kernel()) << "a landed kernel was not adopted";
  into(f1, dx1, dy1);

  // The same compile, blocked for instead of waited on.
  eq.options({.backend = ddx::rt::Backend::Compile});
  ASSERT_TRUE(eq.uses_kernel());
  into(f2, dx2, dy2);

  for (std::size_t i = 0; i < n; ++i) {
    // Bit-exact: one graph, one Options, so one kernel.
    EXPECT_DOUBLE_EQ(f1[i], f2[i]) << "landed value at " << i;
    EXPECT_DOUBLE_EQ(dx1[i], dx2[i]) << "landed d/dx at " << i;
    EXPECT_DOUBLE_EQ(dy1[i], dy2[i]) << "landed d/dy at " << i;
    // The first call may have been the sweep, which contracts differently.
    EXPECT_NEAR(f0[i], f2[i], 1e-12) << "value before landing at " << i;
    EXPECT_NEAR(dx0[i], dx2[i], 1e-12) << "d/dx before landing at " << i;
    EXPECT_NEAR(dy0[i], dy2[i], 1e-12) << "d/dy before landing at " << i;
  }
}

// Dropping an equation mid-compile abandons the result rather than waiting for
// it.  std::async's future joins in its destructor, which would put the whole
// compile back on the critical path of whatever let the equation go.
TEST(RtEquation, DroppingAMidFlightBackgroundCompileDoesNotBlock) {
  // Wide enough that the compile is milliseconds, not microseconds: a compile
  // that finishes first would pass this test without proving anything.
  ddx::rt::Builder<> b;
  std::vector<ddx::rt::RTExpression<double>> v;
  for (std::size_t i = 0; i < 48; ++i) {
    v.push_back(ddx::rt::var(b, "x" + std::to_string(i)));
  }
  auto f = v[0] * log(v[0]);
  for (std::size_t i = 1; i < v.size(); ++i) {
    f = f + v[i] * log(v[i]) + exp(v[i - 1] * v[i]);
  }

  // Calibrated against this machine rather than a constant: a fixed threshold
  // above the compile would pass whether or not the future joined.
  const auto blocking = [&] {
    auto eq = ddx::rt::equation(f);
    eq.options({.backend = ddx::rt::Backend::Compile});
    const auto start = std::chrono::steady_clock::now();
    EXPECT_TRUE(eq.uses_kernel());
    return std::chrono::steady_clock::now() - start;
  }();

  const auto teardown = [&] {
    auto eq = ddx::rt::equation(f);
    eq.options({.backend = ddx::rt::Backend::Background});
    (void)eq.uses_kernel(); // launches the compile, adopts nothing yet

    const auto start = std::chrono::steady_clock::now();
    { const auto dropped = std::move(eq); }
    return std::chrono::steady_clock::now() - start;
  }();

  const auto us = [](std::chrono::steady_clock::duration d) {
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
  };
  // A join would put the whole compile here; abandoning it is ~a microsecond.
  EXPECT_LT(teardown * 4, blocking)
      << "teardown " << us(teardown) << "us against a " << us(blocking)
      << "us compile, so the future joined";
}

// Many threads on one Equation, mixing both lanes.  std::thread rather than
// std::jthread: libc++ shipped jthread late and behind _LIBCPP_ENABLE_EXPERIMENTAL.
//
// Background, deliberately: it is the path that abandons compiles mid-flight
// when an equation goes away, which is what compile_async orders its statics
// against.
TEST(RtEquation, ConcurrentConstCallsAgreeWithOneThread) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto eq = ddx::rt::equation(x * log(x) + y * exp(x * y) + sqrt(x + y));
  eq.options({.backend = ddx::rt::Backend::Background});

  constexpr std::size_t n = 32;
  std::vector<double> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.02 * static_cast<double>(i);
    cy[i] = 0.7 - 0.01 * static_cast<double>(i);
  }
  const double *const columns[]{cx.data(), cy.data()};

  const std::size_t hcols = *eq.hessian_columns();
  const auto once = [&] {
    std::vector<double> f(n), dx(n), dy(n);
    std::vector<std::vector<double>> hs(hcols, std::vector<double>(n));
    std::vector<double *> hp(hcols);
    for (std::size_t c = 0; c < hcols; ++c) {
      hp[c] = hs[c].data();
    }
    double *const values[]{f.data()};
    double *const partials[]{dx.data(), dy.data()};
    EXPECT_TRUE(eq.hessian(columns, values, partials, hp, n).has_value());
    return std::tuple{f, dx, dy, hs};
  };
  const auto expected = once();

  std::vector<std::thread> racers;
  std::vector<decltype(once())> got(8);
  for (std::size_t t = 0; t < got.size(); ++t) {
    racers.emplace_back([&, t] {
      // Interleaved with the read-only accessors, which take the same locks.
      (void)eq.uses_kernel();
      (void)eq.value_columns();
      (void)eq.jacobian_columns();
      got[t] = once();
    });
  }
  for (auto &r : racers) {
    r.join();
  }

  const auto &[ef, edx, edy, ehs] = expected;
  for (const auto &[f, dx, dy, hs] : got) {
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_NEAR(f[i], ef[i], 1e-12) << "value at " << i;
      EXPECT_NEAR(dx[i], edx[i], 1e-12) << "d/dx at " << i;
      EXPECT_NEAR(dy[i], edy[i], 1e-12) << "d/dy at " << i;
      for (std::size_t c = 0; c < hs.size(); ++c) {
        EXPECT_NEAR(hs[c][i], ehs[c][i], 1e-12) << "h" << c << " at " << i;
      }
    }
  }
}

// Two equations over ONE borrowed Builder, each asked for a Hessian from its own
// thread.  rt::hessian appends to the arena, so before the sweep moved into the
// constructor no per-Equation lock could have made this safe.
TEST(RtEquation, TwoEquationsSharingAnArenaDoNotRaceIt) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  auto first = ddx::rt::equation(x * log(x) + y * y);
  auto second = ddx::rt::equation(y * exp(x) + x * x);

  const auto at = std::array{0.4, 0.9};
  const auto expected_first = *first.hessian(at);
  const auto expected_second = *second.hessian(at);

  std::vector<double> a, c;
  std::thread one{[&] { a = *first.hessian(at); }};
  std::thread two{[&] { c = *second.hessian(at); }};
  one.join();
  two.join();

  EXPECT_EQ(a, expected_first);
  EXPECT_EQ(c, expected_second);
}

// Interpret must not launch anything, however long one waits.
TEST(RtEquation, InterpretNeverCompilesInTheBackground) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  auto eq = ddx::rt::equation(x * log(x));
  eq.options({.backend = ddx::rt::Backend::Interpret});
  EXPECT_FALSE(eq.wait_for_kernel());
  EXPECT_FALSE(eq.uses_kernel());
}
#endif
