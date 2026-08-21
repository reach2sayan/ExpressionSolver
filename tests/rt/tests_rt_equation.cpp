#include "ddx.hpp"
#include "rt/bridge.hpp"
#include "rt/equation.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numbers>
#include <ranges>
#include <span>
#include <vector>

// ===========================================================================
// One Equation (include/rt/equation.hpp)
//
// `Equation` is one name for both kinds of expression: the compile-time tree
// and the runtime graph.  The specialisation is keyed on the RTExpression<T>
// *pattern* rather than on a constraint, because a constraint over
// <TFirst, TRest...> has the same pattern as the compile-time one and neither
// constraint subsumes the other -- that would be ambiguous, not overloaded.
//
// The bridge is what makes these tests possible: to_graph lowers a typed
// expression, so the same function can be asked the same question both ways.
// ===========================================================================

namespace {
constexpr auto sx = ddx::var<"x">;
constexpr auto sy = ddx::var<"y">;

// constexpr, never consteval (NOTES.md): one implementation serves a
// static_assert and a value read at run time.  A runtime graph exists for the
// second, so this proves the first still works rather than the other way round.
consteval double slope_at_two() {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto eq = *ddx::rt::equation(x * x + 3.0 * x);
  return (*eq.gradient(2.0))[0];
}
static_assert(slope_at_two() == 7.0);

TEST(RtEquation, MatchesTheCompileTimeEquation) {
  const auto f = exp(sx) * sin(sy) + sx * sy;
  const ddx::Equation compile_time{f};

  ddx::rt::Builder<> b;
  const auto runtime = *ddx::rt::equation(ddx::rt::to_graph(b, f));

  EXPECT_EQ(runtime.arity(), 2u);
  EXPECT_NEAR(*runtime.evaluate(1.3, 0.7), compile_time.evaluate(1.3, 0.7),
              1e-12);

  const auto want = compile_time.gradient(1.3, 0.7);
  const auto got = *runtime.gradient(1.3, 0.7);
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
  const auto eq = *ddx::rt::equation(log(x) * sqrt(y));

  const auto positional = *eq.gradient(1.3, 0.7);
  const auto by_range = *eq.gradient(std::array{1.3, 0.7});
  const auto by_name = *eq.gradient(ddx::named<"y">(0.7), ddx::named<"x">(1.3));

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
  const auto eq = *ddx::rt::equation(x * y + sin(x), exp(x) - y * y);

  const auto J = *eq.jacobian(1.3, 0.7);
  ASSERT_EQ(J.size(), 4u); // 2 functions x 2 symbols, row-major

  // Row k is the gradient of function k, which the compile-time side agrees on.
  const auto row0 = ddx::Equation{sx * sy + sin(sx)}.gradient(1.3, 0.7);
  const auto row1 = ddx::Equation{exp(sx) - sy * sy}.gradient(1.3, 0.7);
  EXPECT_NEAR(J[0], row0[0], 1e-12);
  EXPECT_NEAR(J[1], row0[1], 1e-12);
  EXPECT_NEAR(J[2], row1[0], 1e-12);
  EXPECT_NEAR(J[3], row1[1], 1e-12);
}

TEST(RtEquation, MultipleOutputsEvaluateToAVector) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto eq = *ddx::rt::equation(x * x, sin(x));
  const auto v = *eq.evaluate(0.5);
  ASSERT_EQ(v.size(), 2u);
  EXPECT_DOUBLE_EQ(v[0], 0.25);
  EXPECT_NEAR(v[1], std::sin(0.5), 1e-15);
}

TEST(RtEquation, AWrongSizedPointIsRejected) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = *ddx::rt::equation(x + y);
  EXPECT_EQ(eq.gradient(std::array{1.0}).error().code, ddx::errc::wrong_arity);
  EXPECT_EQ(eq.gradient(ddx::named<"z">(1.0)).error().code,
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
  const auto eq = *ddx::rt::equation(ddx::rt::to_graph(b, f));
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

  const auto coupled = *ddx::rt::equation(x * y + y * z + x * z);
  EXPECT_EQ(coupled.hessian_colors(), 3u) << "every pair couples";

  ddx::rt::Builder<> b2;
  const auto a = var(b2, "a");
  const auto c = var(b2, "c");
  const auto d = var(b2, "d");
  const auto separable = *ddx::rt::equation(sin(a) + exp(c) + d * d);
  EXPECT_EQ(separable.hessian_colors(), 1u) << "nothing couples: one sweep";
}

} // namespace

namespace {

// A Builder is an arena, not something a caller should have to name or keep
// alive.  equation() makes one, hands out symbols from it, and moves it into
// the result -- so the Equation owns exactly what it needs.
TEST(RtEquation, TheFactoryHidesTheArena) {
  const auto eq = *ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return exp(x) * sin(y);
  });

  EXPECT_EQ(eq.arity(), 2u);
  const auto g = *eq.gradient(1.0, 2.0);
  ASSERT_EQ(g.size(), 2u);
  EXPECT_NEAR(*eq.evaluate(1.0, 2.0), std::exp(1.0) * std::sin(2.0), 1e-12);
  EXPECT_NEAR(g[0], std::exp(1.0) * std::sin(2.0), 1e-12);
  EXPECT_NEAR(g[1], std::exp(1.0) * std::cos(2.0), 1e-12);
}

// The point of owning it: an Equation built this way can be returned, stored
// and passed around.  One whose arena the caller holds cannot.
TEST(RtEquation, AnOwningEquationOutlivesItsScope) {
  const auto make = [](double c) {
    return *ddx::rt::equation([c] {
      const auto x = ddx::rt::var("x");
      return c * x * x;
    });
  };

  const auto eq = make(3.0);
  EXPECT_NEAR(*eq.evaluate(2.0), 12.0, 1e-12);
  EXPECT_NEAR((*eq.gradient(2.0))[0], 12.0, 1e-12); // d(3x^2)/dx = 6x
  EXPECT_NEAR((*eq.hessian(2.0))[0], 6.0, 1e-12);
}

// Same answers whichever backend ran: with the JIT this goes through a compiled
// kernel, without it through the interpreter, and the call site is identical.
TEST(RtEquation, BatchAgreesWithPerPoint) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = *ddx::rt::equation(exp(x) * sin(y) + x * y);

  constexpr std::size_t n = 8;
  std::vector<double> cx(n), cy(n), f(n), dx(n), dy(n);
  for (std::size_t i = 0; i < n; ++i) {
    cx[i] = 0.3 + 0.1 * static_cast<double>(i);
    cy[i] = 1.0 + 0.2 * static_cast<double>(i);
  }
  const double *const columns[]{cx.data(), cy.data()};
  double *const values[]{f.data()};
  double *const partials[]{dx.data(), dy.data()};

  *eq.gradient(columns, values, partials, n);

  for (std::size_t i = 0; i < n; ++i) {
    const auto want = *eq.gradient(cx[i], cy[i]);
    EXPECT_NEAR(f[i], *eq.evaluate(cx[i], cy[i]), 1e-12) << "point " << i;
    EXPECT_NEAR(dx[i], want[0], 1e-12) << "point " << i;
    EXPECT_NEAR(dy[i], want[1], 1e-12) << "point " << i;
  }
}

TEST(RtEquation, BatchHessianFillsTheCompressedColumns) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = *ddx::rt::equation(exp(x) * sin(y) + x * x * y);

  constexpr std::size_t n = 4;
  const std::size_t columns = eq.hessian_columns();
  ASSERT_GT(columns, 0u);

  std::vector<double> cx(n, 0.7), cy(n, 1.1), f(n), dx(n), dy(n);
  std::vector<std::vector<double>> blocks(columns, std::vector<double>(n));
  const auto block_ptrs =
      blocks | std::views::transform([](auto &c) { return c.data(); }) |
      std::ranges::to<std::vector<double *>>();
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
  const auto eq = *ddx::rt::equation(x * y);

  constexpr std::size_t n = 3;
  std::vector<double> cx(n, 2.0), cy(n, 3.0), f(n), dx(n), dy(n);

  const std::vector<const double *> as_vector{cx.data(), cy.data()};
  const double *const as_array[]{cx.data(), cy.data()};
  std::vector<double *> outputs{f.data()};
  std::vector<double *> partials{dx.data(), dy.data()};

  *eq.gradient(as_vector, outputs, partials, n);
  EXPECT_DOUBLE_EQ(f[0], 6.0);
  EXPECT_DOUBLE_EQ(dx[0], 3.0);

  std::ranges::fill(f, 0.0);
  *eq.gradient(as_array, outputs, partials, n);
  EXPECT_DOUBLE_EQ(f[0], 6.0);

  *eq.gradient(std::span{as_vector}, std::span{outputs}, std::span{partials},
               n);
  EXPECT_DOUBLE_EQ(f[0], 6.0);
}

// A wrong column count is silent memory corruption if it reaches the kernel;
// the ABI has no way to notice.  This is the layer that can.
TEST(RtEquation, AMismatchedColumnCountIsRejected) {
  ddx::rt::Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto eq = *ddx::rt::equation(x * y);

  std::vector<double> cx(2, 1.0), cy(2, 1.0), f(2), dx(2), dy(2);
  const std::vector<const double *> xs{cx.data(), cy.data()};
  std::vector<double *> values{f.data()};
  std::vector<double *> partials{dx.data(), dy.data()};
  EXPECT_TRUE(eq.gradient(xs, values, partials, 2).has_value());

  // One partial per symbol, so one column is one too few.
  const std::vector<double *> one_partial{dx.data()};
  EXPECT_EQ(eq.gradient(xs, values, one_partial, 2).error().code,
            ddx::errc::wrong_column_count);

  // One value column for one function, so two is one too many.
  const std::vector<double *> two_values{f.data(), dy.data()};
  EXPECT_EQ(eq.gradient(xs, two_values, partials, 2).error().code,
            ddx::errc::wrong_column_count);

  // And the point itself has to name every symbol.
  const std::vector<const double *> short_point{cx.data()};
  EXPECT_EQ(eq.gradient(short_point, values, partials, 2).error().code,
            ddx::errc::wrong_column_count);
}

} // namespace

namespace {

// Symbols come from equation(); no arena is ever named.
TEST(RtEquation, TheArenaIsInvisible) {
  const auto eq = *ddx::rt::equation([] {
    const auto x = ddx::rt::var("x");
    const auto y = ddx::rt::var("y");
    return x * y + sin(x);
  });
  EXPECT_EQ(eq.arity(), 2u);
  EXPECT_NEAR(*eq.evaluate(2.0, 3.0), 6.0 + std::sin(2.0), 1e-12);

  // Naming one outside a model has no arena to go in.  var() cannot answer with
  // an error without costing every model lambda a dereference, so it poisons
  // the expression instead: the mistake survives every operator applied to it
  // and surfaces once, here, rather than as a graph with a zero where a symbol
  // should be.
  const auto stray = ddx::rt::var("stray");
  EXPECT_TRUE(stray.poisoned());
  EXPECT_TRUE((stray * 2.0 + 1.0).poisoned());
  EXPECT_EQ(ddx::rt::equation(stray * 2.0).error().code, ddx::errc::no_arena);
}

// A system has one Hessian per output -- the shape Equation::hessian returns as
// nd_stack_t<S, m, n, 2> on the compile-time side.
TEST(RtEquation, HessianOfASystemIsOneBlockPerOutput) {
  const auto eq = *ddx::rt::equation([] {
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
  const auto eq = *ddx::rt::equation([] {
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
      *ddx::rt::equation([] { return ddx::rt::var("x") * ddx::rt::var("y"); });
  EXPECT_EQ(two.univariate_derivative<1>(1.0).error().code,
            ddx::errc::not_univariate);
}

} // namespace
