#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

#include "expression_differentiator.hpp"
#include "minimizer/minimizer.hpp"

// Tolerance matching NR's default for Golden (3e-8 * |xmin|)
static constexpr double kTol = 1e-5;

// ─────────────────────────────────────────────────────────────
// Bracketmethod tests
// ─────────────────────────────────────────────────────────────

TEST(Bracketmethod, QuadraticBrackets) {
    // f(x) = (x - 3)^2  — minimum at x = 3
    auto x   = diff::Variable<double, 'x'>{0.0};
    auto f   = (x - diff::Constant<double>{3.0}) * (x - diff::Constant<double>{3.0});

    diff::min::Bracketmethod bm{f};
    bm.bracket(0.0, 1.0);

    // After bracketing, bx must be strictly between ax and cx with f(bx) < f(ax) and f(bx) < f(cx)
    EXPECT_LT(bm.fb, bm.fa);
    EXPECT_LT(bm.fb, bm.fc);
    // The minimum (x=3) must lie within [ax, cx]
    double lo = std::min(bm.ax, bm.cx);
    double hi = std::max(bm.ax, bm.cx);
    EXPECT_LE(lo, 3.0);
    EXPECT_GE(hi, 3.0);
}

TEST(Bracketmethod, SingleCycleConverges) {
    // f(x) = x^4 - 14x^3 + 60x^2 - 70x — multiple features; bracket from (0,1)
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = x * x * x * x
           - diff::Constant<double>{14.0} * x * x * x
           + diff::Constant<double>{60.0} * x * x
           - diff::Constant<double>{70.0} * x;

    diff::min::Bracketmethod bm{f};
    EXPECT_NO_THROW(bm.bracket(0.0, 1.0));
    EXPECT_LT(bm.fb, bm.fa);
    EXPECT_LT(bm.fb, bm.fc);
}

// ─────────────────────────────────────────────────────────────
// Golden section search tests
// ─────────────────────────────────────────────────────────────

TEST(Golden, QuadraticMinimum) {
    // f(x) = (x - 2)^2  =>  xmin = 2, fmin = 0
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = (x - diff::Constant<double>{2.0}) * (x - diff::Constant<double>{2.0});

    diff::min::Golden golden{f};
    double xmin = golden.minimize(0.0, 5.0);

    EXPECT_NEAR(xmin,         2.0, kTol);
    EXPECT_NEAR(golden.xmin,  2.0, kTol);
    EXPECT_NEAR(golden.fmin,  0.0, kTol * kTol);
}

TEST(Golden, SineMinimum) {
    // sin(x) local minimum near 3π/2 ≈ 4.71238898.
    // bracket() is an unbounded search; set the triplet manually to confine
    // golden section to the [3, 6] bowl around the local minimum.
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = sin(y);

    diff::min::Golden g{f};
    g.ax = 3.0;
    g.bx = std::numbers::pi * 1.5;   // known minimum — satisfies f(bx) < f(ax,cx)
    g.cx = 6.0;
    g.fa = g.eval_at(g.ax);
    g.fb = g.eval_at(g.bx);
    g.fc = g.eval_at(g.cx);
    double xmin = g.minimize();

    EXPECT_NEAR(xmin, 3.0 * std::numbers::pi / 2.0, kTol);
    EXPECT_NEAR(g.fmin, -1.0, kTol);
}

TEST(Golden, NegativeQuadratic) {
    // f(x) = -(x - 1)^2 + 4  =>  this is a maximum, minimum is at the bracket edges.
    // Instead test f(x) = (x + 1)^2, minimum at x = -1
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = (x + diff::Constant<double>{1.0}) * (x + diff::Constant<double>{1.0});

    diff::min::Golden g{f};
    double xmin = g.minimize(-3.0, 2.0);

    EXPECT_NEAR(xmin,  -1.0, kTol);
    EXPECT_NEAR(g.fmin, 0.0, kTol * kTol);
}

TEST(Golden, CustomTolerance) {
    // Tighter tolerance: 1e-10
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = (x - diff::Constant<double>{7.5}) * (x - diff::Constant<double>{7.5});

    diff::min::Golden g{f, 1.0e-10};
    double xmin = g.minimize(5.0, 10.0);

    EXPECT_NEAR(xmin, 7.5, 1e-7);
}

TEST(Golden, ManualBracketThenMinimize) {
    // Set bracket manually, then call minimize() without ax0/bx0
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = (x - diff::Constant<double>{4.0}) * (x - diff::Constant<double>{4.0});

    diff::min::Golden g{f};
    g.bracket(2.0, 3.0);   // bracket first
    double xmin = g.minimize();  // then minimize

    EXPECT_NEAR(xmin, 4.0, kTol);
}

// ─────────────────────────────────────────────────────────────
// Compile-time / trait tests
// ─────────────────────────────────────────────────────────────

TEST(GoldenTraits, SymbolAutoDeduced) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = x * x;

    // Symbol extracted at compile time — no char template arg needed on Golden
    using Syms = diff::extract_symbols_from_expr_t<decltype(f)>;
    constexpr bool one_var = boost::mp11::mp_size<Syms>::value == 1;
    EXPECT_TRUE(one_var);
}
