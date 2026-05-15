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
// Brent tests
// ─────────────────────────────────────────────────────────────

TEST(Brent, QuadraticMinimum) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = (x - diff::Constant<double>{2.0}) * (x - diff::Constant<double>{2.0});

    diff::min::Brent b{f};
    double xmin = b.minimize(0.0, 5.0);

    EXPECT_NEAR(xmin,   2.0, kTol);
    EXPECT_NEAR(b.fmin, 0.0, kTol * kTol);
}

TEST(Brent, SineMinimum) {
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = sin(y);

    diff::min::Brent b{f};
    b.ax = 3.0; b.bx = std::numbers::pi * 1.5; b.cx = 6.0;
    b.fa = b.eval_at(b.ax); b.fb = b.eval_at(b.bx); b.fc = b.eval_at(b.cx);
    double xmin = b.minimize();

    EXPECT_NEAR(xmin,   3.0 * std::numbers::pi / 2.0, kTol);
    EXPECT_NEAR(b.fmin, -1.0, kTol);
}

TEST(Brent, QuarticMinimum) {
    // f(x) = (x-1)^4  — flat near minimum, good stress test for parabolic interpolation
    auto x = diff::Variable<double, 'x'>{0.0};
    auto d = x - diff::Constant<double>{1.0};
    auto f = d * d * d * d;

    diff::min::Brent b{f};
    double xmin = b.minimize(0.0, 3.0);

    EXPECT_NEAR(xmin,   1.0, kTol);
    EXPECT_NEAR(b.fmin, 0.0, kTol * kTol);
}

// ─────────────────────────────────────────────────────────────
// LinMin tests
// ─────────────────────────────────────────────────────────────

TEST(LinMin, AxisDirection) {
    // f(x,y) = (x-3)^2 + (y-4)^2; start (0,0), dir (1,0)
    // minimum along x-axis at t=3 → p=(3,0)
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{3.0}) * (x - diff::Constant<double>{3.0})
           + (y - diff::Constant<double>{4.0}) * (y - diff::Constant<double>{4.0});

    diff::min::LinMin lm{f};
    std::array<double, 2> p{0.0, 0.0};
    std::array<double, 2> dir{1.0, 0.0};
    lm.minimize(p, dir);

    EXPECT_NEAR(p[0],   3.0, kTol);
    EXPECT_NEAR(p[1],   0.0, kTol);   // y unchanged
    EXPECT_NEAR(lm.fret, 16.0, kTol); // (3-3)^2 + (0-4)^2 = 16
}

TEST(LinMin, DiagonalDirection) {
    // f(x,y) = (x-3)^2 + (y-4)^2; start (0,0), dir (1,1)
    // minimise (t-3)^2 + (t-4)^2 → t=3.5 → p=(3.5, 3.5)
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{3.0}) * (x - diff::Constant<double>{3.0})
           + (y - diff::Constant<double>{4.0}) * (y - diff::Constant<double>{4.0});

    diff::min::LinMin lm{f};
    std::array<double, 2> p{0.0, 0.0};
    std::array<double, 2> dir{1.0, 1.0};
    lm.minimize(p, dir);

    EXPECT_NEAR(p[0],   3.5, kTol);
    EXPECT_NEAR(p[1],   3.5, kTol);
    EXPECT_NEAR(lm.fret, 0.5, kTol); // (3.5-3)^2 + (3.5-4)^2 = 0.5
}

TEST(LinMin, DirScaledByStep) {
    // dir should be multiplied by xmin after minimize
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{3.0}) * (x - diff::Constant<double>{3.0})
           + (y - diff::Constant<double>{4.0}) * (y - diff::Constant<double>{4.0});

    diff::min::LinMin lm{f};
    std::array<double, 2> p{0.0, 0.0};
    std::array<double, 2> dir{1.0, 1.0};
    lm.minimize(p, dir);

    // dir = xmin * original_dir; p = original_p + dir
    EXPECT_NEAR(dir[0], p[0], kTol);
    EXPECT_NEAR(dir[1], p[1], kTol);
}

// ─────────────────────────────────────────────────────────────
// Powell tests
// ─────────────────────────────────────────────────────────────

TEST(Powell, Bowl2D) {
    // f(x,y) = (x-1)^2 + (y-2)^2  — minimum at (1, 2)
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
           + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

    diff::min::Powell pw{f};
    auto p = pw.minimize({0.0, 0.0});

    EXPECT_NEAR(p[0],   1.0, kTol);
    EXPECT_NEAR(p[1],   2.0, kTol);
    EXPECT_NEAR(pw.fret, 0.0, kTol * kTol);
}

TEST(Powell, Rosenbrock) {
    // f(x,y) = (1-x)^2 + 100*(y-x^2)^2  — minimum at (1,1), fmin=0
    auto x  = diff::Variable<double, 'x'>{0.0};
    auto y  = diff::Variable<double, 'y'>{0.0};
    auto t1 = diff::Constant<double>{1.0} - x;
    auto t2 = y - x * x;
    auto f  = t1 * t1 + diff::Constant<double>{100.0} * t2 * t2;

    diff::min::Powell pw{f, 1e-10};
    auto p = pw.minimize({-1.0, 1.0});

    EXPECT_NEAR(p[0],    1.0, 1e-4);
    EXPECT_NEAR(p[1],    1.0, 1e-4);
    EXPECT_NEAR(pw.fret, 0.0, 1e-6);
}

// ─────────────────────────────────────────────────────────────
// Frprmn (Conjugate Gradient) tests
// ─────────────────────────────────────────────────────────────

TEST(Frprmn, Bowl2D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
           + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

    diff::min::Frprmn cg{f};
    auto p = cg.minimize({0.0, 0.0});

    EXPECT_NEAR(p[0],    1.0, kTol);
    EXPECT_NEAR(p[1],    2.0, kTol);
    EXPECT_NEAR(cg.fret, 0.0, kTol * kTol);
}

TEST(Frprmn, Rosenbrock) {
    auto x  = diff::Variable<double, 'x'>{0.0};
    auto y  = diff::Variable<double, 'y'>{0.0};
    auto t1 = diff::Constant<double>{1.0} - x;
    auto t2 = y - x * x;
    auto f  = t1 * t1 + diff::Constant<double>{100.0} * t2 * t2;

    diff::min::Frprmn cg{f, 1e-10};
    auto p = cg.minimize({-1.0, 1.0});

    EXPECT_NEAR(p[0],    1.0, 1e-4);
    EXPECT_NEAR(p[1],    1.0, 1e-4);
    EXPECT_NEAR(cg.fret, 0.0, 1e-6);
}

TEST(Frprmn, Quadratic3D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto z = diff::Variable<double, 'z'>{0.0};
    auto f = x * x
           + diff::Constant<double>{2.0} * y * y
           + diff::Constant<double>{3.0} * z * z;

    diff::min::Frprmn cg{f};
    auto p = cg.minimize({3.0, 3.0, 3.0});

    EXPECT_NEAR(p[0],    0.0, kTol);
    EXPECT_NEAR(p[1],    0.0, kTol);
    EXPECT_NEAR(p[2],    0.0, kTol);
    EXPECT_NEAR(cg.fret, 0.0, kTol * kTol);
}

TEST(Frprmn, FletcherReeves) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
           + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

    diff::min::Frprmn<decltype(f), diff::min::CGMethod::FletcherReeves> cg{f};
    auto p = cg.minimize({0.0, 0.0});

    EXPECT_NEAR(p[0], 1.0, kTol);
    EXPECT_NEAR(p[1], 2.0, kTol);
}

// ─────────────────────────────────────────────────────────────
// BFGS tests
// ─────────────────────────────────────────────────────────────

TEST(BFGS, Bowl2D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
           + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

    diff::min::BFGS bfgs{f};
    auto p = bfgs.minimize({0.0, 0.0});

    EXPECT_NEAR(p[0],      1.0, kTol);
    EXPECT_NEAR(p[1],      2.0, kTol);
    EXPECT_NEAR(bfgs.fret, 0.0, kTol * kTol);
}

TEST(BFGS, Rosenbrock) {
    auto x  = diff::Variable<double, 'x'>{0.0};
    auto y  = diff::Variable<double, 'y'>{0.0};
    auto t1 = diff::Constant<double>{1.0} - x;
    auto t2 = y - x * x;
    auto f  = t1 * t1 + diff::Constant<double>{100.0} * t2 * t2;

    diff::min::BFGS bfgs{f, 1e-10};
    auto p = bfgs.minimize({-1.0, 1.0});

    EXPECT_NEAR(p[0],      1.0, 1e-4);
    EXPECT_NEAR(p[1],      1.0, 1e-4);
    EXPECT_NEAR(bfgs.fret, 0.0, 1e-6);
}

TEST(BFGS, Quadratic3D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto z = diff::Variable<double, 'z'>{0.0};
    auto f = x * x
           + diff::Constant<double>{2.0} * y * y
           + diff::Constant<double>{3.0} * z * z;

    diff::min::BFGS bfgs{f};
    auto p = bfgs.minimize({3.0, 3.0, 3.0});

    EXPECT_NEAR(p[0],      0.0, kTol);
    EXPECT_NEAR(p[1],      0.0, kTol);
    EXPECT_NEAR(p[2],      0.0, kTol);
    EXPECT_NEAR(bfgs.fret, 0.0, kTol * kTol);
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
