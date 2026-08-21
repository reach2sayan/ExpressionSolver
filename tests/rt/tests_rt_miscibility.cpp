#include "ddx.hpp"
#include "energy_models.hpp"
#include "rt/equation.hpp"

#include <gtest/gtest.h>

#include <cmath>

// ===========================================================================
// A binary miscibility gap (models::regular_solution)
//
// This one is worth having because both derivatives have closed forms:
//
//   f(x)  = c x(1-x) + k( x ln x + (1-x) ln(1-x) )
//   f'(½) = 0                       exactly, by symmetry, for every c and k
//   f''   = -2c + k/(x(1-x))        exactly
//
// So the gradient is checked against an exact zero and the Hessian against an
// exact expression, rather than against finite differences. And the *shape*
// the derivatives imply -- one minimum or two -- is a physical claim a solver
// can be made to confirm.
// ===========================================================================

namespace {
using ddx::rt::Builder;

auto free_energy(Builder<> &b, double c, double k) {
  return *ddx::rt::equation(models::regular_solution(var(b, "x"), c, k));
}

constexpr double second_derivative(double x, double c, double k) {
  return -2.0 * c + k / (x * (1.0 - x));
}

// Symmetry is exact, so this is an equality rather than a tolerance.
TEST(RtMiscibility, HalfIsAlwaysStationary) {
  for (const double c : {2.0, 0.4, 0.0, -1.5}) {
    for (const double k : {0.25, -0.25}) {
      Builder<> b;
      EXPECT_DOUBLE_EQ((*free_energy(b, c, k).gradient(0.5))[0], 0.0)
          << "c=" << c << " k=" << k;
    }
  }
}

TEST(RtMiscibility, HessianMatchesTheClosedForm) {
  for (const double c : {2.0, 0.4, -1.5}) {
    for (const double k : {0.25, -0.25}) {
      Builder<> b;
      const auto f = free_energy(b, c, k);
      for (const double x : {0.05, 0.25, 0.5, 0.8}) {
        EXPECT_NEAR((*f.hessian(x))[0], second_derivative(x, c, k), 1e-12)
            << "c=" << c << " k=" << k << " x=" << x;
      }
    }
  }
}

// c > 2k opens the gap: x = 1/2 turns from a minimum into a maximum.
TEST(RtMiscibility, TheGapOpensWhenEnthalpyBeatsEntropy) {
  constexpr double k = 0.25;
  Builder<> b1;
  EXPECT_LT((*free_energy(b1, 1.0, k).hessian(0.5))[0], 0.0)
      << "c > 2k: a maximum";
  Builder<> b2;
  EXPECT_GT((*free_energy(b2, 0.4, k).hessian(0.5))[0], 0.0)
      << "c < 2k: a minimum";

  // k < 0 inverts the curvature at the ends, so no interior gap for any c.
  Builder<> b3;
  const auto inverted = free_energy(b3, 2.0, -k);
  EXPECT_LT((*inverted.hessian(0.02))[0], 0.0) << "concave at the edge";
}

// Newton on f', using f'' as the derivative -- so a wrong Hessian shows up as
// failure to converge, not merely as a wrong number.
TEST(RtMiscibility, NewtonFindsTwoSymmetricMinima) {
  constexpr double c = 1.0;
  constexpr double k = 0.25; // c > 2k, so the gap is open
  Builder<> b;
  const auto f = free_energy(b, c, k);
  ASSERT_LT((*f.hessian(0.5))[0], 0.0) << "no gap to find";

  const auto minimise_from = [&](double start) {
    double at = start;
    for (int iteration = 0; iteration < 100; ++iteration) {
      const double slope = (*f.gradient(at))[0];
      if (std::abs(slope) < 1e-14) {
        break;
      }
      at = std::clamp(at - slope / (*f.hessian(at))[0], 1e-12, 1.0 - 1e-12);
    }
    return at;
  };

  const double lower = minimise_from(0.05);
  const double upper = minimise_from(0.95);

  EXPECT_NEAR((*f.gradient(lower))[0], 0.0, 1e-12);
  EXPECT_NEAR((*f.gradient(upper))[0], 0.0, 1e-12);
  EXPECT_GT((*f.hessian(lower))[0], 0.0)
      << "a minimum, not the central maximum";
  EXPECT_GT((*f.hessian(upper))[0], 0.0);
  EXPECT_LT(lower, 0.5);
  EXPECT_GT(upper, 0.5);

  // The two coexisting compositions are mirror images: the model is symmetric,
  // so this must hold to rounding rather than approximately.
  EXPECT_NEAR(lower + upper, 1.0, 1e-12);
}

} // namespace
