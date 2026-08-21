#include "energy_models.hpp"
#include "rt/coupling.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <string>

// ===========================================================================
// Energy functions at scale (tests/rt/energy_models.hpp)
//
// UNIQUAC, Peng-Robinson and MSE over many species: the workload the runtime
// graph exists for.  There is no compile-time twin to check against -- that is
// the point of these models -- so the reference is central differences on the
// interpreter, which shares no code with the reverse sweep.
//
// Two structural facts these tests pin down, both of which affect what the JIT
// is worth on this workload: the gradient stays cheap as species are added,
// and the Hessian does not, because a mixing rule couples every species to
// every other and there is nothing for the colouring to exploit.
// ===========================================================================

namespace {
using ddx::rt::Builder;
using models::RE;

std::vector<RE> species(Builder<> &b, std::size_t n) {
  std::vector<RE> x;
  x.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    x.push_back(var(b, "x" + std::to_string(i)));
  }
  return x;
}

// An equimolar-ish composition summing to one.
std::vector<double> composition(std::size_t n) {
  std::vector<double> pt(n);
  for (std::size_t i = 0; i < n; ++i) {
    pt[i] = (1.0 + 0.3 * static_cast<double>(i % 5)) / static_cast<double>(n);
  }
  const double total = std::accumulate(pt.begin(), pt.end(), 0.0);
  for (double &v : pt) {
    v /= total;
  }
  return pt;
}

// Central differences on the interpreter: an independent path to the same
// number, sharing nothing with the sweep that built the derivative nodes.
void expect_gradient_matches_differences(Builder<> &b, ddx::rt::NodeId root,
                                         const ddx::rt::Gradient &g,
                                         std::vector<double> pt,
                                         double tol = 2e-6) {
  const auto values = ddx::rt::evaluate_all(b, pt);
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const double h = 1e-6 * std::max(1.0, std::abs(pt[i]));
    auto hi = pt, lo = pt;
    hi[i] += h;
    lo[i] -= h;
    const double fd =
        (ddx::rt::evaluate(b, root, hi) - ddx::rt::evaluate(b, root, lo)) /
        (2 * h);
    EXPECT_NEAR(values[g.partial[i]], fd, tol * std::max(1.0, std::abs(fd)))
        << "d/dx" << i;
  }
}

TEST(RtEnergy, UniquacGradient) {
  Builder<> b;
  const auto x = species(b, 8);
  const auto f = models::uniquac(x);
  const auto g = ddx::rt::gradient(b, f.id(b));
  expect_gradient_matches_differences(b, f.id(b), g, composition(8));
}

TEST(RtEnergy, PengRobinsonGradient) {
  Builder<> b;
  auto x = species(b, 6);
  const auto Z = var(b, "Z");
  const auto f = models::peng_robinson(x, Z);
  const auto g = ddx::rt::gradient(b, f.id(b));
  auto pt = composition(6);
  pt.push_back(0.93); // Z is the last symbol
  expect_gradient_matches_differences(b, f.id(b), g, pt);
}

TEST(RtEnergy, MseGradient) {
  Builder<> b;
  const auto x = species(b, 6);
  const auto f = models::mse(x);
  const auto g = ddx::rt::gradient(b, f.id(b));
  expect_gradient_matches_differences(b, f.id(b), g, composition(6));
}

// A trace species is the normal case, not the edge case: an electrolyte model
// is routinely handed 1e-12 mole fractions.  UNIQUAC has x*log(x) and
// log(phi/x) in it, so this is where it would fall over if anywhere.
TEST(RtEnergy, StableDownToTraceMoleFractions) {
  constexpr std::size_t n = 6;
  Builder<> b;
  const auto x = species(b, n);
  const auto f = models::uniquac(x);
  const auto g = ddx::rt::gradient(b, f.id(b));

  double previous_value = 0;
  double previous_trace = 0;
  for (const double trace : {1e-3, 1e-6, 1e-9, 1e-12, 1e-15}) {
    std::vector<double> pt(n, (1.0 - trace) / (n - 1));
    pt[0] = trace;
    const auto values = ddx::rt::evaluate_all(b, pt);

    EXPECT_TRUE(std::isfinite(values[g.value])) << "G at trace " << trace;
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_TRUE(std::isfinite(values[g.partial[i]]))
          << "dG/dx" << i << " at trace " << trace;
    }
    // The trace species' contribution has to vanish *with* its amount: the
    // change per decade is proportional to the trace, not below some fixed
    // tolerance.  Measured, it is about 1.6x the trace -- a cancellation bug
    // shows up as a change that stays put or grows while the trace shrinks.
    if (previous_trace != 0) {
      EXPECT_LT(std::abs(values[g.value] - previous_value),
                100.0 * previous_trace)
          << "G moved too much going from " << previous_trace << " to "
          << trace;
    }
    previous_trace = trace;
    previous_value = values[g.value];
  }
}

TEST(RtEnergy, HessianMatchesDifferencesOfTheGradient) {
  constexpr std::size_t n = 4;
  Builder<> b;
  const auto x = species(b, n);
  const auto f = models::uniquac(x);
  const auto h = ddx::rt::hessian(b, f.id(b));
  const auto pt = composition(n);
  const auto values = ddx::rt::evaluate_all(b, pt);

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double step = 1e-5;
      auto hi = pt, lo = pt;
      hi[j] += step;
      lo[j] -= step;
      const double fd = (ddx::rt::evaluate_all(b, hi)[h.partial[i]] -
                         ddx::rt::evaluate_all(b, lo)[h.partial[i]]) /
                        (2 * step);
      EXPECT_NEAR(values[h.at(i, j)], fd, 1e-4 * std::max(1.0, std::abs(fd)))
          << "H[" << i << "][" << j << "]";
    }
  }
}

// A mixing rule is a double sum over species, so every pair couples and the
// colouring has nothing to exploit.  Asserting it rather than hoping: a
// colouring that came back with fewer colours than this would be wrong, not
// clever, and would silently sum two second derivatives into one cell.
TEST(RtEnergy, MixingRulesGiveADenseHessian) {
  for (const std::size_t n : {4u, 8u}) {
    Builder<> b;
    auto x = species(b, n);
    const auto f = models::peng_robinson(x, RE{0.9}); // Z pinned: n symbols
    const auto rows = ddx::rt::coupling_pattern(b, f.id(b));
    const auto coloring = ddx::rt::color_columns(rows);
    EXPECT_EQ(coloring.count, n) << "n=" << n;

    // Dense means dense: every pair of species couples, so no two columns may
    // share a colour.  Fewer colours than this would be a wrong answer.
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        EXPECT_TRUE(rows[i].test(j))
            << "expected " << i << "," << j << " coupled";
      }
    }
  }
}

// The gradient stays proportional to the expression; the Hessian does not.
// This is the number that decides whether a full Hessian is worth compiling.
TEST(RtEnergy, GradientStaysCheapAsSpeciesAreAdded) {
  std::size_t previous_ratio = 0;
  for (const std::size_t n : {5u, 10u, 20u}) {
    Builder<> b;
    const auto x = species(b, n);
    const auto f = models::uniquac(x);
    const std::size_t built = b.size();
    (void)ddx::rt::gradient(b, f.id(b));
    const std::size_t added = b.size() - built;

    // One reverse sweep, so the gradient costs a small multiple of the
    // function however many species there are -- not a multiple of n.
    EXPECT_LT(added, 3 * built) << "n=" << n;
    const std::size_t ratio = added / built;
    if (previous_ratio != 0) {
      EXPECT_LE(ratio, previous_ratio + 1) << "gradient cost is growing with n";
    }
    previous_ratio = ratio;
  }
}

} // namespace

namespace {

// The other half of the picture.  A finite interaction range makes the Hessian
// banded, and then the colouring is the difference between a handful of sweeps
// and one per species -- which is the whole reason it is there.
TEST(RtEnergySparse, FiniteRangeGivesFewColours) {
  for (const std::size_t n : {20u, 60u, 200u}) {
    Builder<> b;
    const auto s = species(b, n);
    const auto e = models::cluster_expansion(s, 2);
    const auto coloring =
        ddx::rt::color_columns(ddx::rt::coupling_pattern(b, e.id(b)));

    // Range 2 on a ring couples each site to four neighbours, so the column
    // graph has bounded degree and the colour count must not grow with n.
    // Measured at 5 with smallest-last ordering; the bound is there to catch a
    // regression in the ordering, which is worth ~2.4x on this pattern.
    EXPECT_LE(coloring.count, 6u) << "n=" << n;
    EXPECT_LT(coloring.count, n);
  }
}

TEST(RtEnergySparse, BondedChainIsBanded) {
  constexpr std::size_t n = 40;
  Builder<> b;
  const auto r = species(b, n);
  const auto e = models::bonded_chain(r);
  const auto rows = ddx::rt::coupling_pattern(b, e.id(b));

  // A bend term spans three sites, so nothing couples further than two apart.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (rows[i].test(j)) {
        const auto span = i > j ? i - j : j - i;
        EXPECT_LE(span, 2u) << "unexpected coupling " << i << "," << j;
      }
    }
  }
  EXPECT_LE(ddx::rt::color_columns(rows).count, 5u);
}

// Correctness is the thing colouring can silently break, so a sparse Hessian
// gets the same elementwise check the dense one does.
TEST(RtEnergySparse, ColouredHessianIsStillCorrect) {
  // Comfortably larger than the interaction range: a ring of 8 with range 2
  // couples every site to half the others, and there is nothing to save.
  constexpr std::size_t n = 16;
  Builder<> b;
  const auto s = species(b, n);
  const auto e = models::cluster_expansion(s, 2);
  const auto h = ddx::rt::hessian(b, e.id(b));
  EXPECT_LT(h.colors(), n)
      << "a banded pattern should beat one sweep per column";

  const auto pt = composition(n);
  const auto values = ddx::rt::evaluate_all(b, pt);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double step = 1e-5;
      auto hi = pt, lo = pt;
      hi[j] += step;
      lo[j] -= step;
      const double fd = (ddx::rt::evaluate_all(b, hi)[h.partial[i]] -
                         ddx::rt::evaluate_all(b, lo)[h.partial[i]]) /
                        (2 * step);
      EXPECT_NEAR(values[h.at(i, j)], fd, 1e-4 * std::max(1.0, std::abs(fd)))
          << "H[" << i << "][" << j << "]";
    }
  }
}

} // namespace
