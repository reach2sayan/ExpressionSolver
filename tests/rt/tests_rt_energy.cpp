#include "energy_models.hpp"
#include "rt/coupling.hpp"
#include "rt/interpret.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <format>
#include <numeric>
#include <string>

// There is no compile-time twin for these models, so the reference is central
// differences on the interpreter, which shares no code with the reverse sweep.
// Two structural facts: the Jacobian stays cheap as species are added, and the
// Hessian does not -- a mixing rule couples every species to every other.

namespace {
using ddx::rt::Builder;
using models::RE;

// Zero-padded: a slot is the symbol's place in the alphabet, so x9 has to sort
// before x10 for slot order to be site order -- which is what banded means.
std::vector<RE> species(Builder<> &b, std::size_t n) {
  const auto width = std::to_string(n - 1).size();
  std::vector<RE> x;
  x.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    x.push_back(var(b, std::format("x{:0{}}", i, width)));
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

// Central differences on the interpreter: independent of the sweep.
void expect_jacobian_matches_differences(Builder<> &b, ddx::rt::NodeId root,
                                         const ddx::rt::JacobianRow &g,
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

TEST(RtEnergy, UniquacJacobian) {
  Builder<> b;
  const auto x = species(b, 8);
  const auto f = models::uniquac(x);
  const auto g = ddx::rt::build_jacobian_impl(b, f.id(b));
  expect_jacobian_matches_differences(b, f.id(b), g, composition(8));
}

TEST(RtEnergy, PengRobinsonJacobian) {
  Builder<> b;
  auto x = species(b, 6);
  const auto Z = var(b, "Z");
  const auto f = models::peng_robinson(x, Z);
  const auto g = ddx::rt::build_jacobian_impl(b, f.id(b));
  auto pt = composition(6);
  pt.push_back(0.93); // Z is the last symbol
  expect_jacobian_matches_differences(b, f.id(b), g, pt);
}

TEST(RtEnergy, MseJacobian) {
  Builder<> b;
  const auto x = species(b, 6);
  const auto f = models::mse(x);
  const auto g = ddx::rt::build_jacobian_impl(b, f.id(b));
  expect_jacobian_matches_differences(b, f.id(b), g, composition(6));
}

// 1e-12 mole fractions are routine, and UNIQUAC has x*log(x) and log(phi/x).
TEST(RtEnergy, StableDownToTraceMoleFractions) {
  constexpr std::size_t n = 6;
  Builder<> b;
  const auto x = species(b, n);
  const auto f = models::uniquac(x);
  const auto g = ddx::rt::build_jacobian_impl(b, f.id(b));

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
    // The change per decade must be proportional to the trace (~1.6x it),
    // not merely below a fixed tolerance.
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

TEST(RtEnergy, HessianMatchesDifferencesOfTheJacobian) {
  constexpr std::size_t n = 4;
  Builder<> b;
  const auto x = species(b, n);
  const auto f = models::uniquac(x);
  const auto h = ddx::rt::build_hessian_impl(b, f.id(b));
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

// Every pair couples, so fewer colours than this would silently sum two second
// derivatives into one cell.
TEST(RtEnergy, MixingRulesGiveADenseHessian) {
  for (const std::size_t n : {4u, 8u}) {
    Builder<> b;
    auto x = species(b, n);
    const auto f = models::peng_robinson(x, RE{0.9}); // Z pinned: n symbols
    const auto rows = ddx::rt::coupling_pattern(b, f.id(b));
    const auto coloring = ddx::rt::color_columns(rows);
    EXPECT_EQ(coloring.count, n) << "n=" << n;

    // No two columns may share a colour.
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        EXPECT_TRUE(rows[i].test(j))
            << "expected " << i << "," << j << " coupled";
      }
    }
  }
}

// The Jacobian stays proportional to the expression; the Hessian does not.
TEST(RtEnergy, JacobianStaysCheapAsSpeciesAreAdded) {
  std::size_t previous_ratio = 0;
  for (const std::size_t n : {5u, 10u, 20u}) {
    Builder<> b;
    const auto x = species(b, n);
    const auto f = models::uniquac(x);
    const std::size_t built = b.size();
    (void)ddx::rt::build_jacobian_impl(b, f.id(b));
    const std::size_t added = b.size() - built;

    // One reverse sweep: a small multiple of the function, not of n.
    EXPECT_LT(added, 3 * built) << "n=" << n;
    const std::size_t ratio = added / built;
    if (previous_ratio != 0) {
      EXPECT_LE(ratio, previous_ratio + 1) << "Jacobian cost is growing with n";
    }
    previous_ratio = ratio;
  }
}

} // namespace

namespace {

// A finite interaction range makes the Hessian banded, and the colouring is
// then a handful of sweeps rather than one per species.
TEST(RtEnergySparse, FiniteRangeGivesFewColours) {
  for (const std::size_t n : {20u, 60u, 200u}) {
    Builder<> b;
    const auto s = species(b, n);
    const auto e = models::cluster_expansion(s, 2);
    const auto coloring =
        ddx::rt::color_columns(ddx::rt::coupling_pattern(b, e.id(b)));

    // Bounded degree, so the colour count must not grow with n.  Measured at
    // 5 with smallest-last ordering, which is worth ~2.4x here.
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

// Colouring can silently break correctness, so check elementwise.
TEST(RtEnergySparse, ColouredHessianIsStillCorrect) {
  // Larger than the interaction range: a ring of 8 at range 2 saves nothing.
  constexpr std::size_t n = 16;
  Builder<> b;
  const auto s = species(b, n);
  const auto e = models::cluster_expansion(s, 2);
  const auto h = ddx::rt::build_hessian_impl(b, e.id(b));
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
