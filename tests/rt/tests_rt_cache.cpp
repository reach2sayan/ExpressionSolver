#include "rt/cache.hpp"
#include "rt/equation.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <tuple>
#include <vector>

// What the cache is for is that nothing about an answer changes: every test
// here is an equation given one against the same equation given none, and the
// comparison is == on doubles and not a tolerance.  A remembered call that
// came back nearly right would be a bug this file could not see.
namespace {

using ddx::rt::var;

// A cache of a caller's own, which is the whole of what the concept asks for:
// somewhere to put one call per lane.  It counts what the mixin asked it for,
// so a test can say the path was taken and not merely that the answer was
// right.
struct Tally {
  struct Counts {
    std::size_t reads = 0;
    std::size_t served = 0;
    std::size_t writes = 0;
    std::size_t commits = 0;
  };

  // Shared, so a copy into the equation still counts into the test's own.
  std::shared_ptr<Counts> counts = std::make_shared<Counts>();
  ddx::rt::LastValue<double> kept;

  class WriteLease {
  public:
    WriteLease(ddx::rt::LastValue<double>::WriteLease lease, Counts &counts)
        : lease_(std::move(lease)), counts_(&counts) {}

    [[nodiscard]] explicit operator bool() const noexcept {
      return static_cast<bool>(lease_);
    }
    [[nodiscard]] bool filled() const noexcept { return lease_.filled(); }
    [[nodiscard]] ddx::rt::Recorded<double> entry() noexcept {
      return lease_.entry();
    }
    void commit() noexcept {
      ++counts_->commits;
      lease_.commit();
    }

  private:
    ddx::rt::LastValue<double>::WriteLease lease_;
    Counts *counts_;
  };

  [[nodiscard]] auto read(ddx::rt::Want want,
                          const ddx::rt::Extent &extent) const {
    ++counts->reads;
    auto lease = kept.read(want, extent);
    counts->served += static_cast<std::size_t>(lease && lease.filled());
    return lease;
  }
  [[nodiscard]] WriteLease write(ddx::rt::Want want,
                                 const ddx::rt::Extent &extent) const {
    auto lease = kept.write(want, extent);
    counts->writes += static_cast<std::size_t>(static_cast<bool>(lease));
    return WriteLease{std::move(lease), *counts};
  }
  void clear() const { kept.clear(); }
};

static_assert(ddx::rt::CValueCache<Tally, double>);
static_assert(ddx::rt::CValueCache<ddx::rt::LastValue<double>, double>);

// Three symbols that reach different parts of the graph, so a point moving one
// of them leaves most of it alone -- which is the case the amendment exists
// for and the one most likely to be wrong.
constexpr auto model = [] {
  const auto x = var("x");
  const auto y = var("y");
  const auto z = var("z");
  return exp(x * y) + sin(z) * x + z * z * z * y + log(x + 4.0);
};

// Every per-point spelling at one point, as one comparable value.
[[nodiscard]] auto everything(const auto &eq, const std::vector<double> &at) {
  const std::vector<double> direction{1.0, -0.5, 0.25};
  const std::vector<double> weights{2.0};
  return std::tuple{*eq.evaluate(at),       *eq.jacobian(at),
                    *eq.gradient(at),       *eq.hessian(at),
                    *eq.hvp(direction, at), *eq.jvp(direction, at),
                    *eq.vjp(weights, at)};
}

TEST(RtCache, ARememberedPointAnswersInTheSameBits) {
  const auto cold = ddx::rt::equation(model);
  const auto warm = ddx::rt::equation(model, ddx::rt::LastValue{});

  std::mt19937 rng{20260830};
  std::uniform_real_distribution<double> spread{-2.0, 2.0};
  std::vector<double> at{0.5, 1.25, 2.0};
  for (std::size_t call = 0; call < 200; ++call) {
    // One symbol, then all of them, then none: an amendment, a fill, and a
    // point already answered.
    if (call % 3 == 1) {
      at[call % at.size()] = spread(rng);
    } else if (call % 3 == 2) {
      std::ranges::generate(at, [&] { return spread(rng); });
    }
    EXPECT_EQ(everything(cold, at), everything(warm, at)) << "call " << call;
  }
}

TEST(RtCache, ZeroAndMinusZeroAreDifferentPoints) {
  const auto reciprocal = [] { return 1.0 / var("x"); };
  const auto warm = ddx::rt::equation(reciprocal, ddx::rt::LastValue{});
  EXPECT_EQ(*warm.evaluate(std::vector{0.0}),
            std::numeric_limits<double>::infinity());
  EXPECT_EQ(*warm.evaluate(std::vector{-0.0}),
            -std::numeric_limits<double>::infinity());
}

TEST(RtCache, APointOfNaNsIsAskedAgainAndNotSweptAgain) {
  const auto square = [] { return var("x") * var("x"); };
  const auto warm = ddx::rt::equation(square, Tally{});
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan(*warm.evaluate(std::vector{nan})));
  EXPECT_TRUE(std::isnan(*warm.evaluate(std::vector{nan})));
}

TEST(RtCache, ACacheOfOnesOwnIsAskedAndFilled) {
  Tally tally;
  const auto warm = ddx::rt::equation(model, tally);
  const std::vector<double> at{0.5, 1.25, 2.0};

  (void)warm.evaluate(at);
  EXPECT_EQ(tally.counts->reads, 1u);
  EXPECT_EQ(tally.counts->served, 0u) << "nothing to serve on a first call";
  EXPECT_EQ(tally.counts->writes, 1u);
  EXPECT_EQ(tally.counts->commits, 1u);

  // The same point again: the shared lock answers it and nothing is written.
  (void)warm.evaluate(at);
  EXPECT_EQ(tally.counts->reads, 2u);
  EXPECT_EQ(tally.counts->served, 1u);
  EXPECT_EQ(tally.counts->writes, 1u);
  EXPECT_EQ(tally.counts->commits, 1u);

  // Another lane is another slot, so it has its own first call.
  (void)warm.gradient(at);
  EXPECT_EQ(tally.counts->writes, 2u);
  EXPECT_EQ(tally.counts->commits, 2u);
}

TEST(RtCache, ABatchIsNotRemembered) {
  Tally tally;
  const auto warm = ddx::rt::equation(model, tally);
  const auto cold = ddx::rt::equation(model);

  constexpr std::size_t n = 16;
  std::vector<double> cx(n), cy(n), cz(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto k = static_cast<double>(i);
    cx[i] = 0.3 + 0.02 * k;
    cy[i] = 0.7 - 0.01 * k;
    cz[i] = 0.1 + 0.03 * k;
  }
  const double *const columns[]{cx.data(), cy.data(), cz.data()};
  const auto batch = [&columns](const auto &eq) {
    std::vector<double> f(n), g(n * 3);
    double *const values[]{f.data()};
    double *const partials[]{g.data(), g.data() + n, g.data() + 2 * n};
    EXPECT_TRUE(eq.jacobian(columns, values, partials, n).has_value());
    return std::pair{f, g};
  };
  EXPECT_EQ(batch(cold), batch(warm));
  EXPECT_EQ(tally.counts->reads, 0u) << "a batch never asks";
}

TEST(RtCache, ASystemIsRememberedPerLaneToo) {
  const auto system = [] {
    const auto x = var("x");
    const auto y = var("y");
    return std::array{x * x + y, exp(x) * y};
  };
  const auto cold = ddx::rt::equation(system);
  const auto warm = ddx::rt::equation(system, ddx::rt::LastValue{});
  for (const double y : {1.0, 1.0, 2.0, 2.0, 1.0}) {
    const std::vector<double> at{0.75, y};
    EXPECT_EQ(*cold.evaluate(at), *warm.evaluate(at));
    EXPECT_EQ(*cold.jacobian(at), *warm.jacobian(at));
    EXPECT_EQ(*cold.gradient(at), *warm.gradient(at));
  }
}

// The try_lock makes hit counts a race by design, so nothing here asserts on
// them: what is deterministic is that every thread gets the cold answer.
TEST(RtCache, ConcurrentCallsAgreeWithTheColdAnswer) {
  const auto cold = ddx::rt::equation(model);
  const auto warm = ddx::rt::equation(model, ddx::rt::LastValue{});
  const std::vector<double> shared{0.5, 1.25, 2.0};

  constexpr std::size_t threads = 8;
  std::barrier start{static_cast<std::ptrdiff_t>(threads)};
  std::vector<std::thread> racers;
  std::vector<bool> agreed(threads, false);
  for (std::size_t t = 0; t < threads; ++t) {
    racers.emplace_back([&, t] {
      // Half at one point, so readers overlap; half at their own, so writers
      // contend and the losers sweep.
      const std::vector<double> at =
          t % 2 == 0
              ? shared
              : std::vector<double>{0.5 + static_cast<double>(t), 1.25, 2.0};
      start.arrive_and_wait();
      bool same = true;
      for (int i = 0; i < 50; ++i) {
        same = same && everything(cold, at) == everything(warm, at);
      }
      agreed[t] = same;
    });
  }
  for (auto &racer : racers) {
    racer.join();
  }
  EXPECT_EQ(std::ranges::count(agreed, true), threads);
}

TEST(RtCache, NoCacheCostsNothing) {
  static_assert(sizeof(ddx::impl::Equation<ddx::rt::RTExpression<double>>) ==
                sizeof(ddx::impl::Equation<ddx::rt::RTExpression<double>,
                                           ddx::rt::detail::NoCache<double>>));
}

#ifdef DDX_HAS_JIT
// codegen decides the arithmetic, so a remembered call is about the graph that
// answered it and no other.
TEST(RtCache, ChangingTheBackendForgets) {
  auto warm = ddx::rt::equation(model, ddx::rt::LastValue{});
  const auto cold = ddx::rt::equation(model);
  const std::vector<double> at{0.5, 1.25, 2.0};

  EXPECT_EQ(*cold.jacobian(at), *warm.jacobian(at));
  warm.options({.backend = ddx::rt::Backend::Compile});
  EXPECT_TRUE(warm.wait_for_kernel());
  auto compiled = ddx::rt::equation(model);
  compiled.options({.backend = ddx::rt::Backend::Compile});
  EXPECT_TRUE(compiled.wait_for_kernel());
  // Twice: the first fills the slot from the kernel, the second is served.
  EXPECT_EQ(*compiled.jacobian(at), *warm.jacobian(at));
  EXPECT_EQ(*compiled.jacobian(at), *warm.jacobian(at));
}

// A kernel computes its whole graph, so a lane it answers keeps no tape and
// every moved point is a fresh call -- which must still be the right one.
TEST(RtCache, ACompiledLaneIsStillRemembered) {
  auto warm = ddx::rt::equation(model, ddx::rt::LastValue{});
  auto cold = ddx::rt::equation(model);
  warm.options({.backend = ddx::rt::Backend::Compile});
  cold.options({.backend = ddx::rt::Backend::Compile});
  EXPECT_TRUE(warm.wait_for_kernel());
  EXPECT_TRUE(cold.wait_for_kernel());
  for (const double z : {2.0, 2.0, 3.0, 3.0, 2.0}) {
    const std::vector<double> at{0.5, 1.25, z};
    EXPECT_EQ(*cold.jacobian(at), *warm.jacobian(at));
    EXPECT_EQ(*cold.hessian(at), *warm.hessian(at));
  }
}
#endif

} // namespace
