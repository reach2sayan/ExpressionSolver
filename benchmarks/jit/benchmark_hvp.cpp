// What H(x)v costs against the Hessian it never forms: one Hessian, one HVP,
// and n HVPs, the third being what makes the first two honest.
//
// Not a Google Benchmark loop: the argument is a scaling exponent, so each cell
// is run at its own n and the table prints the log-log slope against the row
// above.  Wall clock and the minimum over repetitions, per NOTES.md house
// style -- do not re-tune this with callgrind, which is how the deleted
// vector-forward driver was mis-priced.

#include "rt/equation.hpp"
#include "rt/expressions.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <vector>

namespace {

using ddx::rt::Builder;
using RE = ddx::rt::RTExpression<double>;

// Every symbol touches every other, so the coupling graph is complete and the
// colouring buys nothing: colours == n, which is the worst case for the dense
// path and the case a Newton-Krylov solve actually meets.
RE dense_model(std::span<const RE> v) {
  RE acc{0.0};
  for (const auto &a : v) {
    for (const auto &b : v) {
      acc = acc + exp(a * b * 0.01);
    }
  }
  return acc;
}

// A band: each symbol couples only its neighbour, so the colouring finds a
// handful of colours however large n is.  The case where the Hessian wins.
RE banded_model(std::span<const RE> v) {
  RE acc{0.0};
  for (const auto [i, a] : v | std::views::enumerate) {
    acc = acc + sin(a) * sin(a);
    if (static_cast<std::size_t>(i) + 1 < v.size()) {
      acc = acc + a * v[static_cast<std::size_t>(i) + 1];
    }
  }
  return acc;
}

// Enough points that a call's fixed cost is not what is being compared.
constexpr std::size_t kPoints = 256;

using Clock = std::chrono::steady_clock;

[[nodiscard]] double seconds(auto &&f, int reps) {
  double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i < reps; ++i) {
    const auto t0 = Clock::now();
    f();
    const auto t1 = Clock::now();
    best = std::min(
        best, std::chrono::duration<double>(t1 - t0).count());
  }
  return best;
}

// The slope of a log-log line through two cells: what the table is for.
[[nodiscard]] std::string exponent(double prev_n, double prev_t, double n,
                                   double t) {
  if (prev_n <= 0.0 || prev_t <= 0.0 || t <= 0.0) {
    return "     -";
  }
  return std::format("{:6.2f}", std::log(t / prev_t) / std::log(n / prev_n));
}

struct Row {
  std::size_t n = 0;
  std::size_t colours = 0;
  std::size_t hessian_columns = 0;
  std::size_t hvp_columns = 0;
  double hessian_s = 0.0;
  double hvp_s = 0.0;
  double n_hvps_s = 0.0;
};

void run(const char *name, RE (*build)(std::span<const RE>),
         std::span<const std::size_t> sizes) {
  std::printf("\n%s\n", name);
  std::printf("%5s %8s %10s %8s %12s %12s %8s %12s %8s\n", "n", "colours",
              "H cols", "Hv cols", "hessian(s)", "hvp(s)", "exp", "n*hvp(s)",
              "H/hvp");

  Row prev;
  for (const std::size_t n : sizes) {
    Builder<> b;
    std::vector<RE> vars;
    vars.reserve(n);
    for (const std::size_t i : std::views::iota(0uz, n)) {
      vars.push_back(var(b, std::format("x{:04}", i)));
    }
    auto eq = ddx::rt::equation(build(vars));
    if (eq.poisoned()) {
      std::printf("%5zu  poisoned\n", n);
      continue;
    }
    eq.options({.backend = ddx::jit::Backend::Compile});
    (void)eq.wait_for_kernel();

    // A batch, not a point: at one point the answer is per-call overhead, and
    // the two blocks are the same size in that noise.
    std::vector<std::vector<double>> xcols(n, std::vector<double>(kPoints, 0.5));
    std::vector<std::vector<double>> vcols(n, std::vector<double>(kPoints, 1.0));
    std::vector<const double *> xs;
    std::vector<const double *> vs;
    for (const std::size_t j : std::views::iota(0uz, n)) {
      xs.push_back(xcols[j].data());
      vs.push_back(vcols[j].data());
    }

    Row row{.n = n,
            .colours = *eq.hessian_colors(),
            .hessian_columns = *eq.hessian_columns(),
            .hvp_columns = *eq.hvp_columns()};

    std::vector<double> fbuf(kPoints);
    std::vector<double> gbuf(*eq.jacobian_columns() * kPoints);
    std::vector<double> hbuf(row.hessian_columns * kPoints);
    std::vector<double> pbuf(row.hvp_columns * kPoints);
    const auto rows = [kPoints_ = kPoints](std::vector<double> &buf,
                                           std::size_t count) {
      std::vector<double *> out;
      for (const std::size_t k : std::views::iota(0uz, count)) {
        out.push_back(buf.data() + k * kPoints_);
      }
      return out;
    };
    const auto f = rows(fbuf, 1);
    const auto g = rows(gbuf, *eq.jacobian_columns());
    const auto h = rows(hbuf, row.hessian_columns);
    const auto pr = rows(pbuf, row.hvp_columns);

    // Warm both lanes before timing: the first call freezes and compiles.
    (void)eq.hessian(xs, f, g, h, kPoints);
    (void)eq.hvp(xs, vs, f, g, pr, kPoints);

    row.hessian_s =
        seconds([&] { (void)eq.hessian(xs, f, g, h, kPoints); }, 5);
    row.hvp_s = seconds([&] { (void)eq.hvp(xs, vs, f, g, pr, kPoints); }, 5);
    row.n_hvps_s = seconds(
        [&] {
          for ([[maybe_unused]] const std::size_t j :
               std::views::iota(0uz, n)) {
            (void)eq.hvp(xs, vs, f, g, pr, kPoints);
          }
        },
        3);

    std::printf("%5zu %8zu %10zu %8zu %12.6f %12.6f %8s %12.6f %8.2f\n", row.n,
                row.colours, row.hessian_columns, row.hvp_columns,
                row.hessian_s, row.hvp_s,
                exponent(static_cast<double>(prev.n), prev.hvp_s,
                         static_cast<double>(row.n), row.hvp_s)
                    .c_str(),
                row.n_hvps_s, row.hessian_s / row.hvp_s);
    prev = row;
  }
}

} // namespace

int main() {
  std::printf("H(x)v against the Hessian it never forms.\n");
  std::printf("Batches of %zu points.  `exp` is the log-log slope of hvp(s) "
              "against the row above;\n",
              kPoints);
  std::printf("`H/hvp` is how much one Hessian costs over one HVP, and "
              "`n*hvp(s)` is what n of them\n");
  std::printf("come to -- above hessian(s), the colouring is still the right "
              "way to get the whole matrix.\n");

  static constexpr std::array dense_sizes{8uz, 16uz, 32uz, 64uz};
  static constexpr std::array banded_sizes{8uz, 16uz, 32uz, 64uz, 128uz};
  run("dense coupling (colours == n)", dense_model, dense_sizes);
  run("banded coupling (colours stay small)", banded_model, banded_sizes);
  return 0;
}
