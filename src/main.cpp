// A tour of every way to ask this library for a derivative, with timings.
//
//   LAMBDA — an ordinary templated function; a black box the drivers can only
//            evaluate at seeded dual numbers.
//   GRAPH  — the same formula built from Variable<>, so it is a compile-time
//            tree whose sparsity, symbolic partials and constexpr evaluation
//            all fall out of its type.
//
// Both give the same numbers; see the last two sections for the difference.

#include "api.hpp"                     // the public surface: Equation, var, named
#include "drivers/coupling.hpp"        // hessian_pattern(), color_columns()
#include <tuple>
#include "drivers/hessian.hpp" // hessian(), gradient()
#include "expr/bound.hpp"              // eval(), bind(), named<>()
#include "expr/equation.hpp"           // Equation: vector-valued f, Jacobians

#include <array>
#include <chrono>
#include <concepts>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <utility>

using namespace diff::impl;

// <format> plus a stream, so this does not gate on libstdc++ shipping <print>.
template <typename... Args>
static void println(std::format_string<Args...> fmt, Args &&...args) {
  std::cout << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

namespace {

// Somewhere the optimiser cannot reason about, so timing is not of an empty loop.
volatile double g_sink = 0.0;

// Nanoseconds per call, best-of-rounds.  The minimum, not the mean: one
// descheduled run inflates a mean beyond recovery.  One warm-up is discarded.
template <std::invocable F>
double bench_ns(F &&f, int reps = 4000, int rounds = 7) {
  using clock = std::chrono::steady_clock;
  f(); // warm
  double best = std::numeric_limits<double>::max();
  for (int r = 0; r < rounds; ++r) {
    const auto t0 = clock::now();
    double acc = 0.0;
    for (int i = 0; i < reps; ++i) {
      acc += f();
    }
    const auto t1 = clock::now();
    g_sink = g_sink + acc;
    const double ns =
        std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;
    best = std::min(best, ns);
  }
  return best;
}

void row(const std::string &label, double value, double ns) {
  println("  {:<44}{:>12.4f}{:>11.1f} ns", label, value, ns);
}

// f(x, y, z) = x*log(x) + y^2*z + exp(x*z), as a plain function ...
template <Numeric T> T energy(const T *v) {
  using std::exp, std::log;
  return v[0] * log(v[0]) + v[1] * v[1] * v[2] + exp(v[0] * v[2]);
}

// ... and as a graph.  Dual<double> is what lets forward-over-reverse run on it.
auto make_graph() {
  using D = Dual<double>;
  Variable<D, FixedString{"x"}> x;
  Variable<D, FixedString{"y"}> y;
  Variable<D, FixedString{"z"}> z;
  return x * log(x) + y * y * z + exp(x * z);
}

// The same graph over plain double, for the constexpr-capable symbolic API.
constexpr auto make_graph_plain() {
  Variable<double, FixedString{"x"}> x;
  Variable<double, FixedString{"y"}> y;
  Variable<double, FixedString{"z"}> z;
  return x * log(x) + y * y * z + exp(x * z);
}

// An 8-variable chain: neighbour coupling plus one long-range term -- the shape
// the sparsity pass is built for.
template <Numeric T> T chain_energy(const T *v, std::size_t n) {
  using std::exp, std::log;
  T acc = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    acc = acc + v[i] * log(v[i]);
  }
  for (std::size_t i = 0; i + 1 < n; ++i) {
    const double c = 0.5 + 0.01 * static_cast<double>(i);
    acc = acc + c * (v[i] - v[i + 1]) * (v[i] - v[i + 1]);
  }
  return acc + exp(v[0] * v[n - 1]);
}

auto make_chain8() {
  using D = Dual<double>;
  Variable<D, FixedString{"x0"}> a;
  Variable<D, FixedString{"x1"}> b;
  Variable<D, FixedString{"x2"}> c;
  Variable<D, FixedString{"x3"}> d;
  Variable<D, FixedString{"x4"}> e;
  Variable<D, FixedString{"x5"}> f;
  Variable<D, FixedString{"x6"}> g;
  Variable<D, FixedString{"x7"}> h;
  return a * log(a) + b * log(b) + c * log(c) + d * log(d) + e * log(e) +
         f * log(f) + g * log(g) + h * log(h) + 0.50 * (a - b) * (a - b) +
         0.51 * (b - c) * (b - c) + 0.52 * (c - d) * (c - d) +
         0.53 * (d - e) * (d - e) + 0.54 * (e - f) * (e - f) +
         0.55 * (f - g) * (f - g) + 0.56 * (g - h) * (g - h) + exp(a * h);
}

} // namespace

int main() {
  // Positional values are in CANONICAL (alphabetical, not source) order; the
  // named<> forms below sidestep the question.
  std::array<double, 3> p{0.4, 0.7, 1.1}; // x, y, z
  const std::span<const double> xs{p.data(), p.size()};

  auto lambda = [](const auto *dof) { return energy(dof); };
  const auto graph = make_graph();
  constexpr auto gplain = make_graph_plain();

  println("point (x, y, z) = ({}, {}, {})", p[0], p[1], p[2]);
  println("timings are ns/call, best of 7 rounds x 4000 reps\n");

  // An expression is an empty type: the whole tree costs no bytes.
  {
    constexpr auto x = var<"x", int>;
    constexpr auto y = var<"y", int>;
    auto symbols_only = x * y + x * y * y;
    static_assert(std::is_empty_v<decltype(symbols_only)>,
                  "an expression over stateless leaves stores nothing");
    println("sizeof(x*y + x*y*y)   = {} byte", sizeof(symbols_only));
    auto with_constant = x * y + PC(3) * x * y * y;
    println("sizeof(.. + 3*x*y*y)  = {} bytes (one stored constant)\n",
                 sizeof(with_constant));
  }

  // ---- VALUE
  println("VALUE");
  row("lambda  energy(p.data())", energy(p.data()),
      bench_ns([&] { return energy(p.data()); }));
  row("graph   expr.eval(x, y, z)", gplain.eval(p[0], p[1], p[2]),
      bench_ns([&] { return gplain.eval(p[0], p[1], p[2]); }));
  row("graph   eval(expr, named<\"x\">(..), ..)",
      eval(gplain, named<"x">(p[0]), named<"y">(p[1]), named<"z">(p[2])),
      bench_ns([&] {
        return eval(gplain, named<"x">(p[0]), named<"y">(p[1]),
                    named<"z">(p[2]));
      }));

  // ---- GRADIENT
  println("\nGRADIENT");
  row("lambda  gradient(f, xs)           [0]", diff::impl::gradient(lambda, xs)[0],
      bench_ns([&] { return diff::impl::gradient(lambda, xs)[0]; }, 2000));
  row("graph   gradient<Reverse>(expr,p) [0]",
      Equation{gplain}.gradient(p)[0],
      bench_ns([&] { return Equation{gplain}.gradient(p)[0]; }));
  row("graph   ...bound by name          [0]",
      Equation{gplain}.gradient(named<"x">(p[0]), named<"y">(p[1]), named<"z">(p[2]))[0],
      bench_ns([&] {
        return Equation{gplain}.gradient(named<"x">(p[0]), named<"y">(p[1]), named<"z">(p[2]))[0];
      }));

  // ---- HESSIAN.  hessian() routes on what it is given: the lambda routes
  // allocate, the symbolic one returns a packed md_tensor on the stack.
  println("\nHESSIAN            H(0,2)");
  row("lambda  hessian(f, xs)", std::get<2>(diff::impl::hessian(lambda, xs))[0 * xs.size() + 2],
      bench_ns([&] { return std::get<2>(diff::impl::hessian(lambda, xs))[0 * xs.size() + 2]; }, 1000));
  row("graph   hessian(expr, xs)", std::get<2>(diff::impl::hessian(graph, xs))[0 * xs.size() + 2],
      bench_ns([&] { return std::get<2>(diff::impl::hessian(graph, xs))[0 * xs.size() + 2]; }, 1000));
  {
    const auto Hs = Equation{graph}.hessian(p);
    row("graph   hessian<Reverse>(expr, p)", (Hs[0, 2]), bench_ns([&] {
          const auto H = Equation{graph}.hessian(p);
          return (H[0, 2]);
        }));
    println("          packed: stores {} cells, not {}", Hs.size(), 3 * 3);
  }

  // ---- HIGHER ORDER, graph only: there is no lambda entry point above 2nd.
  println("\nHIGHER ORDER  (graph only)");
  {
    const auto T = Equation{gplain}.template derivative_tensor<3>(p);
    row("graph   derivative_tensor<3>  d3f/dx dz dz", (T[0, 2, 2]),
        bench_ns([&] {
          const auto t = Equation{gplain}.template derivative_tensor<3>(p);
          return (t[0, 2, 2]);
        }, 1000));
    println("          same cell, chained spelling t[0][2][2] = {:.4f}",
                 T[0][2][2]);
    println("          packed: stores {} cells, not {} — symmetric",
                 T.size(), 3 * 3 * 3);
  }

  // ---- VECTOR-VALUED f: Equation holds one expression per output.
  println("\nJACOBIAN  (vector-valued -> Equation)");
  {
    Variable<double, FixedString{"x"}> x;
    Variable<double, FixedString{"y"}> y;
    Variable<double, FixedString{"z"}> z;
    auto sys = Equation(x * y, sin(x) * z, y * y + z); // f: R^3 -> R^3

    row("graph   .jacobian<Symbolic>(p)      J(1,0)",
        (sys.jacobian<DiffMode::Symbolic>(p)[1, 0]), bench_ns([&] {
          const auto J = sys.jacobian<DiffMode::Symbolic>(p);
          return (J[1, 0]);
        }));
    row("graph   .jacobian<Reverse>(p)       J(1,0)",
        (sys.jacobian<DiffMode::Reverse>(p)[1, 0]), bench_ns([&] {
          const auto J = sys.jacobian<DiffMode::Reverse>(p);
          return (J[1, 0]);
        }));
    row("graph   .derivative_tensor<2>(p)  d2f1/dxdz",
        (sys.derivative_tensor<2>(p)[1, 0, 2]), bench_ns([&] {
          const auto T = sys.derivative_tensor<2>(p);
          return (T[1, 0, 2]);
        }, 1000));
  }

  // ---- The same 8-variable chain both ways.  The graph's sparsity is known
  // from its type, so the sweep count follows the bandwidth rather than n --
  // a gap that widens with n (~10x at n = 16).
  println("\nSTRUCTURED PROBLEM — 8-variable chain, Hessian");
  {
    std::array<double, 8> q{};
    for (std::size_t k = 0; k < q.size(); ++k) {
      q[k] = 0.15 + 0.6 * (static_cast<double>(k) + 1.0) / 9.0;
    }
    const std::span<const double> qs{q.data(), q.size()};
    auto chain_lambda = [n = q.size()](const auto *dof) {
      return chain_energy(dof, n);
    };
    const auto chain_graph = make_chain8();

    const double t_lambda =
        bench_ns([&] { return std::get<2>(diff::impl::hessian(chain_lambda, qs))[0 * qs.size() + 7]; }, 400);
    const double t_graph =
        bench_ns([&] { return std::get<2>(diff::impl::hessian(chain_graph, qs))[0 * qs.size() + 7]; }, 400);

    using G = std::remove_cvref_t<decltype(chain_graph)>;
    constexpr auto pattern = hessian_pattern<G>();
    constexpr auto colours = color_columns<8>(pattern);

    println("  lambda  hessian(f, qs)      {:>10.1f} ns   (probes all {} entries)",
                 t_lambda, 8 * 9 / 2);
    println("  graph   hessian(expr, qs)   {:>10.1f} ns   ({} colours -> {} sweeps)",
                 t_graph, colours.count, colours.count);
    println("  ---> graph is {:.1f}x faster, and the gap grows with n",
                 t_lambda / t_graph);
    println("\n  sparsity read off the TYPE (. = provably always zero):");
    for (std::size_t i = 0; i < 8; ++i) {
      std::string line;
      for (std::size_t j = 0; j < 8; ++j) {
        line += pattern[i][j] ? "# " : ". ";
      }
      println("    {}", line);
    }
    println("  {} of {} entries can be nonzero", hessian_nnz<G>(), 8 * 8);
  }

  // ---- ...and the whole symbolic path runs in constant evaluation.
  constexpr auto ct = [] {
    Variable<double, FixedString{"a"}> a;
    Variable<double, FixedString{"b"}> b;
    return Equation{a * b}.gradient(std::array{3.0, 4.0});
  }();
  static_assert(ct[0] == 4.0 && ct[1] == 3.0,
                "the symbolic path must be constant-evaluable");
  println("\nCOMPILE TIME                                        0.0 ns");
  println("  static_assert(gradient<Reverse>(a*b, {{3,4}})[0] == 4.0)  OK");
}
