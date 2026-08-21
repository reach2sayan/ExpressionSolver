#include "expr/format.hpp"
#include "jit/kernel.hpp"
#include "rt/derivative.hpp"
#include "rt/expr.hpp"
#include "rt/graph.hpp"

#include <gtest/gtest.h>

#include <format>
#include <regex>
#include <sstream>
#include <string>

// ===========================================================================
// Vectorisation (src/jit/compiler.cpp)
//
// Batch is the only shape in which the JIT beats AOT-compiled ddx, and it only
// does so if the loop actually vectorises.  A scalar fallback is silent -- the
// kernel still returns right answers, just slowly -- so it has to be asserted
// on the IR rather than inferred from a benchmark.
//
// The transcendentals are the whole point: README measures the libm call at
// around three quarters of a gradient, so a loop that vectorises the
// arithmetic and leaves sin() scalar has won nothing.
// ===========================================================================

namespace {
using ddx::rt::Builder;
using ddx::rt::Graph;

// A host with no native target has no JIT to test; every test here needs one,
// so bring it up once and let the assertion name the reason if it will not.
ddx::jit::Compiler &compiler() {
  static ddx::jit::result<ddx::jit::Compiler> c = ddx::jit::Compiler::create();
  EXPECT_TRUE(c.has_value()) << (c ? "" : c.error().detail);
  return *c;
}

// A compile that has to succeed for the test to mean anything.  compile()
// answers with result<Kernel>; the assertion names LLVM's own reason when it
// does not, instead of an empty Kernel three lines later.
ddx::jit::Kernel must_compile(auto &&...args) {
  auto k = compiler().compile(static_cast<decltype(args) &&>(args)...);
  EXPECT_TRUE(k.has_value()) << (k ? std::string{} : k.error().detail);
  return k ? std::move(*k) : ddx::jit::Kernel{};
}

// Any width: libmvec offers 2 (SSE), 4 (AVX2) and 8 (AVX512), and which one
// the vectoriser picks is the host's business, not this test's.
bool has_vector_doubles(const std::string &ir) {
  return std::regex_search(ir, std::regex{R"(<[0-9]+ x double>)"});
}

bool calls_vector_libm(const std::string &ir, const std::string &fn) {
  return std::regex_search(ir,
                           std::regex{"_ZGV[a-z]N[0-9]+v+_" + fn + R"(\()"});
}

std::string ir_for(auto build, std::size_t nvars, ddx::jit::Options opt = {}) {
  Builder<> b;
  std::vector<ddx::rt::RTExpression<>> vars;
  static constexpr const char *names[] = {"x", "y"};
  for (std::size_t i = 0; i < nvars; ++i) {
    vars.push_back(var(b, names[i]));
  }
  const auto root = build(vars);
  const auto graph = Graph<>::freeze(b, std::array{root.id(b)});
  return std::format("{}", ddx::jit::Ir{compiler(), graph, opt});
}

constexpr bool host_has_libmvec =
#if defined(__x86_64__) && defined(__linux__)
    true;
#else
    false;
#endif

// nounwind is the IR spelling of noexcept, and Kernel::operator() is noexcept
// on the strength of it.  willreturn and memory(argmem: readwrite) go with it:
// the body is arithmetic over the argument pointers and calls that are
// themselves nounwind and memory(none), so the optimiser may move code across
// them.  Asserted on the *optimised* IR, which is what actually runs.
TEST(JitVectorize, TheKernelIsNounwind) {
  const auto ir = ir_for([](auto &v) { return exp(v[0]) * sin(v[1]); }, 2);

  const std::smatch attrs = [&] {
    std::smatch m;
    // The attribute group the kernel definition names, then that group's body.
    std::smatch def;
    EXPECT_TRUE(std::regex_search(
        ir, def, std::regex{R"(define[^\n]*@ddx_kernel_\w+\([^\n]*#(\d+))"}))
        << "no kernel definition in the IR";
    std::regex_search(
        ir, m, std::regex{"attributes #" + def[1].str() + R"( = \{([^}]*)\})"});
    return m;
  }();
  ASSERT_FALSE(attrs.empty()) << "the kernel names no attribute group";

  const std::string got = attrs[1].str();
  EXPECT_NE(got.find("nounwind"), std::string::npos) << got;
  EXPECT_NE(got.find("willreturn"), std::string::npos) << got;
  EXPECT_NE(got.find("memory(argmem: readwrite)"), std::string::npos) << got;
}

TEST(JitVectorize, ArithmeticLoopVectorises) {
  const auto ir = ir_for([](auto &v) { return v[0] * v[1] + v[0]; }, 2);
  EXPECT_TRUE(has_vector_doubles(ir)) << "the arithmetic loop stayed scalar";
}

TEST(JitVectorize, TranscendentalsUseVectorLibm) {
  if constexpr (!host_has_libmvec) {
    GTEST_SKIP() << "libmvec is glibc on x86-64";
  }
  const auto ir = ir_for([](auto &v) { return exp(v[0]) * sin(v[1]); }, 2);
  EXPECT_TRUE(has_vector_doubles(ir)) << "loop stayed scalar";
  EXPECT_TRUE(calls_vector_libm(ir, "sin")) << "sin was not vectorised";
  EXPECT_TRUE(calls_vector_libm(ir, "exp")) << "exp was not vectorised";
}

TEST(JitVectorize, GradientLoopVectorises) {
  if constexpr (!host_has_libmvec) {
    GTEST_SKIP() << "libmvec is glibc on x86-64";
  }
  // The gradient graph is wider and shares subexpressions; it must still
  // vectorise, since that is the call anyone actually cares about.
  Builder<> b;
  const auto x = var(b, "x");
  const auto y = var(b, "y");
  const auto graph =
      ddx::rt::GraphBuilder{b}.value(exp(x) * sin(y)).gradient().build();
  const auto ir = std::format("{}", ddx::jit::Ir{compiler(), graph});
  EXPECT_TRUE(has_vector_doubles(ir)) << "the gradient loop stayed scalar";
  EXPECT_TRUE(calls_vector_libm(ir, "sin"));
}

// The escape hatch has to actually disable the thing, or it is no use when a
// mapping names a symbol the host's glibc lacks.
TEST(JitVectorize, VecLibNoneLeavesTranscendentalsScalar) {
  ddx::jit::Options off;
  off.veclib = ddx::jit::VecLib::None;
  const auto ir = ir_for([](auto &v) { return exp(v[0]) * sin(v[1]); }, 2, off);
  EXPECT_FALSE(calls_vector_libm(ir, "sin"));
  EXPECT_FALSE(calls_vector_libm(ir, "exp"));
}

// Every vector symbol the vectoriser chose must be resolvable, or the lookup in
// compile() fails: LLVM's table is written against a newer glibc than some
// hosts ship.
TEST(JitVectorize, EveryVectorSymbolResolves) {
  if constexpr (!host_has_libmvec) {
    GTEST_SKIP() << "libmvec is glibc on x86-64";
  }
  Builder<> b;
  const auto x = var(b, "x");
  auto f = sin(x) + cos(x) + exp(x) + log(x) + tanh(x) + erf(x) + cbrt(x);
  f = f + atan(x) + asinh(x) + pow(x, ddx::rt::RTExpression<>{3});

  const auto graph = Graph<>::freeze(b, std::array{f.id(b)});
  const auto kernel = must_compile(graph);
  ASSERT_TRUE(static_cast<bool>(kernel));

  const std::array col{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  const std::array<const double *, 1> xs{col.data()};
  std::array<double, 8> got{};
  double *const value_columns[]{got.data()};
  kernel(xs, value_columns, {}, {}, got.size());
  for (const double v : got) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

} // namespace

namespace {

// operator<< goes through the formatter, as it does for an expression.
TEST(JitVectorize, IrStreamsAndFormatsAlike) {
  Builder<> b;
  const auto x = var(b, "x");
  const auto graph = Graph<>::freeze(b, std::array{sin(x).id(b)});
  const ddx::jit::Ir ir{compiler(), graph};

  std::ostringstream os;
  os << ir;
  EXPECT_EQ(os.str(), std::format("{}", ir));
  EXPECT_NE(os.str().find("define"), std::string::npos);
}

} // namespace
