#pragma once
// Everything tests_tensor.cpp and dual/tests_tensor_dual.cpp both build on: the
// includes, the model expressions and the local helpers.  Split out so the
// two suites share one definition of each rather than a copy apiece.

#include "tests_common.hpp"

// ---------------------------------------------------------------------------
// Bit-exactness gate.
//
// This library's contract is exact answers, so a build flag or a kernel rewrite
// is only acceptable if it leaves every derivative bit-for-bit unchanged.
// EXPECT_DOUBLE_EQ cannot check that -- it allows 4 ULP, exactly the room a
// "harmless" reassociation needs.
//
// So: run every driver over a sweep of points, fold the raw IEEE bits of every
// result into one hash, and pin it.  Two builds agree iff the hash agrees.
// Recompute and update the constant deliberately, NEVER to make a red test
// green -- a change here means results moved, and that needs a reason.
//
// The pinned value is necessarily per-toolchain: where a compiler forms an FMA
// changes the last bit, so GCC, clang and MSVC each have their own answer and
// none is wrong.  The constant is therefore checked only when the build asks
// for it with -DDDX_PIN_BIT_HASH=<value>; the hash is always reported.
//
// The way to use it is a same-compiler A/B, which is the question it answers:
//
//     ctest -R BitExactness --output-on-failure     # note the hash
//     ...change a flag or a kernel...
//     ctest -R BitExactness --output-on-failure     # it must not have moved
//
// The value for GCC 15 on x86-64 with the flags CMakeLists.txt sets is
// 0x4af90585ebef1b44.
// ---------------------------------------------------------------------------
namespace {
struct BitHash {
  std::uint64_t h = 1469598103934665603ull;
  void operator()(double d) noexcept {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    for (int i = 0; i < 8; ++i) {
      h ^= (b >> (i * 8)) & 0xff;
      h *= 1099511628211ull;
    }
  }
};
} // namespace

// ===========================================================================
// mdspan layer — layouts, accessors, and the owning tensor.
//
// The layouts are the load-bearing part: a wrong mapping silently returns the
// wrong derivative rather than failing, so each one is pinned both as a
// runtime test and, below, inside constant evaluation.
// ===========================================================================

// The packed tensor with the permutations skipped must equal the dense tensor
// with every multi-index evaluated: a wrong symmetry assumption or simplex
// ranking shows up here and nowhere else.
template <std::size_t Order, CExpression Expr, CTupleLike Values>
void ExpectPackedMatchesDense(const Expr &expr, const Values &values) {
  const auto packed = Equation{expr}.template derivative_tensor<Order>(values);
  constexpr std::size_t N = std::tuple_size_v<Values>;

  // Recompute every cell densely, straight from the definition, with no
  // symmetry assumed anywhere.
  using S = double;
  using U = ddx::impl::nth_dual_t<S, Order>;
  using symbols = ddx::impl::extract_symbols_from_expr_t<Expr>;
  for (const auto &multi : ddx::impl::detail::index_grid<N, Order>()) {
    const auto idx = std::apply(
        [](auto... i) { return std::array<std::size_t, Order>{i...}; }, multi);
    std::array<U, N> seeds{};
    for (std::size_t k = 0; k < N; ++k) {
      seeds[k] =
          ddx::impl::detail::make_mixed_seed<S, Order>(values[k], idx, k);
    }
    const U val = expr.template eval_seeded<symbols>(seeds);
    const double dense = ddx::impl::detail::extract_nth<Order>(val);
    EXPECT_DOUBLE_EQ(packed.at_index(idx), dense)
        << "order " << Order << " at index " << idx[0];
  }
}
