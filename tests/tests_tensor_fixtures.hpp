#pragma once
// Shared by tests_tensor.cpp and dual/tests_tensor_dual.cpp.

#include "tests_common.hpp"

// Bit-exactness gate: fold the raw IEEE bits of every driver result over a
// sweep into one FNV hash.  Checked only under -DDDX_PIN_BIT_HASH=<value>; the
// hash is per-toolchain (FMA formation moves the last bit).  Recompute the
// pinned value deliberately -- it moving means results moved.

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

// The packed tensor with the permutations skipped must equal the dense tensor
// with every multi-index evaluated.
template <std::size_t Order, CExpression Expr, CTupleLike Values>
void ExpectPackedMatchesDense(const Expr &expr, const Values &values) {
  const auto packed = Equation{expr}.template derivative_tensor<Order>(values);
  constexpr std::size_t N = std::tuple_size_v<Values>;

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
