#pragma once

#include "md/md.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>

// The one layout this library needs beyond the standard ones: mdspan's layout is
// a policy, which is what lets a derivative tensor's symmetry become storage
// rather than a convention.  The sparse-pattern layout is in coupling.hpp, next
// to the consteval pass that derives the pattern it maps.
//
// The mapping is a literal type with a constexpr constructor, so a tensor over it
// stays usable in constant evaluation -- which is why nothing here allocates or
// holds anything but extents.
namespace diff {

namespace detail {

// C(n, k).  Integer arithmetic only, in the multiply-then-divide form, which
// never overflows early because the partial product of k consecutive integers is
// always divisible by k!.  This value is a storage *offset*, so a rounded double
// would file a derivative in the wrong cell.
//
// consteval: this only ever sizes storage.  The runtime subscript path uses
// binomial_fixed below, and the compiler enforces the split.
[[nodiscard]] consteval std::size_t binomial(std::size_t n,
                                             std::size_t k) noexcept {
  if (k > n) {
    return 0;
  }
  k = std::min(k, n - k); // C(n, k) == C(n, n - k); take the shorter fold
  return std::ranges::fold_left(
      std::views::iota(std::size_t{1}, k + 1), std::size_t{1},
      [n, k](std::size_t acc, std::size_t i) { return acc * (n - k + i) / i; });
}

static_assert(binomial(4, 2) == 6);
static_assert(binomial(8, 3) == 56);
static_assert(binomial(10, 0) == 1);

// C(n, K) with K fixed at compile time: the falling product n(n-1)...(n-K+1)
// over K!.  binomial() above is structurally wrong for the *subscript* path --
// its `k = min(k, n - k)` makes the fold's bound depend on the runtime n, so the
// loop cannot unroll and every subscript pays a real loop.  Here nothing but the
// factors depends on n, so it collapses to a few multiplies and one divide.
// constexpr, not consteval: the mapping calls it on every subscript.
template <std::size_t K>
[[nodiscard]] constexpr std::size_t binomial_fixed(std::size_t n) noexcept {
  if (n < K) {
    return 0;
  }
  std::size_t num = 1;
  std::size_t den = 1;
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    ((num *= (n - I), den *= (I + 1)), ...);
  }(std::make_index_sequence<K>{});
  return num / den;
}

static_assert(binomial_fixed<2>(4) == 6);
static_assert(binomial_fixed<3>(8) == 56);
static_assert(binomial_fixed<0>(10) == 1);
static_assert(binomial_fixed<1>(7) == 7);
static_assert(binomial_fixed<2>(1) == 0);

} // namespace detail

// ---------------------------------------------------------------------------
// layout_leading_simplex<Lead> — the rank-Order generalisation of the above,
// with Lead leading axes left dense.
//
// A derivative tensor is symmetric under every permutation of its *derivative*
// axes (Clairaut/Schwarz), so only non-decreasing multi-indices over those carry
// distinct values: C(N + Order - 1, Order) of them against N^Order dense cells --
// at Order 3 over 6 variables, 56 instead of 216.
//
// Lead is for a vector-valued Equation, which stacks one such tensor per output:
// the output axis is neither interchangeable with the derivative axes nor the
// same length, so it stays a dense outer dimension.  Lead == 0 is the scalar case.
//
// The symmetric part sorts its indices and ranks the resulting multiset in
// colex order, through the standard multiset-to-combination bijection
// b_t = a_t + t.
// ---------------------------------------------------------------------------
template <std::size_t Lead> struct layout_leading_simplex {
  template <md::CExtents Ext> class mapping {
    static_assert(Ext::rank() > Lead,
                  "layout_leading_simplex needs at least one symmetric axis");
    static constexpr std::size_t kOrder = Ext::rank() - Lead;

  public:
    using extents_type = Ext;
    using index_type = typename Ext::index_type;
    using rank_type = typename Ext::rank_type;
    using layout_type = layout_leading_simplex;

    // The symmetric axes must be uniform: two axes can only be interchangeable
    // if they have the same length, and every use here comes from
    // uniform_extents_t / stacked_extents_t.
    constexpr mapping() noexcept = default;
    constexpr explicit mapping(const Ext &e) noexcept : ext_(e) {}

    [[nodiscard]] constexpr const Ext &extents() const noexcept { return ext_; }

    // Cells per output block: C(N + Order - 1, Order).
    [[nodiscard]] constexpr index_type block_size() const noexcept {
      return static_cast<index_type>(detail::binomial_fixed<kOrder>(
          static_cast<std::size_t>(ext_.extent(Lead)) + kOrder - 1));
    }

    [[nodiscard]] constexpr index_type required_span_size() const noexcept {
      const index_type lead = std::ranges::fold_left(
          std::views::iota(std::size_t{0}, Lead), index_type{1},
          [this](index_type acc, std::size_t r) {
            return acc * ext_.extent(r);
          });
      return lead * block_size();
    }

    [[nodiscard]] constexpr index_type
    operator()(std::integral auto... idx) const noexcept
      requires(sizeof...(idx) == Ext::rank())
    {
      const std::array<index_type, Ext::rank()> all{
          static_cast<index_type>(idx)...};

      // Dense row-major over the leading axes — Horner over the extents.
      const index_type lead = std::ranges::fold_left(
          std::views::iota(std::size_t{0}, Lead), index_type{0},
          [&](index_type acc, std::size_t r) {
            return acc * ext_.extent(r) + all[r];
          });

      std::array<index_type, kOrder> a{};
      std::ranges::copy(all | std::views::drop(Lead), a.begin());
      std::ranges::sort(a);
      // b_t = a_t + t is strictly increasing, so it is a plain combination;
      // rank it in colex order, which is the sum of C(b_t, t + 1).
      //
      // Expanded over an index_sequence rather than folded over a zip so that
      // `t` — and therefore binomial's `k` — is a compile-time constant at each
      // term.
      const index_type slot = [&]<std::size_t... T>(std::index_sequence<T...>) {
        return static_cast<index_type>(
            (detail::binomial_fixed<T + 1>(static_cast<std::size_t>(a[T]) + T) +
             ... + std::size_t{0}));
      }(std::make_index_sequence<kOrder>{});
      return lead * block_size() + slot;
    }

    static constexpr bool is_always_unique() noexcept { return kOrder == 1; }
    static constexpr bool is_always_exhaustive() noexcept { return true; }
    static constexpr bool is_always_strided() noexcept { return false; }
    [[nodiscard]] constexpr bool is_unique() const noexcept {
      return kOrder == 1;
    }
    [[nodiscard]] constexpr bool is_exhaustive() const noexcept { return true; }
    [[nodiscard]] constexpr bool is_strided() const noexcept { return false; }

    [[nodiscard]] friend constexpr bool operator==(const mapping &a,
                                                   const mapping &b) noexcept {
      return a.ext_ == b.ext_;
    }

  private:
    Ext ext_{};
  };
};

// The scalar case: every axis is a derivative axis.
using layout_simplex_packed = layout_leading_simplex<0>;

} // namespace diff
