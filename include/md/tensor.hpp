#pragma once

#include "expressions.hpp" // Numeric
#include "md/layouts.hpp"  // layout_leading_simplex — the symmetric packings
#include "md/md.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>
#include <utility>

namespace diff {

// ===========================================================================
// md_tensor — an owning, constant-evaluable tensor with compile-time extents.
// Two spellings, on purpose:
//
//   t[i, j, k]     the mdspan spelling, one mapping call, no intermediates
//   t[i][j][k]     the nested-array spelling every existing caller uses
//
// The second is not a slice.  A rank-reducing view would need the layout to
// support submdspan, which the packed and sparse layouts do not -- so the
// proxy just accumulates the index prefix and calls the mapping once, at the
// last subscript.  That keeps the old spelling working over *any* layout, and
// keeps it as cheap as the mdspan one: the proxy is a pointer plus a small
// index array, all of it constexpr and none of it surviving -O2.
// ===========================================================================

namespace detail {

// A pack of Order copies of N, as an md::extents.  The index_sequence element
// is discarded -- always_v exists only so the pack has something to expand.
template <std::size_t, std::size_t V> inline constexpr std::size_t always_v = V;

template <std::size_t N, std::size_t... I>
consteval auto uniform_extents_fn(std::index_sequence<I...>) noexcept {
  return std::type_identity<md::extents<std::size_t, always_v<I, N>...>>{};
}

template <std::size_t Lead, std::size_t N, std::size_t... I>
consteval auto stacked_extents_fn(std::index_sequence<I...>) noexcept {
  return std::type_identity<
      md::extents<std::size_t, Lead, always_v<I, N>...>>{};
}

} // namespace detail

// extents<size_t, N, N, ..., N> with Order repetitions — the shape of a
// derivative tensor of order Order over N variables.
template <std::size_t N, std::size_t Order>
using uniform_extents_t = typename decltype(detail::uniform_extents_fn<N>(
    std::make_index_sequence<Order>{}))::type;

// The same, with one leading axis of a different size — a per-output stack of
// derivative tensors, which is what a vector-valued Equation produces.  The
// old spelling was std::array<nd_array_t<S, N, Order>, Lead>, i.e. an outer
// dimension that lived in a different type from the inner ones.
template <std::size_t Lead, std::size_t N, std::size_t Order>
using stacked_extents_t = typename decltype(detail::stacked_extents_fn<Lead, N>(
    std::make_index_sequence<Order>{}))::type;

template <Numeric S, md::CStaticExtents Ext, md::CLayoutFor<Ext> Layout>
class md_tensor;

namespace detail {

// The index-prefix proxy behind t[i][j][k].  Depth indices have been supplied;
// it resolves to a reference once Depth + 1 == rank.
//
// It holds a pointer to the tensor rather than to the data so that both the
// mapping and the const-ness travel with it, and it is deliberately not
// storable in any useful way -- like std::vector<bool>'s proxy, it exists for
// the duration of the subscript chain and nothing else.
template <typename Tensor, std::size_t Depth> class md_index_proxy {
  using index_type = typename Tensor::index_type;
  static constexpr std::size_t kRank = Tensor::rank();
  static_assert(Depth < kRank);

  Tensor *tensor_;
  std::array<index_type, Depth> prefix_;

public:
  constexpr md_index_proxy(Tensor &t,
                           const std::array<index_type, Depth> &prefix) noexcept
      : tensor_(&t), prefix_(prefix) {}

  // The length of the axis this proxy is about to index, so a caller can walk
  // a row the way it walked the nested array's inner std::array.
  [[nodiscard]] static constexpr std::size_t size() noexcept {
    return static_cast<std::size_t>(Tensor::extent(Depth));
  }

  [[nodiscard]] constexpr decltype(auto)
  operator[](index_type i) const noexcept {
    const auto next = [&]<std::size_t... K>(std::index_sequence<K...>) {
      return std::array<index_type, Depth + 1>{prefix_[K]..., i};
    }(std::make_index_sequence<Depth>{});

    if constexpr (Depth + 1 == kRank) {
      return tensor_->at_index(next);
    } else {
      return md_index_proxy<Tensor, Depth + 1>(*tensor_, next);
    }
  }
};

} // namespace detail

template <Numeric S, md::CStaticExtents Ext,
          md::CLayoutFor<Ext> Layout = md::layout_right>
class md_tensor {
public:
  using element_type = S;
  using value_type = S;
  using extents_type = Ext;
  using layout_type = Layout;
  using mapping_type = typename Layout::template mapping<Ext>;
  using index_type = typename Ext::index_type;
  using reference = S &;
  using const_reference = const S &;

  static constexpr std::size_t rank() noexcept { return Ext::rank(); }

private:
  static_assert(std::constructible_from<mapping_type, Ext>,
                "md_tensor needs a layout whose mapping is constructible from "
                "bare extents; layouts carrying extra state (layout_stride) "
                "cannot be owned this way");
  static_assert(rank() > 0, "md_tensor: rank 0 is just S");

  static constexpr mapping_type kMapping{Ext{}};
  static constexpr std::size_t kSize =
      static_cast<std::size_t>(kMapping.required_span_size());

  std::array<S, kSize> data_{};

public:
  constexpr md_tensor() noexcept = default;

  [[nodiscard]] static constexpr Ext extents() noexcept { return Ext{}; }
  [[nodiscard]] static constexpr mapping_type mapping() noexcept {
    return kMapping;
  }
  // The number of *stored* elements, which under a packed layout is fewer than
  // the number of addressable ones.
  [[nodiscard]] static constexpr std::size_t size() noexcept { return kSize; }
  [[nodiscard]] static constexpr index_type extent(std::size_t r) noexcept {
    return Ext{}.extent(r);
  }

  [[nodiscard]] constexpr S *data() noexcept { return data_.data(); }
  [[nodiscard]] constexpr const S *data() const noexcept {
    return data_.data();
  }

  [[nodiscard]] constexpr auto view() noexcept {
    return md::mdspan<S, Ext, Layout>(data_.data(), kMapping);
  }
  [[nodiscard]] constexpr auto view() const noexcept {
    return md::mdspan<const S, Ext, Layout>(data_.data(), kMapping);
  }

  // t[i, j, k] — the mdspan spelling.
  template <std::integral... I>
    requires(sizeof...(I) == Ext::rank())
  [[nodiscard]] constexpr reference operator[](I... idx) noexcept {
    return data_[static_cast<std::size_t>(
        kMapping(static_cast<index_type>(idx)...))];
  }
  template <std::integral... I>
    requires(sizeof...(I) == Ext::rank())
  [[nodiscard]] constexpr const_reference operator[](I... idx) const noexcept {
    return data_[static_cast<std::size_t>(
        kMapping(static_cast<index_type>(idx)...))];
  }

  // t[i][j][k] — the nested-array spelling.  Only reachable at rank >= 2; at
  // rank 1 the variadic overload above already takes a single index, so there
  // is no ambiguity to resolve.
  [[nodiscard]] constexpr auto operator[](index_type i) noexcept
    requires(Ext::rank() >= 2)
  {
    return detail::md_index_proxy<md_tensor, 1>(*this,
                                                std::array<index_type, 1>{i});
  }
  [[nodiscard]] constexpr auto operator[](index_type i) const noexcept
    requires(Ext::rank() >= 2)
  {
    return detail::md_index_proxy<const md_tensor, 1>(
        *this, std::array<index_type, 1>{i});
  }

  // The leading axis as a range, so `for (const auto &row : t.rows())` reads
  // the way `for (const auto &row : nested_array)` used to.  Not begin()/end()
  // on the tensor itself: a rank-3 tensor's "elements" would then be proxies,
  // which is a range of things that are not the value_type and would make
  // md_tensor satisfy range concepts it has no business satisfying.
  [[nodiscard]] constexpr auto rows() noexcept
    requires(Ext::rank() >= 2)
  {
    return std::views::iota(index_type{0}, Ext{}.extent(0)) |
           std::views::transform([this](index_type i) { return (*this)[i]; });
  }
  [[nodiscard]] constexpr auto rows() const noexcept
    requires(Ext::rank() >= 2)
  {
    return std::views::iota(index_type{0}, Ext{}.extent(0)) |
           std::views::transform([this](index_type i) { return (*this)[i]; });
  }

  // The proxy's terminal step, and the array-of-indices spelling nd_index used
  // to offer.
  [[nodiscard]] constexpr reference
  at_index(const std::array<index_type, Ext::rank()> &idx) noexcept {
    return [&]<std::size_t... K>(std::index_sequence<K...>) -> reference {
      return data_[static_cast<std::size_t>(kMapping(idx[K]...))];
    }(std::make_index_sequence<Ext::rank()>{});
  }
  [[nodiscard]] constexpr const_reference
  at_index(const std::array<index_type, Ext::rank()> &idx) const noexcept {
    return [&]<std::size_t... K>(std::index_sequence<K...>) -> const_reference {
      return data_[static_cast<std::size_t>(kMapping(idx[K]...))];
    }(std::make_index_sequence<Ext::rank()>{});
  }

  // Compares stored cells.  Under a packed layout that is exactly the set of
  // distinct addressable values, so this stays the right comparison.
  [[nodiscard]] friend constexpr bool operator==(const md_tensor &a,
                                                 const md_tensor &b) noexcept {
    return a.data_ == b.data_;
  }
};

// The direct replacement for nd_array_t<S, N, Order>.
//
// The default layout is the symmetric packing, not layout_right: a derivative
// tensor is symmetric in every pair of its axes, so the dense form stores
// N^Order cells to hold C(N + Order - 1, Order) distinct values — 216 cells
// for 56 values at Order 3 over 6 variables.  Every index still reads, the
// mapping just sorts it first.  Pass md::layout_right explicitly for a tensor
// that is not symmetric.
template <Numeric S, std::size_t N, std::size_t Order,
          typename Layout = layout_simplex_packed>
using nd_tensor_t = md_tensor<S, uniform_extents_t<N, Order>, Layout>;

// A per-output stack of them: Lead tensors of rank Order over N variables,
// as one rank-(Order + 1) object.  The output axis is dense — it indexes
// different functions, which have nothing to be symmetric about.
template <Numeric S, std::size_t Lead, std::size_t N, std::size_t Order,
          typename Layout = layout_leading_simplex<1>>
using nd_stack_t = md_tensor<S, stacked_extents_t<Lead, N, Order>, Layout>;

// Write a rank-1 range into row `i` of a rank-2 tensor.
//
// Index-based rather than a pointer walk into data(), so it holds for any
// layout — under a packed mapping a "row" is not contiguous and the two
// spellings would disagree.  The sweeps still fill a plain std::array first:
// backward() writes through a CNumericBuffer, and a rank-1 mdspan is not a
// range, so the tensor cannot be the sweep's out-param directly.
template <Numeric S, md::CStaticExtents Ext, md::CLayoutFor<Ext> Layout,
          std::ranges::input_range R>
  requires(Ext::rank() == 2)
constexpr void assign_row(md_tensor<S, Ext, Layout> &t,
                          typename Ext::index_type i, R &&src) noexcept {
  for (auto &&[k, v] : std::views::zip(
           std::views::iota(typename Ext::index_type{0}, Ext{}.extent(1)),
           src)) {
    t[i, k] = v;
  }
}

} // namespace diff
