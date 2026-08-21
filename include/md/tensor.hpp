#pragma once

#include "expr/expressions.hpp" // Numeric
#include "md/layouts.hpp" // layout_leading_simplex — the symmetric packings
#include "md/md.hpp"
#include "util/config.hpp" // DDX_SELF

#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>
#include <utility>

namespace ddx::impl {

// md_tensor: an owning, constant-evaluable tensor with compile-time extents.
// Both t[i, j, k] and t[i][j][k] work.  The latter is not a slice -- a
// rank-reducing view needs submdspan, which the packed and sparse layouts do
// not support -- so the proxy accumulates the prefix and calls the mapping once
// at the last subscript.  Works over any layout, and none of it survives -O2.

namespace detail {

// A pack of Order copies of N, as an md::extents.
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

// The shape of a derivative tensor of order Order over N variables.
template <std::size_t N, std::size_t Order>
using uniform_extents_t = typename decltype(detail::uniform_extents_fn<N>(
    std::make_index_sequence<Order>{}))::type;

// The same with one leading axis: a per-output stack, as a vector-valued
// Equation produces.
template <std::size_t Lead, std::size_t N, std::size_t Order>
using stacked_extents_t = typename decltype(detail::stacked_extents_fn<Lead, N>(
    std::make_index_sequence<Order>{}))::type;

template <Numeric S, md::CStaticExtents Ext, md::CLayoutFor<Ext> Layout>
class md_tensor;

namespace detail {

// The index-prefix proxy behind t[i][j][k]; resolves to a reference once
// Depth + 1 == rank.  It points at the tensor, not the data, so the mapping and
// the const-ness travel with it.
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

  [[nodiscard]] constexpr decltype(auto)
  operator[](index_type i) const noexcept {
    const auto next = index_apply<Depth>([&]<std::size_t... K>() {
      return std::array<index_type, Depth + 1>{prefix_[K]..., i};
    });

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

  [[nodiscard]] static constexpr mapping_type mapping() noexcept {
    return kMapping;
  }
  // Stored, which under a packed layout is fewer than addressable.
  [[nodiscard]] static constexpr std::size_t size() noexcept { return kSize; }
  [[nodiscard]] static constexpr index_type extent(std::size_t r) noexcept {
    return Ext{}.extent(r);
  }

  [[nodiscard]] constexpr S *data() noexcept { return data_.data(); }
  [[nodiscard]] constexpr const S *data() const noexcept {
    return data_.data();
  }

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

  // t[i][j][k].  Only at rank >= 2; at rank 1 the overload above already takes
  // a single index.
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

  // The proxy's terminal step.  The two value categories differ only in the
  // constness of data_, so P0847 writes it once; DDX_DEDUCING_THIS=off is a
  // supported configuration, hence the pair below it.
#if DDX_DEDUCING_THIS
  [[nodiscard]] constexpr decltype(auto)
  at_index(DDX_SELF, const std::array<index_type, Ext::rank()> &idx) noexcept {
    return index_apply<Ext::rank()>([&]<std::size_t... K>() -> decltype(auto) {
      return self.data_[static_cast<std::size_t>(kMapping(idx[K]...))];
    });
  }
#else
  [[nodiscard]] constexpr reference
  at_index(const std::array<index_type, Ext::rank()> &idx) noexcept {
    return index_apply<Ext::rank()>([&]<std::size_t... K>() -> reference {
      return data_[static_cast<std::size_t>(kMapping(idx[K]...))];
    });
  }
  [[nodiscard]] constexpr const_reference
  at_index(const std::array<index_type, Ext::rank()> &idx) const noexcept {
    return index_apply<Ext::rank()>([&]<std::size_t... K>() -> const_reference {
      return data_[static_cast<std::size_t>(kMapping(idx[K]...))];
    });
  }
#endif

  [[nodiscard]] friend constexpr bool operator==(const md_tensor &a,
                                                 const md_tensor &b) noexcept {
    return a.data_ == b.data_;
  }
};

// The default layout is the symmetric packing, not layout_right:
template <Numeric S, std::size_t N, std::size_t Order,
          typename Layout = layout_simplex_packed>
using nd_tensor_t = md_tensor<S, uniform_extents_t<N, Order>, Layout>;

// A per-output stack, as one rank-(Order + 1) object.  The output axis is
// dense: it indexes different functions.
template <Numeric S, std::size_t Lead, std::size_t N, std::size_t Order,
          typename Layout = layout_leading_simplex<1>>
using nd_stack_t = md_tensor<S, stacked_extents_t<Lead, N, Order>, Layout>;

// Index-based rather than a pointer walk into data(): under a packed mapping a
// row is not contiguous.  The sweeps still fill a plain array first, since a
// rank-1 mdspan is not a range.
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

} // namespace ddx::impl
