#pragma once

// The one place the library says "mdspan": it picks between the vendored
// reference implementation (third_party/mdspan.hpp, Apache-2.0 WITH
// LLVM-exception) and the standard one, and everything else spells the names
// diff::impl::md::... without learning which it got.
//
// Explicit `using` declarations, not a namespace alias: `namespace md = std`
// would make diff::impl::md::vector compile.

#include <version> // __cpp_lib_mdspan / __cpp_lib_submdspan

// Set by the CMake option DIFF_MDSPAN_MODE=vendored, so that path stays
// continuously compiled even on a toolchain that could use the standard one.
#if defined(DIFF_MDSPAN_FORCE_VENDORED)
#define DIFF_MDSPAN_USE_STD 0
#elif __has_include(<mdspan>) && defined(__cpp_lib_mdspan) &&                   \
    __cpp_lib_mdspan >= 202207L && defined(__cpp_lib_submdspan) &&             \
    __cpp_lib_submdspan >= 202306L
// Both halves or neither: a std::mdspan without std::submdspan leaves slicing
// with no spelling at all.
#define DIFF_MDSPAN_USE_STD 1
#else
#define DIFF_MDSPAN_USE_STD 0
#endif

#if DIFF_MDSPAN_USE_STD
#include <mdspan>
#define DIFF_MDSPAN_NS std
// P2642's padded layouts rode in with the C++26 mdspan revision.
#if __cpp_lib_mdspan >= 202406L
#define DIFF_MDSPAN_HAS_PADDED 1
#else
#define DIFF_MDSPAN_HAS_PADDED 0
#endif
#else
// Before the include: otherwise the vendored header plants itself in ::std,
// which is UB and collides with a real <mdspan> in the same TU.
#ifndef MDSPAN_IMPL_STANDARD_NAMESPACE
#define MDSPAN_IMPL_STANDARD_NAMESPACE diff_mdspan
#endif
#include "third_party/mdspan.hpp"
#define DIFF_MDSPAN_NS MDSPAN_IMPL_STANDARD_NAMESPACE
#define DIFF_MDSPAN_HAS_PADDED 1
#endif

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace diff::impl::md {

using DIFF_MDSPAN_NS::dextents;
using DIFF_MDSPAN_NS::dynamic_extent;
using DIFF_MDSPAN_NS::extents;
using DIFF_MDSPAN_NS::mdspan;

using DIFF_MDSPAN_NS::default_accessor;
using DIFF_MDSPAN_NS::layout_left;
using DIFF_MDSPAN_NS::layout_right;
using DIFF_MDSPAN_NS::layout_stride;

using DIFF_MDSPAN_NS::full_extent;
using DIFF_MDSPAN_NS::full_extent_t;
using DIFF_MDSPAN_NS::strided_slice;
using DIFF_MDSPAN_NS::submdspan;
using DIFF_MDSPAN_NS::submdspan_extents;
// submdspan_mapping is not re-exported: it is found by ADL on the mapping type,
// so a custom layout supplies it as a hidden friend.

#if DIFF_MDSPAN_HAS_PADDED
using DIFF_MDSPAN_NS::layout_left_padded;
using DIFF_MDSPAN_NS::layout_right_padded;
#endif

namespace detail {
template <typename T> inline constexpr bool is_extents_v = false;
template <typename I, std::size_t... E>
inline constexpr bool is_extents_v<extents<I, E...>> = true;

} // namespace detail

template <typename E>
concept CExtents = detail::is_extents_v<std::remove_cvref_t<E>>;

// Every extent is known at compile time, so a tensor over it can own a
// std::array and stay usable in constant evaluation.
template <typename E>
concept CStaticExtents =
    CExtents<E> && (std::remove_cvref_t<E>::rank_dynamic() == 0);

// A layout mapping, checked against the extents it will carry.  There is no
// unparameterised CLayoutPolicy: the interesting layouts here are
// rank-constrained, so probing them with arbitrary extents would hard-error.
template <typename M, typename E>
concept CLayoutMappingOf =
    CExtents<E> && std::copyable<M> && std::equality_comparable<M> &&
    requires(const M m) {
      typename M::extents_type;
      typename M::index_type;
      typename M::rank_type;
      typename M::layout_type;
      // No `M(e)`: layout_stride's mapping needs the strides too.
      { m.extents() } -> std::convertible_to<E>;
      { m.required_span_size() } -> std::convertible_to<typename M::index_type>;
      { M::is_always_unique() } -> std::same_as<bool>;
      { M::is_always_exhaustive() } -> std::same_as<bool>;
      { M::is_always_strided() } -> std::same_as<bool>;
      { m.is_unique() } -> std::same_as<bool>;
      { m.is_exhaustive() } -> std::same_as<bool>;
      { m.is_strided() } -> std::same_as<bool>;
    };

template <typename P, typename E>
concept CLayoutFor = CExtents<E> && requires {
  typename P::template mapping<E>;
} && CLayoutMappingOf<typename P::template mapping<E>, E>;

static_assert(CExtents<extents<std::size_t, 2, 3>>);
static_assert(CStaticExtents<extents<std::size_t, 2, 3>>);
static_assert(!CStaticExtents<dextents<std::size_t, 2>>);
static_assert(CLayoutFor<layout_right, extents<std::size_t, 2, 3>>);
static_assert(CLayoutFor<layout_left, dextents<std::size_t, 2>>);
static_assert(CLayoutFor<layout_stride, dextents<std::size_t, 2>>);

} // namespace diff::impl::md
