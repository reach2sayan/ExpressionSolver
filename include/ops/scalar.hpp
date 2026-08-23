#pragma once

#include "ops/numeric.hpp"

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace ddx::impl {

template <Numeric T> class Dual;

template <typename T> inline constexpr bool is_dual_v = false;
template <Numeric T> inline constexpr bool is_dual_v<Dual<T>> = true;

template <typename X>
concept DualLike = is_dual_v<std::remove_cvref_t<X>>;

template <Numeric T> struct dual_scalar_type {
  using type = T;
};
template <Numeric T> struct dual_scalar_type<Dual<T>> {
  using type = T;
};
template <Numeric T> using dual_scalar_t = typename dual_scalar_type<T>::type;

template <DualLike X> struct dual_value_type;
template <Numeric T> struct dual_value_type<Dual<T>> {
  using type = T;
};
template <DualLike X>
using dual_value_t = typename dual_value_type<std::remove_cvref_t<X>>::type;

template <typename A, typename B>
concept DualCompatible = DualLike<A> && DualLike<B> &&
                         std::same_as<dual_value_t<A>, dual_value_t<B>>;

// N of them nested: 2^N components, one per subset of the ε's, and the all-ones
// component is the Nth derivative -- what extract_nth reads and
// make_mixed_seed seeds for.  Ref: Fike & Alonso, AIAA 2011-886;
// docs/hyperdual_nth_order_by_example.md draws the lattice.
template <Numeric T, std::size_t N> consteval auto nth_dual_impl() noexcept {
  if constexpr (N == 0) {
    return std::type_identity<T>{};
  } else {
    using Inner = typename decltype(nth_dual_impl<T, N - 1>())::type;
    return std::type_identity<Dual<Inner>>{};
  }
}

template <Numeric T, std::size_t N>
using nth_dual_t = typename decltype(nth_dual_impl<T, N>())::type;

template <Numeric T> inline constexpr std::size_t dual_depth_v = 0;
template <Numeric T>
inline constexpr std::size_t dual_depth_v<Dual<T>> = 1 + dual_depth_v<T>;

template <Numeric T> auto scalar_base_impl(std::type_identity<T>) -> T;
template <Numeric T>
auto scalar_base_impl(std::type_identity<Dual<T>>)
    -> decltype(scalar_base_impl(std::type_identity<T>{}));

template <Numeric T>
using scalar_base_t = decltype(scalar_base_impl(std::type_identity<T>{}));

template <Numeric T, std::size_t N>
constexpr nth_dual_t<T, N> embed_constant(T val) noexcept {
  if constexpr (N == 0) {
    return val;
  } else {
    return nth_dual_t<T, N>{embed_constant<T, N - 1>(val),
                            nth_dual_t<T, N - 1>{}};
  }
}

template <Numeric U> struct ConstantEmbedder {
  static constexpr U embed(scalar_base_t<U> val) noexcept {
    return embed_constant<scalar_base_t<U>, dual_depth_v<U>>(val);
  }
};

template <std::size_t N, Numeric T>
constexpr auto get_real_part(const T &x) noexcept {
  if constexpr (N == 0) {
    return x;
  } else {
    return get_real_part<N - 1>(x.template get<0>());
  }
}

template <typename X>
concept DualOrArithmetic = DualLike<X> || CArithmetic<X>;

// The scalar ends of the recursions dual/dual.hpp continues for a Dual.
template <CArithmetic T> constexpr T val(T x) noexcept { return x; }

// Zero in every component; operator== compares val() alone and cannot say this.
template <CArithmetic T> constexpr bool all_zero(T x) noexcept {
  return x == T{};
}

template <Numeric X> constexpr double to_double(const X &x) noexcept {
  return static_cast<double>(val(x));
}

} // namespace ddx::impl

namespace std {
template <ddx::impl::Numeric T>
struct tuple_size<ddx::impl::Dual<T>> : integral_constant<std::size_t, 2> {};
template <ddx::impl::Numeric T, std::size_t N>
struct tuple_element<N, ddx::impl::Dual<T>> {
  using type = T;
};
} // namespace std
