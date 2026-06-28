#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

namespace diff {
template <auto S> struct symbol_type;
} // namespace diff

namespace diff::mpl {

template <class... T> struct mp_list {};
template <class... T> mp_list(T...) -> mp_list<T...>;

template <bool B> using mp_bool = std::integral_constant<bool, B>;

namespace detail {
template <std::size_t I, class... T>
auto at_c(mp_list<T...>)
    -> std::type_identity<std::tuple_element_t<I, std::tuple<T...>>>;
} // namespace detail

template <class L, std::size_t I>
using mp_at_c = typename decltype(detail::at_c<I>(L{}))::type;

// mp_size(list) -> element count.
template <class... T> consteval std::size_t mp_size(mp_list<T...>) noexcept {
  return sizeof...(T);
}

template <class V, class... T>
consteval std::size_t mp_find(V, mp_list<T...>) noexcept {
  const std::array<bool, sizeof...(T)> hit{std::is_same_v<T, V>...};
  return static_cast<std::size_t>(std::find(hit.begin(), hit.end(), true) -
                                  hit.begin());
}

template <auto Needle, class... T>
consteval std::size_t mp_find(mp_list<T...> list) noexcept {
  return mp_find(::diff::symbol_type<Needle>{}, list);
}

namespace detail {

template <class L> struct box {
  using type = L;
};

template <class... A, class... B>
auto operator+(box<mp_list<A...>>, box<mp_list<B...>>)
    -> box<mp_list<A..., B...>>;

template <class... L>
auto append_fold() -> decltype((box<mp_list<>>{} + ... + box<L>{}));

} // namespace detail

template <class... L>
using mp_append = typename decltype(detail::append_fold<L...>())::type;

namespace detail {

// A selection/permutation of indices whose length is known at compile time.
template <std::size_t N> struct index_selection {
  std::array<std::size_t, N> indices{};
  std::size_t count = 0;
};

template <class L, auto Sel, std::size_t... Is>
auto rebuild(std::index_sequence<Is...>)
    -> std::type_identity<mp_list<mp_at_c<L, Sel.indices[Is]>...>>;

template <class L, auto Sel>
using rebuild_t = typename decltype(rebuild<L, Sel>(
    std::make_index_sequence<Sel.count>{}))::type;

// same[i*N + j] == (element i is the same type as element j).
template <class... T>
consteval std::array<bool, sizeof...(T) * sizeof...(T)>
same_matrix(mp_list<T...>) {
  std::array<bool, sizeof...(T) * sizeof...(T)> m{};
  std::size_t k = 0;
  [[maybe_unused]] auto row = [&]<class R>() {
    ((m[k++] = std::is_same_v<R, T>), ...);
  };
  (row.template operator()<T>(), ...);
  return m;
}

// less[i*N + j] == P<element i, element j>::value.
template <template <class...> class P, class... T>
consteval std::array<bool, sizeof...(T) * sizeof...(T)>
less_matrix(mp_list<T...>) {
  std::array<bool, sizeof...(T) * sizeof...(T)> m{};
  std::size_t k = 0;
  [[maybe_unused]] auto row = [&]<class R>() {
    ((m[k++] = P<R, T>::value), ...);
  };
  (row.template operator()<T>(), ...);
  return m;
}

// Indices of first-occurrence elements (keep-first dedup).
template <class L> consteval auto unique_selection() {
  constexpr std::size_t N = mp_size(L{});
  constexpr std::array<bool, N * N> same = same_matrix(L{});
  index_selection<N> sel{};
  for (std::size_t i = 0; i < N; ++i) {
    // Keep i unless some earlier element j < i has the same type.
    const bool *row = same.data() + i * N;
    if (std::none_of(row, row + i, [](bool eq) { return eq; }))
      sel.indices[sel.count++] = i;
  }
  return sel;
}

// Stable sort of the indices [0..N) by the predicate P.
template <class L, template <class...> class P>
consteval auto sort_selection() {
  constexpr std::size_t N = mp_size(L{});
  constexpr std::array<bool, N * N> less = less_matrix<P>(L{});
  index_selection<N> sel{};
  std::iota(sel.indices.begin(), sel.indices.end(), std::size_t{0});
  sel.count = N;
  // std::sort is constexpr in C++20; std::stable_sort is not, so make the
  // comparator a total order (break ties on original index) to stay stable.
  std::sort(sel.indices.begin(), sel.indices.end(),
            [&](std::size_t a, std::size_t b) {
              if (less[a * N + b]) {
                return true;
              } else if (less[b * N + a]) {
                return false;
              }
              return a < b;
            });
  return sel;
}

} // namespace detail

template <class L>
using mp_unique = detail::rebuild_t<L, detail::unique_selection<L>()>;

template <class L, template <class...> class P>
using mp_sort = detail::rebuild_t<L, detail::sort_selection<L, P>()>;

} // namespace diff::mpl
