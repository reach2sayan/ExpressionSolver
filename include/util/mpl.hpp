#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

namespace diff {
template <auto S> struct symbol_type;

// A symbol lifted to a type (see expressions.hpp).  Spelled here rather than
// next to the definition because the list vocabulary below is what needs it.
template <typename T> inline constexpr bool is_symbol_v = false;
template <auto S> inline constexpr bool is_symbol_v<symbol_type<S>> = true;

template <typename T>
concept CSymbol = is_symbol_v<std::remove_cvref_t<T>>;
} // namespace diff

namespace diff::mpl {

template <class... T> struct mp_list {};
template <class... T> mp_list(T...) -> mp_list<T...>;

namespace detail {
template <typename T> inline constexpr bool is_mp_list_v = false;
template <typename... T>
inline constexpr bool is_mp_list_v<mp_list<T...>> = true;

template <typename T> inline constexpr bool is_symbol_list_v = false;
template <typename... T>
inline constexpr bool is_symbol_list_v<mp_list<T...>> =
    (::diff::CSymbol<T> && ...);
} // namespace detail

template <typename L>
concept CTypeList = detail::is_mp_list_v<std::remove_cvref_t<L>>;

// A type ordering, for mp_sort.
template <typename Less, typename A, typename B>
concept CTypeOrder = requires(Less less) {
  { less.template operator()<A, B>() } -> std::same_as<bool>;
};

namespace detail {
template <std::size_t I, class... T>
auto at_c(mp_list<T...>)
    -> std::type_identity<std::tuple_element_t<I, std::tuple<T...>>>;
} // namespace detail

template <CTypeList L, std::size_t I>
using mp_at_c = typename decltype(detail::at_c<I>(L{}))::type;

template <typename... T> consteval std::size_t mp_size(mp_list<T...>) noexcept {
  return sizeof...(T);
}

template <::diff::CSymbol V, typename... T>
consteval std::size_t mp_find(V, mp_list<T...>) noexcept {
  const std::array<bool, sizeof...(T)> hit{std::is_same_v<T, V>...};
  return static_cast<std::size_t>(std::find(hit.begin(), hit.end(), true) -
                                  hit.begin());
}

template <auto Needle, typename... T>
consteval std::size_t mp_find(mp_list<T...> list) noexcept {
  return mp_find(::diff::symbol_type<Needle>{}, list);
}

namespace detail {

template <class L> struct box {
  using type = L;
};

template <typename... A, typename... B>
auto operator+(box<mp_list<A...>>, box<mp_list<B...>>)
    -> box<mp_list<A..., B...>>;

template <CTypeList... L>
auto append_fold() -> decltype((box<mp_list<>>{} + ... + box<L>{}));

} // namespace detail

template <CTypeList... L>
using mp_append = typename decltype(detail::append_fold<L...>())::type;

namespace detail {

// A selection/permutation of indices whose length is known at compile time.
template <std::size_t N> struct index_selection {
  std::array<std::size_t, N> indices{};
  std::size_t count = 0;
};

template <CTypeList L, auto Sel, std::size_t... Is>
auto rebuild(std::index_sequence<Is...>)
    -> std::type_identity<mp_list<mp_at_c<L, Sel.indices[Is]>...>>;

template <CTypeList L, auto Sel>
using rebuild_t = typename decltype(rebuild<L, Sel>(
    std::make_index_sequence<Sel.count>{}))::type;

// same[i*N + j] == (element i is the same type as element j).
template <typename... T>
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

// less[i*N + j] == Less<element i, element j>().
template <auto Less, typename... T>
  requires(CTypeOrder<decltype(Less), T, T> && ...)
consteval std::array<bool, sizeof...(T) * sizeof...(T)>
less_matrix(mp_list<T...>) {
  std::array<bool, sizeof...(T) * sizeof...(T)> m{};
  std::size_t k = 0;
  [[maybe_unused]] auto row = [&]<typename R>() {
    ((m[k++] = Less.template operator()<R, T>()), ...);
  };
  (row.template operator()<T>(), ...);
  return m;
}

// Indices of first-occurrence elements (keep-first dedup).
template <CTypeList L> consteval auto unique_selection() {
  constexpr std::size_t N = mp_size(L{});
  constexpr std::array<bool, N * N> same = same_matrix(L{});

  std::array<std::size_t, N> rep{};
  for (std::size_t i = 0; i < N; ++i) {
    const bool *row = same.data() + i * N;
    rep[i] = static_cast<std::size_t>(std::find(row, row + N, true) - row);
  }

  index_selection<N> sel{};
  std::iota(sel.indices.begin(), sel.indices.end(), std::size_t{0});
  std::sort(sel.indices.begin(), sel.indices.end(),
            [&](std::size_t a, std::size_t b) {
              return rep[a] != rep[b] ? rep[a] < rep[b] : a < b;
            });
  const auto last = std::unique(
      sel.indices.begin(), sel.indices.end(),
      [&](std::size_t a, std::size_t b) { return rep[a] == rep[b]; });
  sel.count = static_cast<std::size_t>(last - sel.indices.begin());
  return sel;
}

// Stable sort of the indices [0..N) by the consteval predicate Less.
template <CTypeList L, auto Less> consteval auto sort_selection() {
  constexpr std::size_t N = mp_size(L{});
  constexpr std::array<bool, N * N> less = less_matrix<Less>(L{});
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

template <CTypeList L>
using mp_unique = detail::rebuild_t<L, detail::unique_selection<L>()>;

template <CTypeList L, auto Less>
using mp_sort = detail::rebuild_t<L, detail::sort_selection<L, Less>()>;

} // namespace diff::mpl

namespace diff {

// A canonical symbol list: mp_list<symbol_type<...>...>, alphabetical by name.
// This is the `Syms` threaded through every seeded sweep, and the thing that
// turns a variable's label into its slot index — so it is worth naming rather
// than accepting any type at all.
template <typename L>
concept CSymbolList = mpl::detail::is_symbol_list_v<std::remove_cvref_t<L>>;

} // namespace diff
