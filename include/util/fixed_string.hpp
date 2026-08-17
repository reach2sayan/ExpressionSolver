#pragma once

#include <algorithm>
#include <compare>
#include <cstddef>
#include <string_view>

namespace diff {

template <std::size_t N> struct FixedString {
  static_assert(N > 0, "FixedString: N counts the terminating NUL, so N >= 1");

  // Public by necessity — see the structural-type note above.
  char data[N];
  consteval FixedString(const char (&str)[N]) noexcept {
    std::copy_n(str, N, data);
  }

  // Length in characters, excluding the terminating NUL.
  [[nodiscard]] static constexpr std::size_t size() noexcept { return N - 1; }
  [[nodiscard]] static constexpr bool empty() noexcept { return size() == 0; }

  // constexpr, not consteval: printing a symbol reads this at run time.
  [[nodiscard]] constexpr std::string_view view() const noexcept {
    return {data, size()};
  }
  [[nodiscard]] constexpr const char *c_str() const noexcept { return data; }

  constexpr bool operator==(const FixedString &) const noexcept = default;
  template <std::size_t M>
  constexpr bool operator==(const FixedString<M> &) const noexcept {
    return false;
  }

  template <std::size_t M>
  [[nodiscard]] constexpr std::strong_ordering
  operator<=>(const FixedString<M> &other) const noexcept {
    return view() <=> other.view();
  }
};

template <std::size_t N> FixedString(const char (&)[N]) -> FixedString<N>;

namespace detail {
template <typename T> inline constexpr bool is_fixed_string_v = false;
template <std::size_t N>
inline constexpr bool is_fixed_string_v<FixedString<N>> = true;
} // namespace detail

template <typename T>
concept CFixedString = detail::is_fixed_string_v<T>;

namespace detail {

template <typename T>
concept CIsExactlyItsChars = (sizeof(T) == T::size() + 1);

static_assert(CIsExactlyItsChars<FixedString<1>>,
              "FixedString has grown beyond its character array: check that "
              "nothing was added to the class and that it gained no base");
static_assert(CIsExactlyItsChars<FixedString<2>>, "see above");
static_assert(CIsExactlyItsChars<FixedString<8>>, "see above");

template <FixedString S> struct nttp_probe {
  static constexpr auto label = S;
};

static_assert(nttp_probe<"xy">::label == FixedString{"xy"});
static_assert(std::same_as<nttp_probe<"xy">, nttp_probe<FixedString{"xy"}>>);
static_assert(!std::same_as<nttp_probe<"xy">, nttp_probe<"xz">>);
static_assert(!std::same_as<nttp_probe<"x">, nttp_probe<"xy">>);

} // namespace detail

} // namespace diff
