#pragma once

#include <algorithm>
#include <compare>
#include <cstddef>
#include <string_view>

// ===========================================================================
// FixedString — a string literal usable as a non-type template parameter.
// ===========================================================================

namespace diff {

template <std::size_t N> struct FixedString {
  static_assert(N > 0, "FixedString: N counts the terminating NUL, so N >= 1");

  // Public by necessity — see the structural-type note above.
  char data[N];

  // Only ever constructed from a literal in a constant expression (a template
  // argument), so this is consteval: a FixedString that is not fully known at
  // compile time would defeat the point of the type.
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

// ===========================================================================
// The NTTP contract.
//
// Everything in this library rests on FixedString being a *structural* type:
// lose that and a symbol stops being a type, and the failure surfaces at some
// user's var<"x"> rather than here.  There is no std trait for "structural", so
// the contract is pinned two ways — the checkable requirements as concepts, and
// an actual template-id, which is what the language really tests.
// ===========================================================================

namespace detail {

// A structural class type has no virtual anything and is a literal type.
template <typename T>
concept CStructuralClass =
    std::is_class_v<T> && !std::is_polymorphic_v<T> && !std::is_abstract_v<T> &&
    std::is_trivially_destructible_v<T> && std::is_trivially_copyable_v<T>;

// ...and every non-static data member is public and non-mutable.  No trait sees
// either property, but both are observable: a private member is inaccessible
// here, and a mutable one stays non-const through a const object.
template <typename T>
concept CPublicImmutableChars = requires(T s, const T cs) {
  { s.data[0] } -> std::same_as<char &>;
  { cs.data[0] } -> std::same_as<const char &>;
};

// A base class or an extra member would keep every trait above satisfied while
// still changing the object, so pin the layout too.
template <typename T>
concept CIsExactlyItsChars = (sizeof(T) == T::size() + 1);

template <typename T>
concept CUsableAsNttp =
    CStructuralClass<T> && CPublicImmutableChars<T> && CIsExactlyItsChars<T>;

// Representative instantiations: the class body does not vary with N, so a
// contract that holds for the empty string and for a normal symbol holds for
// every length.
static_assert(CUsableAsNttp<FixedString<1>>,
              "FixedString is no longer a structural type and can no longer be "
              "a non-type template parameter: check that `data` is still a "
              "public, non-mutable char array, that nothing was added to the "
              "class, and that it gained no base, no virtual member and no "
              "user-declared destructor");
static_assert(CUsableAsNttp<FixedString<2>>, "see above");
static_assert(CUsableAsNttp<FixedString<8>>, "see above");

// The end-to-end check the concepts above only approximate: forming this
// template-id is the language's own test of structural-ness.
template <FixedString S> struct nttp_probe {
  static constexpr auto label = S;
};

static_assert(nttp_probe<"xy">::label == FixedString{"xy"});
// Same spelling must be the same type — this is what makes a symbol a type,
// and so what lets Variable<double, "x"> carry no bytes.
static_assert(std::same_as<nttp_probe<"xy">, nttp_probe<FixedString{"xy"}>>);
// ...and different spellings must be different types, or two symbols would
// silently collide into one slot.
static_assert(!std::same_as<nttp_probe<"xy">, nttp_probe<"xz">>);
static_assert(!std::same_as<nttp_probe<"x">, nttp_probe<"xy">>);

} // namespace detail

} // namespace diff
