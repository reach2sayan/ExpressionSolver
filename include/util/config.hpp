#pragma once
#include <utility>

// P0847 is a hard requirement: every accessor below is written once as an
// explicit object parameter.  Clang compiles the syntax from 18 but only
// defines the macro from 19, and 19 is the floor for <expected> anyway.
#if !defined(__cpp_explicit_this_parameter)
#error "ddx needs deducing this (P0847): GCC 14+, Clang 19+, MSVC 19.32+."
#endif

#define DDX_SELF this auto &&self
#define DDX_FWD_SELF std::forward<decltype(self)>(self)

// The sweep helpers and dual kernels are factored-out code, not calls; GCC
// stops inlining them once a TU exhausts its inlining budget.
#if defined(__GNUC__) || defined(__clang__)
#define DDX_ALWAYS_INLINE [[gnu::always_inline]] inline
#elif defined(_MSC_VER)
#define DDX_ALWAYS_INLINE __forceinline
#else
#define DDX_ALWAYS_INLINE inline
#endif

// What the JIT spells NoAlias on the kernel's columns.  A tape reached through
// a span carries no such promise, and without it every lane store is assumed
// to land in the node array the next lane load reads.
#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define DDX_RESTRICT __restrict
#else
#define DDX_RESTRICT
#endif

// Accessors for a class with a private static slot(auto &&self).  ... is a
// trailing requires-clause.

// One slot under a name of its own: no key parameter, the name is the key.
#define DDX_SLOT_ACCESSOR(NAME, KEY)                                           \
  [[nodiscard]] constexpr decltype(auto) NAME(DDX_SELF) noexcept {             \
    return slot<KEY>(DDX_FWD_SELF);                                            \
  }
#define DDX_KEYED_GET(TPARAMS, KEY, ...)                                       \
  template <TPARAMS>                                                           \
  [[nodiscard]] constexpr decltype(auto) get(DDX_SELF) noexcept __VA_ARGS__ {  \
    return slot<KEY>(DDX_FWD_SELF);                                            \
  }
// The same slot, reached through the empty tag operator[] deduces its key from.
#define DDX_KEYED_SUBSCRIPT(TPARAMS, KEY, SUB_PARAM, ...)                      \
  template <TPARAMS>                                                           \
  [[nodiscard]] constexpr decltype(auto) operator[](                           \
      DDX_SELF, SUB_PARAM) noexcept __VA_ARGS__ {                              \
    return slot<KEY>(DDX_FWD_SELF);                                            \
  }
// Both spellings of one slot.  The two parameter lists differ because get<>
// takes its key and operator[] deduces it from the tag.
#define DDX_KEYED_ACCESSORS(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, ...)     \
  DDX_KEYED_GET(GET_TPARAMS, KEY, __VA_ARGS__)                                 \
  DDX_KEYED_SUBSCRIPT(SUB_TPARAMS, KEY, SUB_PARAM, __VA_ARGS__)
