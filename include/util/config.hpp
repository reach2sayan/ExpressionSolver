#pragma once
#include <utility>

#if !defined(DDX_DEDUCING_THIS)
#if defined(__cpp_explicit_this_parameter)
#define DDX_DEDUCING_THIS 1
#else
#define DDX_DEDUCING_THIS 0
#endif
#endif

#if DDX_DEDUCING_THIS && defined(__GNUC__) && !defined(__clang__) &&           \
    !defined(__cpp_explicit_this_parameter)
#error "DDX_DEDUCING_THIS=1, but this GCC does not implement P0847 (needs 14+)."
#endif

#if DDX_DEDUCING_THIS
#define DDX_SELF this auto &&self
#define DDX_FWD_SELF std::forward<decltype(self)>(self)
#endif

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

// get<Key>() for a class with a private static slot(auto &&self).  TPARAMS is
// the template-parameter list; ... is a trailing requires-clause.
#if DDX_DEDUCING_THIS
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
#else
// One value category; each spelling is stamped out four times below.
#define DDX_KEYED_GET_QUAL(TPARAMS, KEY, QUAL, SELF, ...)                      \
  template <TPARAMS>                                                           \
  [[nodiscard]] constexpr decltype(auto) get() QUAL noexcept __VA_ARGS__ {     \
    return slot<KEY>(SELF);                                                    \
  }
#define DDX_KEYED_SUBSCRIPT_QUAL(TPARAMS, KEY, SUB_PARAM, QUAL, SELF, ...)     \
  template <TPARAMS>                                                           \
  [[nodiscard]] constexpr decltype(auto) operator[](SUB_PARAM)                 \
      QUAL noexcept __VA_ARGS__ {                                              \
    return slot<KEY>(SELF);                                                    \
  }
#define DDX_KEYED_GET(TPARAMS, KEY, ...)                                       \
  DDX_KEYED_GET_QUAL(TPARAMS, KEY, &, *this, __VA_ARGS__)                      \
  DDX_KEYED_GET_QUAL(TPARAMS, KEY, const &, *this, __VA_ARGS__)                \
  DDX_KEYED_GET_QUAL(TPARAMS, KEY, &&, std::move(*this), __VA_ARGS__)          \
  DDX_KEYED_GET_QUAL(TPARAMS, KEY, const &&, std::move(*this), __VA_ARGS__)
#define DDX_KEYED_SUBSCRIPT(TPARAMS, KEY, SUB_PARAM, ...)                      \
  DDX_KEYED_SUBSCRIPT_QUAL(TPARAMS, KEY, SUB_PARAM, &, *this, __VA_ARGS__)     \
  DDX_KEYED_SUBSCRIPT_QUAL(TPARAMS, KEY, SUB_PARAM, const &, *this,            \
                           __VA_ARGS__)                                        \
  DDX_KEYED_SUBSCRIPT_QUAL(TPARAMS, KEY, SUB_PARAM, &&, std::move(*this),      \
                           __VA_ARGS__)                                        \
  DDX_KEYED_SUBSCRIPT_QUAL(TPARAMS, KEY, SUB_PARAM, const &&,                  \
                           std::move(*this), __VA_ARGS__)
#endif

// Both spellings of one slot.  The two parameter lists differ because get<>
// takes its key and operator[] deduces it from the tag.
#define DDX_KEYED_ACCESSORS(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, ...)     \
  DDX_KEYED_GET(GET_TPARAMS, KEY, __VA_ARGS__)                                 \
  DDX_KEYED_SUBSCRIPT(SUB_TPARAMS, KEY, SUB_PARAM, __VA_ARGS__)
