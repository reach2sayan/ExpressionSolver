#pragma once
#include <utility> // std::forward, for DIFF_FWD_SELF

#if !defined(DIFF_DEDUCING_THIS)
#if defined(__cpp_explicit_this_parameter)
#define DIFF_DEDUCING_THIS 1
#else
#define DIFF_DEDUCING_THIS 0
#endif
#endif

#if DIFF_DEDUCING_THIS && defined(__GNUC__) && !defined(__clang__) &&          \
    !defined(__cpp_explicit_this_parameter)
#error "DIFF_DEDUCING_THIS=1, but this GCC does not implement P0847 (needs 14+)."
#endif

#if DIFF_DEDUCING_THIS
#define DIFF_SELF this auto &&self
#define DIFF_FWD_SELF std::forward<decltype(self)>(self)
#endif

// The sweep helpers (reverse_sweep, color_sweeps) and the dual kernels exist to
// stop the same lines being written six times; they are not meant to be calls.
// GCC stops inlining them once a TU exhausts its inlining budget, so the
// decision is stated here rather than left to one.
#if defined(__GNUC__) || defined(__clang__)
#define DIFF_ALWAYS_INLINE [[gnu::always_inline]] inline
#elif defined(_MSC_VER)
#define DIFF_ALWAYS_INLINE __forceinline
#else
#define DIFF_ALWAYS_INLINE inline
#endif

// DIFF_KEYED_ACCESSORS -- both spellings of a keyed accessor, in whichever form
// this toolchain takes.
//
// Three classes hand out a slot by a compile-time key: ValueMap and Bound by
// symbol (bound.hpp), Equation by index (equation.hpp).  Each offers get<Key>()
// and the subscript spelling in the value category of the object it was called
// on, and each delegates to a private static slot(auto &&self).  Only how `self`
// is passed and how the key is named vary, so both are parameters here and the
// branch below is the only place the two forms are described.
//
//   GET_TPARAMS  template parameter list for get<...>()
//   SUB_TPARAMS  template parameter list for operator[] -- not always the same,
//                since operator[] deduces its key from an argument
//   KEY          what to pass to slot<...>
//   SUB_PARAM    the empty tag operator[] takes (symbol_type<S>, idx_t<N>);
//                operator[] has no template-argument syntax, so the key has to
//                arrive as a value
//   ...          trailing requires-clause, if the class needs one
#if DIFF_DEDUCING_THIS
#define DIFF_KEYED_ACCESSORS(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, ...)    \
  template <GET_TPARAMS>                                                       \
  [[nodiscard]] constexpr decltype(auto) get(DIFF_SELF) noexcept __VA_ARGS__ { \
    return slot<KEY>(DIFF_FWD_SELF);                                           \
  }                                                                            \
  template <SUB_TPARAMS>                                                       \
  [[nodiscard]] constexpr decltype(auto) operator[](DIFF_SELF, SUB_PARAM)      \
      noexcept __VA_ARGS__ {                                                   \
    return slot<KEY>(DIFF_FWD_SELF);                                           \
  }
#else
// One value category; DIFF_KEYED_ACCESSORS stamps it out four times.
#define DIFF_KEYED_ACCESSOR_QUAL(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM,     \
                                 QUAL, SELF, ...)                              \
  template <GET_TPARAMS>                                                       \
  [[nodiscard]] constexpr decltype(auto) get() QUAL noexcept __VA_ARGS__ {     \
    return slot<KEY>(SELF);                                                    \
  }                                                                            \
  template <SUB_TPARAMS>                                                       \
  [[nodiscard]] constexpr decltype(auto) operator[](SUB_PARAM)                 \
      QUAL noexcept __VA_ARGS__ {                                              \
    return slot<KEY>(SELF);                                                    \
  }
#define DIFF_KEYED_ACCESSORS(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, ...)    \
  DIFF_KEYED_ACCESSOR_QUAL(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, &, *this, \
                           __VA_ARGS__)                                        \
  DIFF_KEYED_ACCESSOR_QUAL(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, const &,  \
                           *this, __VA_ARGS__)                                 \
  DIFF_KEYED_ACCESSOR_QUAL(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, &&,       \
                           std::move(*this), __VA_ARGS__)                      \
  DIFF_KEYED_ACCESSOR_QUAL(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, const &&, \
                           std::move(*this), __VA_ARGS__)
#endif
