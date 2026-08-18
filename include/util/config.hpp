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

// DIFF_KEYED_ACCESSORS -- both spellings of a keyed accessor, in whichever form
// this toolchain takes.
//
// Three classes hand out a slot by a compile-time key: ValueMap and Bound by
// symbol (bound.hpp), Equation by index (equation.hpp).  Each offers the same
// four things -- get<Key>() and the subscript spelling, each in the value
// category of the object it was called on -- and each delegates to a private
// static slot(auto &&self) that does the real work.  So the only thing that
// varies between the twelve resulting functions is how `self` is passed, and
// the only thing that varies between the three classes is how the key is named.
//
// Spelled out that was three near-identical blocks, each of which had to be
// written twice over (once per DIFF_DEDUCING_THIS branch) and kept in step by
// hand.  Here it is one definition; the branch below is the only place the two
// forms are described.
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
