#pragma once
#include <utility> // std::forward, for DDX_FWD_SELF

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

// get<Key>() and operator[](tag) for a class with a private static
// slot(auto &&self), in whichever form this toolchain takes.  SUB_PARAM is the
// empty tag operator[] deduces its key from; ... is a trailing requires-clause.
#if DDX_DEDUCING_THIS
#define DDX_KEYED_ACCESSORS(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, ...)     \
  template <GET_TPARAMS>                                                       \
  [[nodiscard]] constexpr decltype(auto) get(DDX_SELF) noexcept __VA_ARGS__ {  \
    return slot<KEY>(DDX_FWD_SELF);                                            \
  }                                                                            \
  template <SUB_TPARAMS>                                                       \
  [[nodiscard]] constexpr decltype(auto) operator[](                           \
      DDX_SELF, SUB_PARAM) noexcept __VA_ARGS__ {                              \
    return slot<KEY>(DDX_FWD_SELF);                                            \
  }
#else
// One value category; stamped out four times below.
#define DDX_KEYED_ACCESSOR_QUAL(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM,      \
                                QUAL, SELF, ...)                               \
  template <GET_TPARAMS>                                                       \
  [[nodiscard]] constexpr decltype(auto) get() QUAL noexcept __VA_ARGS__ {     \
    return slot<KEY>(SELF);                                                    \
  }                                                                            \
  template <SUB_TPARAMS>                                                       \
  [[nodiscard]] constexpr decltype(auto) operator[](SUB_PARAM)                 \
      QUAL noexcept __VA_ARGS__ {                                              \
    return slot<KEY>(SELF);                                                    \
  }
#define DDX_KEYED_ACCESSORS(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, ...)     \
  DDX_KEYED_ACCESSOR_QUAL(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, &, *this,  \
                          __VA_ARGS__)                                         \
  DDX_KEYED_ACCESSOR_QUAL(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, const &,   \
                          *this, __VA_ARGS__)                                  \
  DDX_KEYED_ACCESSOR_QUAL(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, &&,        \
                          std::move(*this), __VA_ARGS__)                       \
  DDX_KEYED_ACCESSOR_QUAL(GET_TPARAMS, SUB_TPARAMS, KEY, SUB_PARAM, const &&,  \
                          std::move(*this), __VA_ARGS__)
#endif
