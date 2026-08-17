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
