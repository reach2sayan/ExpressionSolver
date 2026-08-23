#pragma once

// Linkage for libddx, the one binary the project produces.  Almost all of ddx
// is headers, where linkage is not a question anyone has to answer; this is for
// the handful of functions that are compiled rather than instantiated.
//
// Two macros come from the build.  DDX_STATIC_LIB says the archive was chosen,
// and then none of this applies.  DDX_BUILDING is defined only while libddx
// itself compiles, which is what lets one declaration read as an export there
// and an import in a consumer -- a distinction MSVC insists on and the ELF
// toolchains do not.
#if defined(DDX_STATIC_LIB)
#define DDX_API
#elif defined(_MSC_VER)
#if defined(DDX_BUILDING)
#define DDX_API __declspec(dllexport)
#else
#define DDX_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define DDX_API __attribute__((visibility("default")))
#else
#define DDX_API
#endif

// The JIT's spelling of the same thing.  It is one library now, so these are
// the same macro; the name is kept because jit/kernel.hpp is a public header
// and its declarations read better saying which half of ddx they belong to.
#define DDX_JIT_API DDX_API
