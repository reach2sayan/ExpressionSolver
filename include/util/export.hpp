#pragma once

// Linkage for the handful of ddx functions that are compiled rather than
// instantiated.  DDX_STATIC_LIB says the archive was chosen, and none of this
// applies; DDX_BUILDING is defined only while libddx itself compiles, which is
// what lets one declaration read as an export there and an import in a
// consumer -- a distinction MSVC insists on and ELF does not.
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

// The JIT's spelling of the same thing; one library now, so the same macro.
#define DDX_JIT_API DDX_API
