#pragma once

// Linkage for ddx::jit, the one compiled library in the project.  Nothing else
// needs this header: the rest of ddx is header-only, where linkage is not a
// question anyone has to answer.
//
// Two macros come from the build.  DDX_JIT_STATIC says the archive was chosen,
// and then none of this applies.  DDX_JIT_BUILDING is defined only while
// ddx_jit itself compiles, which is what lets one declaration read as an export
// there and an import in a consumer -- a distinction MSVC insists on and the
// ELF toolchains do not.
#if defined(DDX_JIT_STATIC)
#define DDX_JIT_API
#elif defined(_MSC_VER)
#if defined(DDX_JIT_BUILDING)
#define DDX_JIT_API __declspec(dllexport)
#else
#define DDX_JIT_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define DDX_JIT_API __attribute__((visibility("default")))
#else
#define DDX_JIT_API
#endif
