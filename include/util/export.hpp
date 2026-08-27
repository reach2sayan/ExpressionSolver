#pragma once

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

#define DDX_JIT_API DDX_API
