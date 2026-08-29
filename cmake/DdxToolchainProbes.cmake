# The floor the toolchain must clear: <expected> and deducing this.  The macro
# rather than the syntax -- Clang 18 and 19 compile it but do not advertise it.
include_guard(GLOBAL)

include(CheckCXXSourceCompiles)

set(CMAKE_REQUIRED_FLAGS "-std=c++23")
check_cxx_source_compiles(
        "#include <expected>
         #ifndef __cpp_explicit_this_parameter
         #error no deducing this
         #endif
         struct S { int f(this S) { return 0; } };
         int main() { return std::expected<int, int>{S{}.f()}.value(); }"
        DDX_TOOLCHAIN_OK)
unset(CMAKE_REQUIRED_FLAGS)

if (NOT MSVC AND NOT DDX_TOOLCHAIN_OK)
    message(FATAL_ERROR
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} cannot build ddx, which "
            "needs <expected> and deducing this (P0847).  Build with GCC 14+ or Clang 20+.  "
            "The compiler's own diagnostic is in the CMakeConfigureLog.")
endif ()
