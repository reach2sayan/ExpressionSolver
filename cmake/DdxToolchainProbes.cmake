# The floor the toolchain must clear, asked once.  Compiles nothing of ours;
# nothing downstream reads a result, because there is no second path to take.
include_guard(GLOBAL)

include(CheckCXXSourceCompiles)

# Both halves of the floor in one probe: <expected>, because every ddx error
# comes back as a value, and __cpp_explicit_this_parameter, because
# util/config.hpp writes every accessor as an explicit object parameter.  The
# macro rather than the syntax -- Clang 18 and 19 compile one but do not
# advertise it until 20.  MSVC is exempt: the probe's -std= is not its spelling.
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
