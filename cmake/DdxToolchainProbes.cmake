# What the toolchain can do, asked once and answered in cache variables.
# Compiles nothing of ours; the top-level CMakeLists.txt consumes the results.
#
#   DDX_HAS_STD_EXPECTED       hard requirement, fatal if absent
#   DDX_HAS_DEDUCING_THIS      hard requirement, fatal if absent
#   DDX_MDSPAN_VENDORED        set when md.hpp must bind to the vendored copy
include_guard(GLOBAL)

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-std=c++23")

# libstdc++ gates <expected> on __cpp_concepts 202002L, which Clang first
# defines in 19.  MSVC is exempt: the probe's -std= is not its spelling.
# libc++ is not a supported standard library -- it has no views::enumerate.
check_cxx_source_compiles(
        "#include <expected>
         int main() { return std::expected<int, int>{0}.value(); }"
        DDX_HAS_STD_EXPECTED)
if (NOT MSVC AND NOT DDX_HAS_STD_EXPECTED)
    message(FATAL_ERROR
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} has no usable <expected>.  "
            "Build with GCC 14+ or Clang 20+.")
endif ()

# --- deducing this ----------------------------------------------------------
# A hard requirement: include/util/config.hpp writes every accessor once as an
# explicit object parameter, so there is no fallback spelling to select.  The
# macro, not the syntax: Clang 18 and 19 compile an explicit object parameter
# but do not advertise it until 20, and config.hpp gates on the macro.
check_cxx_source_compiles(
        "struct S { int f(this S) { return 0; } };
         #ifndef __cpp_explicit_this_parameter
         #error no macro
         #endif
         int main() { return S{}.f(); }"
        DDX_HAS_DEDUCING_THIS)
if (NOT MSVC AND NOT DDX_HAS_DEDUCING_THIS)
    message(FATAL_ERROR
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} does not implement "
            "deducing this (P0847).  Build with GCC 14+, Clang 20+, or MSVC 19.32+.")
endif ()

# --- which mdspan include/md/md.hpp binds to --------------------------------
#   auto     - standard <mdspan> when complete enough, vendored otherwise
#   std      - require the standard one; hard error if incomplete
#   vendored - always the copy under include/md/third_party
set(DDX_MDSPAN_MODE "auto" CACHE STRING "Which mdspan md.hpp binds to: auto|std|vendored")
set_property(CACHE DDX_MDSPAN_MODE PROPERTY STRINGS auto std vendored)

check_cxx_source_compiles(
        "#include <version>
         #if !defined(__cpp_lib_mdspan) || __cpp_lib_mdspan < 202207L
         #error no mdspan
         #endif
         #if !defined(__cpp_lib_submdspan) || __cpp_lib_submdspan < 202306L
         #error no submdspan
         #endif
         #include <mdspan>
         int main() { double a[2]{}; std::mdspan m(a, 2); return (int)m.extent(0) - 2; }"
        DDX_HAS_STD_MDSPAN)
unset(CMAKE_REQUIRED_FLAGS)

if (DDX_MDSPAN_MODE STREQUAL "std" AND NOT DDX_HAS_STD_MDSPAN)
    message(FATAL_ERROR
            "DDX_MDSPAN_MODE=std but ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} "
            "has no complete <mdspan>/<submdspan>.  Use auto or vendored.")
elseif (DDX_MDSPAN_MODE STREQUAL "vendored" OR NOT DDX_HAS_STD_MDSPAN)
    set(DDX_MDSPAN_VENDORED ON)
    message(STATUS "mdspan: vendored (include/md/third_party/mdspan.hpp)")
else ()
    message(STATUS "mdspan: standard <mdspan>")
endif ()
