# What the toolchain can actually do, asked once and answered in cache
# variables.  Everything here runs at configure time and compiles nothing of
# ours; the results are consumed by the top-level CMakeLists.txt.
#
#   DDX_HAS_STD_EXPECTED       hard requirement, fatal if absent
#   DDX_DEDUCING_THIS_VALUE    defined only when DDX_DEDUCING_THIS overrides
#   DDX_MDSPAN_VENDORED        set when md.hpp must bind to the vendored copy
include_guard(GLOBAL)

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-std=c++23")

# --- <expected> -------------------------------------------------------------
# ddx::result is std::expected, and libstdc++ offers <expected> only where
# __cpp_concepts is 202002L -- which Clang first defines in 19.  Unguarded, the
# toolchain floor shows up as a missing name partway down an include stack.
# MSVC is exempt because the probe's -std= is not its spelling.
check_cxx_source_compiles(
        "#include <expected>
         int main() { return std::expected<int, int>{0}.value(); }"
        DDX_HAS_STD_EXPECTED)
if (NOT MSVC AND NOT DDX_HAS_STD_EXPECTED)
    message(FATAL_ERROR
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} has no usable <expected>.  "
            "Build with GCC 14+, Clang 19+ over libstdc++, or Clang 17+ with -stdlib=libc++.")
endif ()

# --- accessor spelling ------------------------------------------------------
# include/util/config.hpp owns the decision; this only overrides it.
#   auto - the compiler's own answer (__cpp_explicit_this_parameter)
#   on   - deducing this even where the macro is absent
#   off  - the ref-qualified overload set
# Two probes: what `auto` picks up, and what `on` would require.
set(DDX_DEDUCING_THIS "auto" CACHE STRING "Accessor spelling: auto|on|off")
set_property(CACHE DDX_DEDUCING_THIS PROPERTY STRINGS auto on off)

check_cxx_source_compiles(
        "struct S { int f(this S) { return 0; } };
         #ifndef __cpp_explicit_this_parameter
         #error no macro
         #endif
         int main() { return S{}.f(); }"
        DDX_HAS_DEDUCING_THIS)
check_cxx_source_compiles(
        "struct S { int f(this S) { return 0; } };
         int main() { return S{}.f(); }"
        DDX_DEDUCING_THIS_COMPILES)

if (DDX_DEDUCING_THIS STREQUAL "on" AND NOT DDX_DEDUCING_THIS_COMPILES)
    message(FATAL_ERROR
            "DDX_DEDUCING_THIS=on but ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} "
            "cannot compile an explicit object parameter.  Use auto or off.")
elseif (DDX_DEDUCING_THIS STREQUAL "on")
    set(DDX_DEDUCING_THIS_VALUE 1)
    set(DDX_ACCESSORS "deducing this, forced on")
elseif (DDX_DEDUCING_THIS STREQUAL "off")
    set(DDX_DEDUCING_THIS_VALUE 0)
    set(DDX_ACCESSORS "ref-qualified overload set, forced off")
elseif (NOT DDX_DEDUCING_THIS STREQUAL "auto")
    message(FATAL_ERROR "DDX_DEDUCING_THIS must be auto, on or off (got '${DDX_DEDUCING_THIS}').")
elseif (DDX_HAS_DEDUCING_THIS)
    # auto leaves the value undefined, so a consumer building without this file agrees.
    set(DDX_ACCESSORS "deducing this")
else ()
    set(DDX_ACCESSORS "ref-qualified overload set")
endif ()
message(STATUS "accessors: ${DDX_ACCESSORS} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")

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
