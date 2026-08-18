cmake_minimum_required(VERSION 3.20)

# The accessors -- Equation::get<N>(), ValueMap::get<S>() and Bound::get<S>(),
# plus their operator[] spellings -- have two forms behind DIFF_DEDUCING_THIS
# (include/util/config.hpp): one definition using deducing this (P0847), and an
# explicit overload set for toolchains without it.  Each pair shares a single
# static slot() body, so they compile and behave identically and nothing here
# is a hard requirement -- but our own builds should exercise the deducing-this
# path, and the default /usr/bin/c++ is still GCC 13, which lacks it.  Pick the
# newest GCC on PATH when the caller has not chosen a compiler; a caller who
# passes -DCMAKE_CXX_COMPILER or sets $CXX always wins, and a consumer pulling
# this in via add_subdirectory already has the variable set, so the block is a
# no-op there.  CMAKE_CXX_COMPILER must be set before project().
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
        AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    find_program(DIFF_PREFERRED_CXX NAMES g++-15 g++-14)
    if (DIFF_PREFERRED_CXX)
        set(CMAKE_CXX_COMPILER "${DIFF_PREFERRED_CXX}")
    endif ()
endif ()

project(ExpressionDifferentiator VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Which accessor spelling this build takes, and why -- see include/util/config.hpp,
# which owns the decision and is the only header that names P0847's macro.  The
# option is here so the choice is visible at configure time rather than only in
# a disassembly, and so a stale compiler selection is caught immediately.
#
#   auto - the compiler's own answer (__cpp_explicit_this_parameter)
#   on   - deducing this even where the macro is absent.  Clang implements P0847
#          from 18 without ever defining it, so `auto` is merely conservative
#          there and this is the switch that says so.
#   off  - the ref-qualified overload set even where the feature works, for a
#          build whose other tools cannot parse it yet.
#
# Two probes, because the macro and the feature are not the same question: the
# first asks what the compiler advertises (what `auto` will pick up), the second
# whether the feature actually compiles (what `on` requires).
include(CheckCXXSourceCompiles)
set(DIFF_DEDUCING_THIS "auto" CACHE STRING "Accessor spelling: auto|on|off")
set_property(CACHE DIFF_DEDUCING_THIS PROPERTY STRINGS auto on off)

set(CMAKE_REQUIRED_FLAGS "-std=c++23")
check_cxx_source_compiles(
        "struct S { int f(this S) { return 0; } };
         #ifndef __cpp_explicit_this_parameter
         #error no macro
         #endif
         int main() { return S{}.f(); }"
        DIFF_HAS_DEDUCING_THIS)
check_cxx_source_compiles(
        "struct S { int f(this S) { return 0; } };
         int main() { return S{}.f(); }"
        DIFF_DEDUCING_THIS_COMPILES)
unset(CMAKE_REQUIRED_FLAGS)

if (DIFF_DEDUCING_THIS STREQUAL "on" AND NOT DIFF_DEDUCING_THIS_COMPILES)
    message(FATAL_ERROR
            "DIFF_DEDUCING_THIS=on but ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} "
            "cannot compile an explicit object parameter.  Use DIFF_DEDUCING_THIS=auto or =off.")
elseif (DIFF_DEDUCING_THIS STREQUAL "on")
    set(DIFF_DEDUCING_THIS_VALUE 1)
elseif (DIFF_DEDUCING_THIS STREQUAL "off")
    set(DIFF_DEDUCING_THIS_VALUE 0)
elseif (DIFF_DEDUCING_THIS STREQUAL "auto")
    # Leave it undefined and let config.hpp decide, so a consumer compiling
    # against the installed headers without this CMake file gets the same
    # answer as one building through it.
    unset(DIFF_DEDUCING_THIS_VALUE)
else ()
    message(FATAL_ERROR "DIFF_DEDUCING_THIS must be auto, on or off (got '${DIFF_DEDUCING_THIS}').")
endif ()

if (DEFINED DIFF_DEDUCING_THIS_VALUE AND DIFF_DEDUCING_THIS_VALUE)
    message(STATUS "accessors: deducing this, forced on (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
elseif (DEFINED DIFF_DEDUCING_THIS_VALUE)
    message(STATUS "accessors: ref-qualified overload set, forced off (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
elseif (DIFF_HAS_DEDUCING_THIS)
    message(STATUS "accessors: deducing this (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
else ()
    message(STATUS "accessors: ref-qualified overload set (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
endif ()

# include/md/md.hpp needs an mdspan.  No toolchain we target ships one yet --
# libstdc++ 15.2 has no <mdspan> at all -- and three of the pieces the layouts
# and drivers use (submdspan, the padded layouts, the accessor machinery around
# them) are C++26 even where <mdspan> does exist.  So the reference
# implementation is vendored under include/md/third_party and this probe decides
# which one md.hpp binds to, the same way the deducing-this probe above reports
# which get<N>() spelling is live.
#
#   auto     - standard <mdspan> when it is complete enough, vendored otherwise
#   std      - require the standard one; hard error if incomplete
#   vendored - always the vendored one, so that path stays continuously built
set(DIFF_MDSPAN_MODE "auto" CACHE STRING "Which mdspan md.hpp binds to: auto|std|vendored")
set_property(CACHE DIFF_MDSPAN_MODE PROPERTY STRINGS auto std vendored)

set(CMAKE_REQUIRED_FLAGS "-std=c++23")
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
        DIFF_HAS_STD_MDSPAN)
unset(CMAKE_REQUIRED_FLAGS)

if (DIFF_MDSPAN_MODE STREQUAL "std" AND NOT DIFF_HAS_STD_MDSPAN)
    message(FATAL_ERROR
            "DIFF_MDSPAN_MODE=std but ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} "
            "has no complete <mdspan>/<submdspan>.  Use DIFF_MDSPAN_MODE=auto or =vendored.")
elseif (DIFF_MDSPAN_MODE STREQUAL "vendored" OR NOT DIFF_HAS_STD_MDSPAN)
    set(DIFF_MDSPAN_VENDORED ON)
    message(STATUS "mdspan: vendored reference implementation (include/md/third_party/mdspan.hpp)")
else ()
    set(DIFF_MDSPAN_VENDORED OFF)
    message(STATUS "mdspan: standard <mdspan> (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
endif ()

option(DIFF_BUILD_BENCHMARKS "Build Google Benchmark benchmarks" ON)
option(DIFF_BUILD_COMPARE "Build comparison benchmarks against autodiff" OFF)
option(ENABLE_NATIVE_ARCH "Build optimized for this machine" ON)
option(DIFF_USE_EIGEN "Build the Eigen interop boundary (include/interop/eigen_interop.hpp)" ON)

# Codegen flags for OUR targets only.
#
# These used to be add_compile_options(), i.e. global, which meant -march=native
# also landed on gtest, Google Benchmark and autodiff.  That is not just untidy:
# Google Benchmark builds itself with -Werror, and on a CI runner whose native
# arch resolves to something clang only half-supports it turns clang's
#   "invalid feature combination: +avx10.1-256; will be promoted to avx10.1-512"
# into a hard build failure in a dependency we do not control.  Our flags belong
# on our targets.
#
# Not put on the `diff` INTERFACE target either: diff is header-only, so it
# compiles nothing, and pushing -march=native onto a downstream consumer would
# decide their deployment target for them.
if (MSVC)
    set(DIFF_CODEGEN_FLAGS /arch:AVX2)
elseif (ENABLE_NATIVE_ARCH)
    set(DIFF_CODEGEN_FLAGS -march=native)
else ()
    set(DIFF_CODEGEN_FLAGS -march=x86-64-v3)
endif ()

# Floating-point contract.  Two flags, and neither of them trades accuracy for
# speed -- this library's results are exact answers, so nothing here is allowed
# to be lossy.  In particular -ffast-math and its parts (-ffinite-math-only,
# -fno-signed-zeros, -fassociative-math, -freciprocal-math) are deliberately
# NOT used, and neither is glibc's vector math: libmvec is 4 ULP against 0.5
# for the scalar calls.  Measured, -ffast-math is worth about -19% on the
# benchmark suite; it is still refused, and it would break the bitwise
# agreement the suite asserts between the seeded and leaf-storage engines.
#
#   -ffp-contract=fast   is already GCC's default under -std=gnu++.  Naming it
#       changes nothing today and is not a speed flag: it pins the one
#       remaining source of result variation.  Contraction placement decides
#       where a*b+c becomes a single FMA, and the same source compiles to three
#       different bit patterns under fast/on/off.  FMA is one rounding instead
#       of two, so `fast` is both the quickest and the most accurate of the
#       three -- but left unstated, a compiler default change would silently
#       move every derivative in the last bit.
#
#   -fno-math-errno      says nothing in here reads errno after a libm call,
#       which is true.  It lets sqrt() become the sqrtsd instruction (bitwise
#       identical -- IEEE requires sqrt to be correctly rounded) instead of a
#       library call.  Measured at -1.7% on the benchmark suite, and with
#       contraction pinned as above the results are byte-for-byte unchanged.
#
# Not used, on measurement rather than principle: -fno-plt was +0.8% (slower,
# despite PLT stubs showing at 3.2% of the profile) and LTO was -0.2%, inside
# noise and one more input to inlining, hence to contraction placement.
#
# MSVC gets nothing here, and wants nothing: /fp:precise is already its default
# and already forbids the reassociations refused above, and it does not contract
# unless asked (/fp:contract).  There is no /fp: spelling of -fno-math-errno.
# So the conservative behaviour is the default there and the flags below would
# have no equivalent to express -- MSVC results are its own, and the
# bit-exactness test in tests.cpp is a same-compiler A/B for exactly that
# reason.  Do NOT add /fp:fast: it is MSVC's -ffast-math.
option(DIFF_FP_FLAGS "Pin FP contraction and drop errno on libm calls" ON)
if (DIFF_FP_FLAGS AND NOT MSVC)
    list(APPEND DIFF_CODEGEN_FLAGS -ffp-contract=fast -fno-math-errno)
endif ()

# Eigen backs include/interop/eigen_interop.hpp — the boundary where a Hessian leaves
# this library for whatever factorizes it downstream, including the sparse form
# whose structure comes from the compile-time coupling pass and has no
# Eigen-side equivalent.  Nothing in the AD core includes Eigen: the symbolic
# path is constexpr and Eigen types cannot be constant-evaluated.
#
# The dependency is OPTIONAL but ON by default, and REQUIRED once enabled — a
# missing Eigen is then a clear configure-time error rather than a confusing
# failure deep in a header.  With DIFF_USE_EIGEN=OFF the library goes back to
# being header-only with no third-party dependencies, and eigen_interop.hpp is
# the only thing that disappears.
#
# Enabling it also fixes a latent bug on the compare path: DIFF_BUILD_COMPARE
# used to reach Eigen only because autodiff calls find_package(Eigen3) without
# ever linking Eigen3::Eigen to its target, so <Eigen/Core> resolved through a
# /usr/include/Eigen symlink rather than through any include path CMake set.
if (DIFF_USE_EIGEN)
    # No version in the request, then checked by hand.  Eigen's config file is
    # generated with SameMajorVersion compatibility, so asking for "3.4" is
    # asking for "some 3.x" -- and vcpkg now ships Eigen 5, which can never
    # satisfy that however compatible it actually is.  The interop boundary only
    # uses Eigen::Map, Matrix, Vector and SparseMatrix, none of which changed.
    find_package(Eigen3 REQUIRED NO_MODULE)
    if (DEFINED Eigen3_VERSION AND Eigen3_VERSION VERSION_LESS 3.4)
        message(FATAL_ERROR "Eigen 3.4 or newer is required (found ${Eigen3_VERSION})")
    endif ()
    message(STATUS "Eigen: ${Eigen3_VERSION}")
endif ()

include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

if (DIFF_BUILD_BENCHMARKS OR DIFF_BUILD_COMPARE)
    FetchContent_Declare(
            googlebenchmark
            URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif ()

if (DIFF_BUILD_COMPARE)
    set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(AUTODIFF_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
            autodiff_lib
            GIT_REPOSITORY https://github.com/autodiff/autodiff.git
            GIT_TAG v1.1.2
            GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(autodiff_lib)
endif ()

# Use CMAKE_CURRENT_SOURCE_DIR (not CMAKE_SOURCE_DIR) so this file works
# both standalone and as a git submodule via add_subdirectory().
add_library(diff INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/expression_differentiator.hpp
)
# Namespaced alias so consumers can link a stable target name.
add_library(ExpressionDifferentiator::diff ALIAS diff)
target_include_directories(diff INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Each header directory lists its own files: include/CMakeLists.txt walks the
# subdirectories and each one calls target_sources(diff INTERFACE ...) on the
# target created just above.  Nothing is compiled from there -- diff is an
# INTERFACE library, so the lists exist to make the headers visible in IDEs and
# to keep a directory's contents described next to the contents themselves.
add_subdirectory(include)
if (DIFF_USE_EIGEN)
    target_link_libraries(diff INTERFACE Eigen3::Eigen)
    target_compile_definitions(diff INTERFACE DIFF_USE_EIGEN)
endif ()
if (DIFF_MDSPAN_VENDORED)
    target_compile_definitions(diff INTERFACE DIFF_MDSPAN_FORCE_VENDORED)
endif ()
if (DEFINED DIFF_DEDUCING_THIS_VALUE)
    target_compile_definitions(diff INTERFACE DIFF_DEDUCING_THIS=${DIFF_DEDUCING_THIS_VALUE})
endif ()

# Only build the standalone executables when this is the top-level project.
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_executable(main src/main.cpp)
    target_link_libraries(main PUBLIC diff)
    target_compile_options(main PRIVATE ${DIFF_CODEGEN_FLAGS})

    add_executable(tests tests.cpp)
    target_link_libraries(tests gtest_main diff)
    target_compile_options(tests PRIVATE ${DIFF_CODEGEN_FLAGS})

    enable_testing()
    gtest_discover_tests(tests)

    if (DIFF_BUILD_COMPARE)
        add_executable(benchmarks_compare benchmarks/benchmark_compare.cpp)
        target_link_libraries(benchmarks_compare PRIVATE benchmark::benchmark diff autodiff::autodiff)
        target_compile_options(benchmarks_compare PRIVATE ${DIFF_CODEGEN_FLAGS})
        if (NOT MSVC)
            target_compile_options(benchmarks_compare PRIVATE -Wall -Wextra)
        endif ()
    endif ()

    if (DIFF_BUILD_BENCHMARKS)
        add_executable(benchmarks benchmarks/benchmark.cpp)
        target_link_libraries(benchmarks PRIVATE benchmark::benchmark diff)
        target_compile_options(benchmarks PRIVATE ${DIFF_CODEGEN_FLAGS})
        set(BENCHMARK_RESULTS_DIR "${CMAKE_BINARY_DIR}/benchmark-results")
        add_custom_target(benchmark_json
                COMMAND ${CMAKE_COMMAND} -E make_directory "${BENCHMARK_RESULTS_DIR}"
                COMMAND $<TARGET_FILE:benchmarks>
                --benchmark_min_time=0.05s
                --benchmark_out="${BENCHMARK_RESULTS_DIR}/benchmarks.json"
                --benchmark_out_format=json
                DEPENDS benchmarks
                USES_TERMINAL
                COMMENT "Running benchmarks and exporting JSON results")
    endif ()

    if (MSVC)
        # /Wall enables off-by-default noise warnings. Expression/Equation types
        # hold non-assignable members (const/reference data) by design, so the
        # implicit special members are deleted. Silence that family:
        #   C4625 copy ctor implicitly deleted
        #   C4626 copy assignment implicitly deleted
        #   C5026 move ctor implicitly deleted
        #   C5027 move assignment implicitly deleted
        #   C4623/C4625/C4626/C5026/C5027  implicit default/copy/move member
        #       deleted -- by design: Expression and Equation hold const members
        #       and have no default state (an expression node without operands
        #       is not a thing).  C4623 alone accounts for a few hundred lines
        #       of a /Wall log, one per instantiated node type.
        #   C4710/C4711  "function not inlined" / "selected for automatic inline
        #       expansion".  These are inlining *diagnostics*, not defects, and a
        #       header-only expression-template library emits thousands of them:
        #       every operator, every sweep, every std::array copy.  They say
        #       nothing a reader can act on.
        #   C4868  "may not enforce left-to-right evaluation order in a brace
        #       enclosed initializer list".  The standard does guarantee that
        #       order; this warns that MSVC may not.  Nothing here depends on
        #       evaluation order inside a braced list -- the initializers are
        #       independent element computations.
        #   C4820  trailing padding inserted after a member.
        #   C5045  Spectre mitigation would be inserted for a load.
        #   C5246  "the initialization of a subobject should be wrapped in
        #       braces".  Fires on every `std::array{a, b}` in the library.
        #       Brace elision in aggregate initialization is standard and
        #       intended; MSVC is asking for the inner braces back.
        #   C4514  unreferenced inline function removed -- the expected fate of
        #       most of a header-only library in any one TU.
        #   C4324  structure padded due to an alignment specifier.  That is the
        #       whole point of VectorDual's alignas(32) on its partial pack.
        #   C5266  'const' on a return type has no effect -- from the vendored
        #       mdspan in include/md/third_party, not our code to change.
        #
        # Deliberately NOT suppressed: C4365 (signed/unsigned mismatch) and
        # C5219 (implicit integer->float conversion).  Both fire only a handful
        # of times, and both are the kind of warning that can catch a real bug.
        set(DIFF_MSVC_WALL_NOISE /Wall /wd4623 /wd4625 /wd4626 /wd5026 /wd5027
                                 /wd4710 /wd4711 /wd4868 /wd4820 /wd5045
                                 /wd5246 /wd4514 /wd4324 /wd5266)
        target_compile_options(main PRIVATE ${DIFF_MSVC_WALL_NOISE})
        target_compile_options(tests PRIVATE ${DIFF_MSVC_WALL_NOISE})
        if (DIFF_BUILD_BENCHMARKS)
            target_compile_options(benchmarks PRIVATE ${DIFF_MSVC_WALL_NOISE})
        endif ()
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

        foreach (target gtest gtest_main gmock gmock_main benchmark benchmark_main)
            if (TARGET ${target})
                target_compile_options(${target} PRIVATE /W0 /wd4668)
                set_target_properties(${target} PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
            endif ()
        endforeach ()
    else ()
        target_compile_options(main PRIVATE -Wall -Wpedantic -Wfatal-errors -Wextra)
        target_compile_options(tests PRIVATE -Wall -Wpedantic -Wfatal-errors -Wextra)
        if (DIFF_BUILD_BENCHMARKS)
            target_compile_options(benchmarks PRIVATE -Wall -Wpedantic -Wfatal-errors -Wextra)
        endif ()
    endif ()
endif ()
