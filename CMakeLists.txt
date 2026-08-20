cmake_minimum_required(VERSION 3.20)

# Newest GCC on PATH, unless the caller chose a compiler.  Must precede project().
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
        AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    find_program(DDX_PREFERRED_CXX NAMES g++-15 g++-14)
    if (DDX_PREFERRED_CXX)
        set(CMAKE_CXX_COMPILER "${DDX_PREFERRED_CXX}")
    endif ()
endif ()

project(ddx VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Accessor spelling (include/util/config.hpp owns the decision).
#   auto - the compiler's own answer (__cpp_explicit_this_parameter)
#   on   - deducing this even where the macro is absent
#   off  - the ref-qualified overload set
# The first probe asks what `auto` picks up, the second what `on` requires.
include(CheckCXXSourceCompiles)
set(DDX_DEDUCING_THIS "auto" CACHE STRING "Accessor spelling: auto|on|off")
set_property(CACHE DDX_DEDUCING_THIS PROPERTY STRINGS auto on off)

set(CMAKE_REQUIRED_FLAGS "-std=c++23")
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
unset(CMAKE_REQUIRED_FLAGS)

if (DDX_DEDUCING_THIS STREQUAL "on" AND NOT DDX_DEDUCING_THIS_COMPILES)
    message(FATAL_ERROR
            "DDX_DEDUCING_THIS=on but ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} "
            "cannot compile an explicit object parameter.  Use DDX_DEDUCING_THIS=auto or =off.")
elseif (DDX_DEDUCING_THIS STREQUAL "on")
    set(DDX_DEDUCING_THIS_VALUE 1)
elseif (DDX_DEDUCING_THIS STREQUAL "off")
    set(DDX_DEDUCING_THIS_VALUE 0)
elseif (DDX_DEDUCING_THIS STREQUAL "auto")
    # Undefined: config.hpp decides, so a consumer building without this file agrees.
    unset(DDX_DEDUCING_THIS_VALUE)
else ()
    message(FATAL_ERROR "DDX_DEDUCING_THIS must be auto, on or off (got '${DDX_DEDUCING_THIS}').")
endif ()

if (DEFINED DDX_DEDUCING_THIS_VALUE AND DDX_DEDUCING_THIS_VALUE)
    message(STATUS "accessors: deducing this, forced on (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
elseif (DEFINED DDX_DEDUCING_THIS_VALUE)
    message(STATUS "accessors: ref-qualified overload set, forced off (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
elseif (DDX_HAS_DEDUCING_THIS)
    message(STATUS "accessors: deducing this (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
else ()
    message(STATUS "accessors: ref-qualified overload set (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
endif ()

# Which mdspan include/md/md.hpp binds to.
#   auto     - standard <mdspan> when it is complete enough, vendored otherwise
#   std      - require the standard one; hard error if incomplete
#   vendored - always the vendored one under include/md/third_party
set(DDX_MDSPAN_MODE "auto" CACHE STRING "Which mdspan md.hpp binds to: auto|std|vendored")
set_property(CACHE DDX_MDSPAN_MODE PROPERTY STRINGS auto std vendored)

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
        DDX_HAS_STD_MDSPAN)
unset(CMAKE_REQUIRED_FLAGS)

if (DDX_MDSPAN_MODE STREQUAL "std" AND NOT DDX_HAS_STD_MDSPAN)
    message(FATAL_ERROR
            "DDX_MDSPAN_MODE=std but ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} "
            "has no complete <mdspan>/<submdspan>.  Use DDX_MDSPAN_MODE=auto or =vendored.")
elseif (DDX_MDSPAN_MODE STREQUAL "vendored" OR NOT DDX_HAS_STD_MDSPAN)
    set(DDX_MDSPAN_VENDORED ON)
    message(STATUS "mdspan: vendored reference implementation (include/md/third_party/mdspan.hpp)")
else ()
    set(DDX_MDSPAN_VENDORED OFF)
    message(STATUS "mdspan: standard <mdspan> (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
endif ()

option(DDX_BUILD_BENCHMARKS "Build Google Benchmark benchmarks" ON)
option(DDX_BUILD_COMPARE "Build comparison benchmarks against autodiff" OFF)
option(ENABLE_NATIVE_ARCH "Build optimized for this machine" ON)

# Codegen flags for our own targets only -- never global, never on `ddx`.
if (MSVC)
    # /bigobj: one TU instantiates past the 2^16 COFF section limit.
    set(DDX_CODEGEN_FLAGS /arch:AVX2 /bigobj)
elseif (ENABLE_NATIVE_ARCH)
    set(DDX_CODEGEN_FLAGS -march=native)
else ()
    set(DDX_CODEGEN_FLAGS -march=x86-64-v3)
endif ()

# Floating-point contract: pin contraction, drop errno.  Neither flag is lossy.
# Nothing that trades accuracy for speed belongs here -- in particular never
# -ffast-math (nor /fp:fast on MSVC, which needs no flags at all).
option(DDX_FP_FLAGS "Pin FP contraction and drop errno on libm calls" ON)
if (DDX_FP_FLAGS AND NOT MSVC)
    list(APPEND DDX_CODEGEN_FLAGS -ffp-contract=fast -fno-math-errno)
endif ()

# Eigen is not a dependency of the library -- only of the autodiff comparison.
# Requested without a version (Eigen's config is SameMajorVersion, so "3.4"
# would exclude Eigen 5) and checked by hand below.
if (DDX_BUILD_COMPARE)
    find_package(Eigen3 REQUIRED NO_MODULE)
    if (DEFINED Eigen3_VERSION AND Eigen3_VERSION VERSION_LESS 3.4)
        message(FATAL_ERROR "Eigen 3.4 or newer is required (found ${Eigen3_VERSION})")
    endif ()
    message(STATUS "Eigen (for the autodiff comparison only): ${Eigen3_VERSION}")
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

if (DDX_BUILD_BENCHMARKS OR DDX_BUILD_COMPARE)
    FetchContent_Declare(
            googlebenchmark
            URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif ()

if (DDX_BUILD_COMPARE)
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

# CMAKE_CURRENT_SOURCE_DIR, so this works standalone and via add_subdirectory().
add_library(ddx INTERFACE)
# Namespaced alias so consumers can link a stable target name.
add_library(ddx::ddx ALIAS ddx)
target_include_directories(ddx INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Each header directory lists its own files onto the INTERFACE target above,
# for IDE visibility; nothing is compiled from there.
add_subdirectory(include)
if (DDX_MDSPAN_VENDORED)
    target_compile_definitions(ddx INTERFACE DDX_MDSPAN_FORCE_VENDORED)
endif ()
if (DEFINED DDX_DEDUCING_THIS_VALUE)
    target_compile_definitions(ddx INTERFACE DDX_DEDUCING_THIS=${DDX_DEDUCING_THIS_VALUE})
endif ()

# Only build the standalone executables when this is the top-level project.
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_executable(main src/main.cpp)
    target_link_libraries(main PUBLIC ddx)
    target_compile_options(main PRIVATE ${DDX_CODEGEN_FLAGS})

    # One TU per anonymous-namespace helper block, so `-j` can build them at once.
    file(GLOB DDX_TEST_SOURCES CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
    add_executable(tests ${DDX_TEST_SOURCES})
    target_include_directories(tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_link_libraries(tests gtest_main ddx)
    target_compile_options(tests PRIVATE ${DDX_CODEGEN_FLAGS})

    enable_testing()
    gtest_discover_tests(tests)

    if (DDX_BUILD_COMPARE)
        add_executable(benchmarks_compare benchmarks/benchmark_compare.cpp)
        target_link_libraries(benchmarks_compare PRIVATE benchmark::benchmark ddx autodiff::autodiff Eigen3::Eigen)
        target_compile_options(benchmarks_compare PRIVATE ${DDX_CODEGEN_FLAGS})
        if (NOT MSVC)
            target_compile_options(benchmarks_compare PRIVATE -Wall -Wextra)
        endif ()
    endif ()

    if (DDX_BUILD_BENCHMARKS)
        add_executable(benchmarks benchmarks/benchmark.cpp)
        target_link_libraries(benchmarks PRIVATE benchmark::benchmark ddx)
        target_compile_options(benchmarks PRIVATE ${DDX_CODEGEN_FLAGS})
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
        # /Wall noise a header-only expression-template library emits by the
        # thousand.  C4365 and C5219 are deliberately left on.
        set(DDX_MSVC_WALL_NOISE /Wall /wd4623 /wd4625 /wd4626 /wd5026 /wd5027
                                 /wd4710 /wd4711 /wd4868 /wd4820 /wd5045
                                 /wd5246 /wd4514 /wd4324 /wd5266)
        target_compile_options(main PRIVATE ${DDX_MSVC_WALL_NOISE})
        target_compile_options(tests PRIVATE ${DDX_MSVC_WALL_NOISE})
        if (DDX_BUILD_BENCHMARKS)
            target_compile_options(benchmarks PRIVATE ${DDX_MSVC_WALL_NOISE})
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
        if (DDX_BUILD_BENCHMARKS)
            target_compile_options(benchmarks PRIVATE -Wall -Wpedantic -Wfatal-errors -Wextra)
        endif ()
    endif ()
endif ()
