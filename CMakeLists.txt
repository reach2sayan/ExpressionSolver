cmake_minimum_required(VERSION 3.20)

project(ExpressionDifferentiator VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(DIFF_BUILD_BENCHMARKS "Build Google Benchmark benchmarks" ON)
option(DIFF_BUILD_COMPARE "Build comparison benchmarks against autodiff" OFF)
option(ENABLE_NATIVE_ARCH "Build optimized for this machine" ON)

if (MSVC)
   add_compile_options(/arch:AVX2)
else ()
    if (ENABLE_NATIVE_ARCH)
        add_compile_options(-march=native)
    else ()
        add_compile_options(-march=x86-64-v3)
    endif ()
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
        ${CMAKE_CURRENT_SOURCE_DIR}/include/expressions.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/operations.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/mpl.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/traits.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/values.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/equation.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dual.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/forward_driver.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gradient.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/expression_differentiator.hpp
)
# Namespaced alias so consumers can link a stable target name.
add_library(ExpressionDifferentiator::diff ALIAS diff)
target_include_directories(diff INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Only build the standalone executables when this is the top-level project.
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_executable(main src/main.cpp)
    target_link_libraries(main PUBLIC diff)

    add_executable(tests tests.cpp)
    target_link_libraries(tests gtest_main diff)

    enable_testing()
    gtest_discover_tests(tests)

    if (DIFF_BUILD_COMPARE)
        add_executable(benchmarks_compare benchmarks/benchmark_compare.cpp)
        target_link_libraries(benchmarks_compare PRIVATE benchmark::benchmark diff autodiff::autodiff)
        if (NOT MSVC)
            target_compile_options(benchmarks_compare PRIVATE -Wall -Wextra)
        endif ()
    endif ()

    if (DIFF_BUILD_BENCHMARKS)
        add_executable(benchmarks benchmarks/benchmark.cpp)
        target_link_libraries(benchmarks PRIVATE benchmark::benchmark diff)
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
        set(DIFF_MSVC_WALL_NOISE /Wall /wd4625 /wd4626 /wd5026 /wd5027)
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
