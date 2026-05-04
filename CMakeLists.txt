cmake_minimum_required(VERSION 3.20)
if (POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif ()

project(ExpressionDifferentiator VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(DIFF_BUILD_BENCHMARKS "Build Google Benchmark benchmarks" ON)
option(DIFF_BUILD_COMPARE "Build comparison benchmarks against autodiff" OFF)
option(ENABLE_NATIVE_ARCH "Build optimized for this machine" OFF)

if (MSVC)
   add_compile_options(/arch:AVX2)
else ()
    if (ENABLE_NATIVE_ARCH)
        add_compile_options(-march=native)
    else ()
        add_compile_options(-march=x86-64-v3)
    endif ()
endif ()
find_package(Boost 1.74 CONFIG QUIET)

include(FetchContent)
if (NOT TARGET Boost::headers)
    FetchContent_Declare(
            boost_mp11
            GIT_REPOSITORY https://github.com/boostorg/mp11.git
            GIT_TAG c3eba6ac6be2e21af33a1b2ec97634cd01bcc447
    )
    FetchContent_MakeAvailable(boost_mp11)

    add_library(boost_headers INTERFACE)
    add_library(Boost::headers ALIAS boost_headers)
    target_include_directories(boost_headers INTERFACE
            ${boost_mp11_SOURCE_DIR}/include)
endif ()

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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
    # autodiff is header-only; disable its sub-targets to avoid extra build time.
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

add_library(diff INTERFACE
        ${CMAKE_SOURCE_DIR}/include/expressions.hpp
        ${CMAKE_SOURCE_DIR}/include/operations.hpp
        ${CMAKE_SOURCE_DIR}/include/traits.hpp
        ${CMAKE_SOURCE_DIR}/include/values.hpp
        ${CMAKE_SOURCE_DIR}/include/equation.hpp
        ${CMAKE_SOURCE_DIR}/include/dual.hpp
        ${CMAKE_SOURCE_DIR}/include/gradient.hpp
        expression_differentiator.hpp
)
target_include_directories(diff INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(diff INTERFACE Boost::headers)
target_include_directories(diff SYSTEM INTERFACE
        $<TARGET_PROPERTY:Boost::headers,INTERFACE_INCLUDE_DIRECTORIES>)

add_executable(main src/main.cpp)
target_link_libraries(main PUBLIC diff)

add_executable(tests tests.cpp
        expression_differentiator.hpp)
target_link_libraries(tests gtest_main diff)

add_test(NAME tests COMMAND tests)
enable_testing()

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
    target_compile_options(main PRIVATE /Wall)
    target_compile_options(tests PRIVATE /Wall)
    if (DIFF_BUILD_BENCHMARKS)
        target_compile_options(benchmarks PRIVATE /Wall)
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
