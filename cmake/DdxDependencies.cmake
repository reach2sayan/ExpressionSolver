# Every third-party dependency, in one place.  Anything reaching ddx's headers or
# binaries is *found*, so an installed ddx can name it again elsewhere; the build
# tools are fetched, and declaring one downloads nothing until its macro asks.
#
#   ddx_use_boost()            Boost, header-only, found           always
#   ddx_use_llvm()             LLVM 20, found                      DDX_BUILD_JIT
#   ddx_use_googletest()       GoogleTest, fetched                 top-level only
#   ddx_use_googlebenchmark()  Google Benchmark, fetched           DDX_BUILD_BENCHMARKS
#   ddx_use_pybind11()         pybind11, found in the build env    DDX_BUILD_PYTHON
#
# Each is idempotent.  Macros, not functions: FetchContent_MakeAvailable()
# defines variables the caller's scope is expected to see.
include_guard(GLOBAL)

include(CheckCXXSourceCompiles)
include(FetchContent)

# --- versions ---------------------------------------------------------------
# The fetched two only; Boost and LLVM are whatever the machine has.
set(DDX_GOOGLETEST_REF "5376968f6948923e2411081fd9372e71a59d8e77"
        CACHE STRING "GoogleTest commit to fetch")
set(DDX_GOOGLEBENCHMARK_VERSION "1.9.1" CACHE STRING "Google Benchmark release to fetch")

# A range, not a floor: the ORC C++ API is not stable across releases, and both
# ends are hard.  Below 20, Intrinsic::getOrInsertDeclaration and
# lookupIntrinsicID do not exist and getHostCPUFeatures() has another signature;
# above it, 21 replaces the `nocapture` attribute codegen.cpp sets.
set(DDX_LLVM_VERSION_MIN 20)
set(DDX_LLVM_VERSION_MAX 20)

# --- declarations -----------------------------------------------------------
# SYSTEM throughout: a fetched dependency's warnings are not ours to fix.
FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/${DDX_GOOGLETEST_REF}.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM
)
FetchContent_Declare(googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v${DDX_GOOGLEBENCHMARK_VERSION}.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM
)

# --- Boost -------------------------------------------------------------------
# Mp11 for the symbol lists, Graph for the colouring, DynamicBitset for the
# coupling rows; Describe, Endian and CRC for the saved graph.  All header-only.
# Boost.Serialization is deliberately absent -- see boost_no_exceptions.cpp.
#
# Found, not fetched, and the same find ddx-config.cmake makes: ddx's public
# headers include boost's.  A fetched Boost belongs to no export set, so no
# installed ddx could name it.  Config mode is the only one CMake 4 has left.
#
# No version floor -- a partial install is what goes wrong, and that is a missing
# header, which is what the probe asks about.  Rename the cache variable whenever
# this list grows: check_cxx_source_compiles trusts a cached answer.
macro(ddx_use_boost)
    if (NOT TARGET Boost::headers)
        find_package(Boost REQUIRED CONFIG)
        # The target, not its scraped include dirs, so the probe compiles
        # against exactly what the rest of the build does.
        set(CMAKE_REQUIRED_LIBRARIES Boost::headers)
        check_cxx_source_compiles("
                #include <boost/mp11/algorithm.hpp>
                #include <boost/mp11/list.hpp>
                #include <boost/graph/compressed_sparse_row_graph.hpp>
                #include <boost/dynamic_bitset.hpp>
                #include <boost/describe.hpp>
                #include <boost/endian/conversion.hpp>
                #include <boost/crc.hpp>
                int main() {}"
                DDX_BOOST_HEADERS_PRESENT)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if (NOT DDX_BOOST_HEADERS_PRESENT)
            message(FATAL_ERROR
                    "Boost ${Boost_VERSION} was found at ${Boost_INCLUDE_DIRS}, but it does not "
                    "have the headers ddx names: Mp11 (algorithm, list), Graph "
                    "(compressed_sparse_row_graph), DynamicBitset, Describe, Endian and CRC.  "
                    "A modular install wants "
                    "those libraries added; a distro one usually wants the whole headers package "
                    "(libboost-dev).")
        endif ()
    endif ()
endmacro()

# --- LLVM -------------------------------------------------------------------
# Sets LLVM_DEFINITIONS_LIST and DDX_LLVM_LIBS for src/jit.
macro(ddx_use_llvm)
    if (NOT LLVM_FOUND)
        # LLVMConfig matches only an exact version request, so a find_package
        # range never does; check the major here.
        find_package(LLVM REQUIRED CONFIG)
        if (LLVM_VERSION_MAJOR LESS DDX_LLVM_VERSION_MIN
                OR LLVM_VERSION_MAJOR GREATER DDX_LLVM_VERSION_MAX)
            message(FATAL_ERROR
                    "ddx::jit has been built against LLVM ${DDX_LLVM_VERSION_MIN}-${DDX_LLVM_VERSION_MAX}; "
                    "found ${LLVM_PACKAGE_VERSION}.  Point CMake at a supported one with "
                    "-DLLVM_DIR=/usr/lib/llvm-${DDX_LLVM_VERSION_MAX}/lib/cmake/llvm.")
        endif ()
        message(STATUS "LLVM ${LLVM_PACKAGE_VERSION} (${LLVM_INCLUDE_DIRS})")
        separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
        llvm_map_components_to_libnames(DDX_LLVM_LIBS
                core orcjit native passes support analysis target)
    endif ()
endmacro()

# --- pybind11 ---------------------------------------------------------------
# Found in the build *environment*, not on the machine: an extension is built
# against one interpreter's headers and ABI, and scikit-build-core puts the
# matching wheel's cmake directory on CMAKE_PREFIX_PATH.  Naming a path here
# would fight that.  3.0 is a hard floor: native_enum.h arrived after 2.13, and
# the bindings use it to make Backend and VecLib enum.IntEnum.
macro(ddx_use_pybind11)
    if (NOT TARGET pybind11::module)
        # The interpreter first, so pybind11 is the one *it* has: a preset
        # build has only a virtualenv, so ask it for its own cmake directory.
        find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
        if (NOT DEFINED pybind11_DIR)
            execute_process(
                    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                    OUTPUT_VARIABLE pybind11_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
            if (NOT pybind11_DIR)
                message(FATAL_ERROR
                        "${Python_EXECUTABLE} has no pybind11.  The `python` preset builds "
                        "against .venv, so install it there: `uv sync`.  Left to search the "
                        "machine, CMake finds whatever the distro packaged -- which is how a "
                        "2.x shows up against a 3.0 request.")
            endif ()
        endif ()
        find_package(pybind11 3.0 CONFIG REQUIRED)
    endif ()
endmacro()

# --- GoogleTest, Google Benchmark -------------------------------------------
macro(ddx_use_googletest)
    if (NOT TARGET gtest_main)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        _ddx_silence_dependency(gtest gtest_main gmock gmock_main)
    endif ()
endmacro()

macro(ddx_use_googlebenchmark)
    if (NOT TARGET benchmark::benchmark)
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googlebenchmark)
        _ddx_silence_dependency(benchmark benchmark_main)
    endif ()
endmacro()

# The other half of SYSTEM: the dependency compiling itself.  MSVC only -- the
# fetched sources build clean under the GCC and Clang warning set.
function(_ddx_silence_dependency)
    if (NOT MSVC)
        return ()
    endif ()
    foreach (target IN LISTS ARGV)
        if (TARGET ${target})
            target_compile_options(${target} PRIVATE /W0 /wd4668)
            set_target_properties(${target} PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
        endif ()
    endforeach ()
endfunction()

