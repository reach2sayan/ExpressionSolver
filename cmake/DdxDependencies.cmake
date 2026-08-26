# Every third-party dependency ddx uses, in one place: what it is, where it comes
# from, and what asks for it.  The two the build tools need are fetched, and
# declaring those is cheap -- FetchContent downloads nothing until the matching
# ddx_use_*() macro asks -- so the whole manifest can live here while a build
# still fetches only what its options turned on.  The two that reach ddx's own
# headers or its binaries are found on the machine instead, because an installed
# ddx has to be able to name them again on someone else's.
#
#   ddx_use_boost()            Boost.Mp11 + Graph + DynamicBitset  found, always
#   ddx_use_llvm()             LLVM 20, found                      DDX_BUILD_JIT
#   ddx_use_googletest()       GoogleTest, fetched                 top-level only
#   ddx_use_googlebenchmark()  Google Benchmark, fetched           DDX_BUILD_BENCHMARKS
#   ddx_use_pybind11()         pybind11, found in the build env    DDX_BUILD_PYTHON
#
# Each is idempotent and safe to call from wherever the dependency is first
# needed.  Macros rather than functions: FetchContent_MakeAvailable() defines
# variables the caller's scope is expected to see.
include_guard(GLOBAL)

include(CheckCXXSourceCompiles)
include(FetchContent)

# --- versions ---------------------------------------------------------------
# The fetched two only; Boost and LLVM are whatever the machine has.
set(DDX_GOOGLETEST_REF "5376968f6948923e2411081fd9372e71a59d8e77"
        CACHE STRING "GoogleTest commit to fetch")
set(DDX_GOOGLEBENCHMARK_VERSION "1.9.1" CACHE STRING "Google Benchmark release to fetch")

# The LLVM range ddx::jit has been built against.  A range rather than a floor
# because the ORC C++ API is not stable across releases; both ends currently
# name 20, so the range is one release wide.
#
# 20 is a hard floor, not caution: src/jit uses three spellings that do not
# exist below it.  Intrinsic::getOrInsertDeclaration and
# Intrinsic::lookupIntrinsicID are absent from LLVM 18's Intrinsics.h entirely
# -- 18 has getDeclaration and offers the lookup only as
# Function::lookupIntrinsicID, which 20 removed, so no one spelling compiles
# against both.  And sys::getHostCPUFeatures() returns bool through an out
# parameter in 18 where 20 returns the StringMap by value, which compiler.cpp
# relies on.  Above 20 the ceiling is real too: 21 replaces the `nocapture`
# attribute codegen.cpp sets with `captures(none)`.
set(DDX_LLVM_VERSION_MIN 20)
set(DDX_LLVM_VERSION_MAX 20)

# --- declarations -----------------------------------------------------------
# SYSTEM throughout: a fetched dependency's warnings are not ours to fix.
# URL_HASH so a cached archive is verified rather than re-fetched: without one
# CMake reports "File already exists but no hash specified" and downloads 150 MB
# again every time a stamp goes missing.
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
# Mp11 is the type-list vocabulary the symbol lists are built out of, Boost.Graph
# supplies the colouring the runtime graph needs, and Boost.DynamicBitset the
# coupling rows.  Describe, Endian and CRC are the saved graph's: the field lists
# the one serialisation traversal walks, fixed-width little-endian integers, and
# the checksum a loader clears before it trusts a byte.  All header-only, so
# nothing links a compiled Boost library and what a build needs of it is an
# include path.
#
# Boost.Serialization is deliberately not among them, and could not be: it is a
# compiled library, and its only error channel is throwing, which this tree turns
# into abort() (src/rt/boost_no_exceptions.cpp).
#
# Found, not fetched, and the same find ddx-config.cmake makes: ddx's public
# headers include boost's, so whatever a build compiles against is what its
# consumers compile against too.  A fetched Boost could be neither -- it belongs
# to no export set, so no installed ddx could name it.
#
# Config mode.  BoostConfig.cmake has shipped with Boost since 1.70 and is the
# only mode CMake 4 has left, FindBoost being removed there and deprecated since
# 3.30, which is what makes an unqualified find_package(Boost) warn under
# CMP0167.
#
# The cache variable is renamed whenever this list grows: check_cxx_source_compiles
# trusts a cached answer, so an existing build tree would otherwise never re-probe
# for a header that was added to it.
#
# No version floor: all six have been stable for years, so a release
# number would turn working Boosts away to answer a question the headers
# themselves answer.  What does go wrong is a partial install -- vcpkg and the
# distros that split Boost up can have Mp11 and not Graph -- and that is a
# missing header, which is what this asks about.
macro(ddx_use_boost)
    if (NOT TARGET Boost::headers)
        find_package(Boost REQUIRED CONFIG)
        # The target rather than its INTERFACE_INCLUDE_DIRECTORIES scraped out of
        # it: the probe then compiles against whatever Boost::headers carries,
        # generator expressions and several directories included, which is
        # exactly what the rest of the build compiles against.
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
# Sets LLVM_DEFINITIONS_LIST and DDX_LLVM_LIBS for src/jit alongside what
# LLVMConfig itself defines.
macro(ddx_use_llvm)
    if (NOT LLVM_FOUND)
        # LLVMConfig declares itself compatible only with an exact version
        # request, so a find_package version range never matches; check the
        # major ourselves.
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
# Found, not fetched, for the same reason Boost and LLVM are: it reaches a binary
# this project produces.  What makes it unlike those two is *where* it is found --
# the build environment rather than the machine.  An extension module is built
# against one interpreter's headers and ABI, and the thing that knows which one
# is the build frontend: scikit-build-core puts the pybind11 wheel's cmake
# directory on CMAKE_PREFIX_PATH, so `uv pip install .` resolves the interpreter
# and the bindings together.  Naming a path here would only fight that.
#
# 3.0 is a floor rather than caution: native_enum.h arrived after 2.13, and the
# bindings use it so Backend and VecLib are enum.IntEnum on the Python side
# instead of pybind's own enum objects.
macro(ddx_use_pybind11)
    if (NOT TARGET pybind11::module)
        # The interpreter first, so that pybind11 is the one *it* has.  A wheel
        # build already puts that pair on CMAKE_PREFIX_PATH and this finds it
        # there; a preset build has only a virtualenv, so the interpreter is
        # asked where its own cmake directory is.
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

# SYSTEM on the FetchContent_Declare covers a consumer's include of the headers;
# this is the other half, the dependency compiling itself.  MSVC only -- the
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

