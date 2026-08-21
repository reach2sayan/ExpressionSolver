cmake_minimum_required(VERSION 3.21)

# Newest GCC on PATH, unless the caller chose a compiler.  Must precede project().
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
        AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    find_program(DDX_PREFERRED_CXX NAMES g++-15 g++-14)
    if (DDX_PREFERRED_CXX)
        set(CMAKE_CXX_COMPILER "${DDX_PREFERRED_CXX}")
    endif ()
endif ()

project(ddx VERSION 1.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# This file is the assembly order; the modules under cmake/ are the parts.
#   DdxToolchainProbes  what the compiler can do   -> DDX_MDSPAN_VENDORED, ...
#   DdxCompileOptions   our own targets' flags     -> ddx_target_flags(), ...
#   DdxDependencies     every third-party pin      -> ddx_use_boost(), ...
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(DdxToolchainProbes)
include(DdxCompileOptions)
include(DdxDependencies)

# --- the library ------------------------------------------------------------
# Header-only, and header-only in what it depends on: Boost.Mp11 below is the
# whole of it.  CMAKE_CURRENT_SOURCE_DIR, so this works standalone and via
# add_subdirectory().
add_library(ddx INTERFACE)
add_library(ddx::ddx ALIAS ddx)
target_include_directories(ddx INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Each header directory lists its own files onto the target, for IDE visibility;
# nothing is compiled from there.
add_subdirectory(include)

if (DDX_MDSPAN_VENDORED)
    target_compile_definitions(ddx INTERFACE DDX_MDSPAN_FORCE_VENDORED)
endif ()
if (DEFINED DDX_DEDUCING_THIS_VALUE)
    target_compile_definitions(ddx INTERFACE DDX_DEDUCING_THIS=${DDX_DEDUCING_THIS_VALUE})
endif ()

# --- optional runtime graph -------------------------------------------------
# Everything above is headers, and stays that way.  The runtime graph is what an
# expression built at run time lives in, and it is opt-in because it pulls code
# that is compiled and platform-dependent rather than one more header.
option(DDX_BUILD_RT "Build the runtime expression graph (Boost.Graph)" OFF)
option(DDX_BUILD_JIT "Build the LLVM JIT backend (implies DDX_BUILD_RT)" OFF)
if (DDX_BUILD_JIT AND NOT DDX_BUILD_RT)
    set(DDX_BUILD_RT ON)
endif ()

# ddx::jit is the project's only compiled library, and shared is what it wants
# to be: LLVM is the bulk of it, and a static ddx::jit puts that bulk into every
# consumer that links it.  Static remains available for a consumer that would
# rather have one artefact than two.
option(DDX_JIT_STATIC "Build ddx::jit as a static archive rather than a shared library" OFF)

if (DDX_BUILD_RT)
    ddx_use_boost()
    add_subdirectory(src/rt)
    add_subdirectory(include/rt)
endif ()

# After the runtime graph rather than before it: Boost.Graph depends on Mp11, so
# where both are wanted the superproject has already defined Boost::mp11.
# BUILD_INTERFACE for the same reason as Boost::graph on ddx_rt: the fetched
# target is not ours to export, so ddx-config.cmake finds Mp11 again instead.
ddx_use_mp11()
target_link_libraries(ddx INTERFACE $<BUILD_INTERFACE:Boost::mp11>)

if (DDX_BUILD_JIT)
    # The JIT's defaults follow how the project itself was built.  Contraction
    # especially: a kernel that contracts where the compile-time path does not
    # would return different derivative values for the same expression.
    if (DDX_FP_FLAGS)
        set(DDX_JIT_DEFAULT_CONTRACT 1)
    else ()
        set(DDX_JIT_DEFAULT_CONTRACT 0)
    endif ()

    # No predefined macro carries the optimisation *level* -- GCC and Clang
    # define only __OPTIMIZE__, and only as a yes/no -- so CMake is the one
    # place that knows.  O0 is deliberately unreachable: it turns the loop
    # vectoriser off, and vectorising is the only reason a batch kernel beats
    # the compile-time path.  A debug build gets cheaper compiles, not scalar code.
    string(TOUPPER "${CMAKE_BUILD_TYPE}" DDX_BUILD_TYPE_UPPER)
    if (DDX_BUILD_TYPE_UPPER STREQUAL "DEBUG")
        set(DDX_JIT_DEFAULT_OPT 1)
    elseif (DDX_BUILD_TYPE_UPPER STREQUAL "RELEASE")
        set(DDX_JIT_DEFAULT_OPT 3)
    else ()
        set(DDX_JIT_DEFAULT_OPT 2)
    endif ()

    ddx_use_llvm()
    add_subdirectory(src/jit)
endif ()

option(DDX_INSTALL "Generate install rules" ${PROJECT_IS_TOP_LEVEL})
if (DDX_INSTALL)
    include(DdxInstall)
endif ()

# Everything below builds the executables, and only when this is the top-level
# project.  A consumer that add_subdirectory()s us gets the library and nothing else.
if (NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    return ()
endif ()

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ddx)
ddx_target_flags(main)

enable_testing()
add_subdirectory(tests)

option(DDX_BUILD_BENCHMARKS "Build Google Benchmark benchmarks" ON)
if (DDX_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()
