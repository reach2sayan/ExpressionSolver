cmake_minimum_required(VERSION 3.26)

# Newest GCC on PATH, unless the caller chose a compiler.  Must precede project().
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
        AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    find_program(DDX_PREFERRED_CXX NAMES g++-15 g++-14)
    if (DDX_PREFERRED_CXX)
        set(CMAKE_CXX_COMPILER "${DDX_PREFERRED_CXX}")
    endif ()
endif ()

project(ddx VERSION 1.2.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# This file is the assembly order; the modules under cmake/ are the parts.
#   DdxToolchainProbes  the floor a compiler must clear
#   DdxCompileOptions   our own targets' flags     -> ddx_target_flags(), ...
#   DdxDependencies     every third-party pin      -> ddx_use_boost(), ...
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(DdxToolchainProbes)
include(DdxCompileOptions)
include(DdxDependencies)

# --- the library ------------------------------------------------------------
# One interface target per header directory, so what a consumer links and what
# they can include are one statement.  ddx_util owns the include path and the
# rest inherit it, which keeps the install interface in one place.
#
#   ddx::util      language configuration, errors, fixed strings   (leaf)
#   ddx::ops       the operation vocabulary and its derivative rules
#   ddx::md        the multidimensional storage the drivers hand back
#   ddx::symbolic  the compile-time graph: tree, point, sweeps
#   ddx::dual      forward mode -- duals, and the Hessian entry points
#   ddx::ddx       all of the above, which is what most consumers want
#
# ops/ and symbolic/ are fully constexpr and stay that way; nothing below may
# put a run-time dependency in their path.
ddx_use_boost()

# rt/equation.hpp is a public header that compiles a graph on another thread, so
# a consumer needs the threading library as much as ddx_rt does.
find_package(Threads REQUIRED)

# Boost is on the leaf, so every target below inherits it.  ddx fetches Boost
# at a pinned hash and installs it into its own prefix, so it is as available as
# the standard library is and there is nothing for a narrower declaration to
# protect: a header uses the right library rather than the locally permitted one.
add_library(ddx_util INTERFACE)
add_library(ddx::util ALIAS ddx_util)
set_target_properties(ddx_util PROPERTIES EXPORT_NAME util)
target_include_directories(ddx_util INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(ddx_util INTERFACE Boost::headers)

add_library(ddx_ops INTERFACE)
add_library(ddx::ops ALIAS ddx_ops)
set_target_properties(ddx_ops PROPERTIES EXPORT_NAME ops)
target_link_libraries(ddx_ops INTERFACE ddx_util)

add_library(ddx_md INTERFACE)
add_library(ddx::md ALIAS ddx_md)
set_target_properties(ddx_md PROPERTIES EXPORT_NAME md)
target_link_libraries(ddx_md INTERFACE ddx_util ddx_ops)

add_library(ddx_symbolic INTERFACE)
add_library(ddx::symbolic ALIAS ddx_symbolic)
set_target_properties(ddx_symbolic PROPERTIES EXPORT_NAME symbolic)
target_link_libraries(ddx_symbolic INTERFACE ddx_ops ddx_md)

# Forward mode
add_library(ddx_dual INTERFACE)
add_library(ddx::dual ALIAS ddx_dual)
set_target_properties(ddx_dual PROPERTIES EXPORT_NAME dual)
target_link_libraries(ddx_dual INTERFACE ddx_symbolic)

add_library(ddx INTERFACE)
add_library(ddx::ddx ALIAS ddx)
target_link_libraries(ddx INTERFACE ddx_util ddx_ops ddx_md ddx_symbolic ddx_dual)

# Each header directory lists its own files onto its own target, for IDE
# visibility; nothing is compiled from there.
add_subdirectory(include)

# --- libddx -----------------------------------------------------------------
option(DDX_BUILD_JIT "Build the LLVM JIT backend into libddx" OFF)

add_subdirectory(src/rt)
add_subdirectory(include/rt)

if (DDX_BUILD_JIT)
    # The JIT's defaults follow how the project itself was built.
    if (DDX_FP_FLAGS)
        set(DDX_JIT_DEFAULT_CONTRACT 1)
    else ()
        set(DDX_JIT_DEFAULT_CONTRACT 0)
    endif ()

    # No predefined macro carries the optimisation *level*, only __OPTIMIZE__
    # as a yes/no, so CMake is the one place that knows.  Never 0.
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
    add_subdirectory(include/jit)
endif ()

option(DDX_INSTALL "Generate install rules" ${PROJECT_IS_TOP_LEVEL})
if (DDX_INSTALL)
    include(DdxInstall)
endif ()

# Everything below builds the executables, and only when this is the top-level
# project.  A consumer that add_subdirectory()s us gets the library and nothing else.
if (NOT PROJECT_IS_TOP_LEVEL)
    return ()
endif ()

# The tour walks every entry point, second derivatives included.
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ddx)
ddx_target_flags(main)

enable_testing()
add_subdirectory(tests)

# After enable_testing(), which it registers a pytest run with; below the
# top-level guard, a consumer add_subdirectory()ing ddx wanting the library.
option(DDX_BUILD_PYTHON "Build the pybind11 extension module" OFF)
if (DDX_BUILD_PYTHON)
    ddx_use_pybind11()
    add_subdirectory(python)
endif ()

option(DDX_BUILD_BENCHMARKS "Build Google Benchmark benchmarks" ON)
if (DDX_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()
