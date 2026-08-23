cmake_minimum_required(VERSION 3.26)

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
# One interface target per header directory, so that what a consumer links and
# what they can include are the same statement.  ddx_util owns the include path
# and everything else inherits it by linking, which keeps the install interface
# in exactly one place.
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

add_library(ddx_util INTERFACE)
add_library(ddx::util ALIAS ddx_util)
set_target_properties(ddx_util PROPERTIES EXPORT_NAME util)
target_include_directories(ddx_util INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if (DDX_MDSPAN_VENDORED)
    target_compile_definitions(ddx_util INTERFACE DDX_MDSPAN_FORCE_VENDORED)
endif ()
if (DEFINED DDX_DEDUCING_THIS_VALUE)
    target_compile_definitions(ddx_util INTERFACE DDX_DEDUCING_THIS=${DDX_DEDUCING_THIS_VALUE})
endif ()

# No Boost here, deliberately: the operation descriptors are written against
# the concepts in ops/numeric.hpp and nothing else, which is what lets the
# runtime graph link ops without acquiring the compile-time tree.
add_library(ddx_ops INTERFACE)
add_library(ddx::ops ALIAS ddx_ops)
set_target_properties(ddx_ops PROPERTIES EXPORT_NAME ops)
target_link_libraries(ddx_ops INTERFACE ddx_util)

add_library(ddx_md INTERFACE)
add_library(ddx::md ALIAS ddx_md)
set_target_properties(ddx_md PROPERTIES EXPORT_NAME md)
target_link_libraries(ddx_md INTERFACE ddx_util ddx_ops)

# Boost.Mp11 reaches exactly one header, symbolic/symbol.hpp, so it is declared
# exactly here.  Boost::headers is the target because that is what a monolithic
# BoostConfig offers for the header-only libraries, and an include path is all
# any of them amount to.  It is carried in the export set like any other found
# package: ddx-config.cmake finds Boost again and the name resolves.
add_library(ddx_symbolic INTERFACE)
add_library(ddx::symbolic ALIAS ddx_symbolic)
set_target_properties(ddx_symbolic PROPERTIES EXPORT_NAME symbolic)
target_link_libraries(ddx_symbolic INTERFACE ddx_ops ddx_md Boost::headers)

# Forward mode, which every build has: ddx computes derivatives both ways and
# the second-derivative entry points are not something a configuration takes
# away.
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
# Everything above is headers.  This is the one binary the project produces: the
# runtime expression graph always, and the LLVM backend compiled into the same
# library when it is asked for.  There is no second artifact -- ddx::jit names
# the same file ddx::rt does.
if (DEFINED DDX_BUILD_RT AND NOT DDX_BUILD_RT)
    message(WARNING "DDX_BUILD_RT is gone: the runtime graph is always built.")
endif ()
if (DEFINED DDX_JIT_STATIC)
    message(WARNING "DDX_JIT_STATIC is gone: use -DDDX_SHARED_LIBS=OFF.")
endif ()

option(DDX_BUILD_JIT "Build the LLVM JIT backend into libddx" OFF)

# One knob for the one library ddx produces.
option(DDX_SHARED_LIBS "Build libddx as a shared library" ON)
if (DDX_SHARED_LIBS)
    set(DDX_LIBRARY_TYPE SHARED)
else ()
    set(DDX_LIBRARY_TYPE STATIC)
endif ()

add_subdirectory(src/rt)
add_subdirectory(include/rt)

if (DDX_BUILD_JIT)
    # The JIT's defaults follow how the project itself was built.
    if (DDX_FP_FLAGS)
        set(DDX_JIT_DEFAULT_CONTRACT 1)
    else ()
        set(DDX_JIT_DEFAULT_CONTRACT 0)
    endif ()

    # No predefined macro carries the optimisation *level* -- GCC and Clang
    # define only __OPTIMIZE__, and only as a yes/no -- so CMake is the one
    # place that knows.
    # O0 is deliberately unreachable: it turns the loop
    # vectoriser off
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
if (NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    return ()
endif ()

# The tour walks every entry point, second derivatives included.
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ddx)
ddx_target_flags(main)

enable_testing()
add_subdirectory(tests)

option(DDX_BUILD_BENCHMARKS "Build Google Benchmark benchmarks" ON)
if (DDX_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()
