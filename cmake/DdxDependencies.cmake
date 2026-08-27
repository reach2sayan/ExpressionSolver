# Every third-party dependency, in one place.  LLVM and pybind11 are *found*,
# being compiled things the machine owns; the rest are fetched against a pinned
# hash, and declaring one downloads nothing until its macro asks.
#
#   ddx_use_boost()            Boost, header-only, fetched         always
#   ddx_use_llvm()             LLVM 20, found                      DDX_BUILD_JIT
#   ddx_use_googletest()       GoogleTest, fetched                 top-level only
#   ddx_use_googlebenchmark()  Google Benchmark, fetched           DDX_BUILD_BENCHMARKS
#   ddx_use_pybind11()         pybind11, found in the build env    DDX_BUILD_PYTHON
#
# Each is idempotent.  Macros, not functions: FetchContent_MakeAvailable()
# defines variables the caller's scope is expected to see.
include_guard(GLOBAL)

include(FetchContent)

# --- versions ---------------------------------------------------------------
# LLVM is whatever the machine has; the rest are pinned here.
set(DDX_GOOGLETEST_REF "5376968f6948923e2411081fd9372e71a59d8e77"
        CACHE STRING "GoogleTest commit to fetch")
set(DDX_GOOGLEBENCHMARK_VERSION "1.9.1" CACHE STRING "Google Benchmark release to fetch")

# Exact, not a floor: the ORC C++ API is not stable across releases and both
# ends bite.  Below 20, Intrinsic::getOrInsertDeclaration and lookupIntrinsicID
# do not exist and getHostCPUFeatures() has another signature; above it, 21
# replaces the `nocapture` attribute codegen.cpp sets.
set(DDX_LLVM_VERSION 20)

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
# coupling rows; Describe, Endian and CRC for the saved graph; Parser for the
# text surface.  All header-only.
# Boost.Serialization is deliberately absent -- see boost_no_exceptions.cpp.
#
# Fetched and pinned, and never found: the machine's Boost is not consulted at
# all, because a distribution's is a lottery this build loses.  BOOST_ROOT and
# Boost_DIR are ignored along with the rest.  Nothing here is compiled -- the
# headers are used where they land -- so SOURCE_SUBDIR names a directory the
# archive does not have and MakeAvailable stops once it has unpacked.
#
# DDX_BOOST_INCLUDEDIR is the one way past that, and it is all or nothing: name
# a directory holding boost/, and that tree is used with no fetch and no
# fallback.  A tree without boost/version.hpp is a hard error rather than a
# quiet download, so a typo in the override cannot be mistaken for the pin.
#
# Unpacked beside the source rather than under a build tree: eight presets share
# this checkout, and each would otherwise keep its own copy of all of Boost.
set(DDX_BOOST_INCLUDEDIR "" CACHE PATH
        "A Boost include root to use instead of the pinned fetch; must hold boost/version.hpp")
set(DDX_BOOST_VERSION "1.92.0" CACHE STRING "Boost release to fetch")
set(DDX_BOOST_SHA256 "ea7b982002cc9dfbe59b0b217b206f470dc75f3de0bb2973d844118934d82411"
        CACHE STRING "SHA256 of the archive DDX_BOOST_VERSION names")
cmake_path(SET _ddx_deps_default NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../.deps")
set(DDX_DEPS_DIR "${_ddx_deps_default}"
        CACHE PATH "Where fetched Boost is unpacked; shared by every build tree")
unset(_ddx_deps_default)

FetchContent_Declare(boost
        URL https://github.com/boostorg/boost/releases/download/boost-${DDX_BOOST_VERSION}/boost-${DDX_BOOST_VERSION}-b2-nodocs.tar.xz
        URL_HASH SHA256=${DDX_BOOST_SHA256}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR ddx-does-not-build-boost
        SOURCE_DIR "${DDX_DEPS_DIR}/boost-${DDX_BOOST_VERSION}"
        SUBBUILD_DIR "${DDX_DEPS_DIR}/boost-${DDX_BOOST_VERSION}-subbuild"
        BINARY_DIR "${DDX_DEPS_DIR}/boost-${DDX_BOOST_VERSION}-build"
)

macro(ddx_use_boost)
    if (NOT TARGET Boost::headers)
        if (DDX_BOOST_INCLUDEDIR)
            _ddx_adopt_boost("${DDX_BOOST_INCLUDEDIR}")
        else ()
            FetchContent_MakeAvailable(boost)
            _ddx_adopt_boost("${boost_SOURCE_DIR}")
        endif ()
    endif ()
endmacro()

# The include root becomes Boost::headers, whichever of the two supplied it.
# IMPORTED, which is what find_package(Boost CONFIG) would have left: the name
# every target already links, carrying an include path and belonging to no
# export set.  DdxInstall copies the libraries ddx names out of this root into
# ddx's own prefix, so a consumer needs no Boost of their own -- which is the
# same rule as the build, where the machine's Boost is never consulted.
macro(_ddx_adopt_boost root)
    if (NOT EXISTS "${root}/boost/version.hpp")
        message(FATAL_ERROR
                "No Boost at ${root}: it has no boost/version.hpp.  "
                "DDX_BOOST_INCLUDEDIR must name the directory *containing* boost/, "
                "and setting it turns the pinned fetch off entirely.  Unset it to "
                "build against the fetched Boost ${DDX_BOOST_VERSION} instead.")
    endif ()
    add_library(Boost::headers INTERFACE IMPORTED GLOBAL)
    set_target_properties(Boost::headers PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${root}")
    # What ddx-config.cmake reports back when a consumer's Boost differs, read
    # from the tree rather than assumed: an override is any version at all.
    file(STRINGS "${root}/boost/version.hpp" _ddx_boost_version_line
            REGEX "^#define BOOST_LIB_VERSION ")
    string(REGEX REPLACE ".*\"([0-9_]+)\".*" "\\1" Boost_VERSION "${_ddx_boost_version_line}")
    string(REPLACE "_" "." Boost_VERSION "${Boost_VERSION}")
    unset(_ddx_boost_version_line)
    # Where DdxInstall reads the headers it ships.
    set(DDX_BOOST_ROOT "${root}")
    message(STATUS "Boost ${Boost_VERSION} (${root})")
endmacro()

# --- LLVM -------------------------------------------------------------------
# Sets LLVM_DEFINITIONS_LIST and DDX_LLVM_LIBS for src/jit.
macro(ddx_use_llvm)
    if (NOT LLVM_FOUND)
        # LLVMConfig matches only an exact version request, so a find_package
        # range never does; check the major here.
        find_package(LLVM REQUIRED CONFIG)
        if (NOT LLVM_VERSION_MAJOR EQUAL DDX_LLVM_VERSION)
            message(FATAL_ERROR
                    "ddx::jit is built against LLVM ${DDX_LLVM_VERSION}; found "
                    "${LLVM_PACKAGE_VERSION}.  Point CMake at it with "
                    "-DLLVM_DIR=/usr/lib/llvm-${DDX_LLVM_VERSION}/lib/cmake/llvm.")
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

