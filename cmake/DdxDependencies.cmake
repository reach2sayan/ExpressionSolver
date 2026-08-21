# Every third-party dependency ddx uses, in one place: what it is, which version,
# and where it comes from.  Declaring is cheap -- FetchContent downloads nothing
# until the matching ddx_use_*() macro asks for it -- so the whole manifest can
# live here while a build still fetches only what its options turned on.
#
#   ddx_use_boost()            Boost.Graph + Boost.DynamicBitset   DDX_BUILD_RT
#   ddx_use_mp11()             Boost.Mp11                          always
#   ddx_use_llvm()             LLVM 18-20, found not fetched       DDX_BUILD_JIT
#   ddx_use_googletest()       GoogleTest                          top-level only
#   ddx_use_googlebenchmark()  Google Benchmark                    DDX_BUILD_BENCHMARKS
#
# Each is idempotent and safe to call from wherever the dependency is first
# needed.  Macros rather than functions: FetchContent_MakeAvailable() defines
# variables the caller's scope is expected to see.
include_guard(GLOBAL)

# Before include(FetchContent), which defines this itself and would make the
# guard below a no-op.
if (NOT DEFINED CACHE{FETCHCONTENT_BASE_DIR})
    set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.deps" CACHE PATH
            "Shared download cache for fetched dependencies")
endif ()
include(FetchContent)

# --- versions ---------------------------------------------------------------
set(DDX_BOOST_VERSION "1.92.0" CACHE STRING "Boost release to fetch")
set(DDX_GOOGLETEST_REF "5376968f6948923e2411081fd9372e71a59d8e77"
        CACHE STRING "GoogleTest commit to fetch")
set(DDX_GOOGLEBENCHMARK_VERSION "1.9.1" CACHE STRING "Google Benchmark release to fetch")

# The LLVM range ddx::jit has been built against.
# The ORC C++ API is not stable across releases, hence a range rather than a floor.
set(DDX_LLVM_VERSION_MIN 18)
set(DDX_LLVM_VERSION_MAX 20)

# --- declarations -----------------------------------------------------------
# SYSTEM throughout: a fetched dependency's warnings are not ours to fix.
FetchContent_Declare(Boost
        URL https://github.com/boostorg/boost/releases/download/boost-${DDX_BOOST_VERSION}/boost-${DDX_BOOST_VERSION}-cmake.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM
)
FetchContent_Declare(mp11
        URL https://github.com/boostorg/mp11/archive/refs/tags/boost-${DDX_BOOST_VERSION}.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM
)
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

# --- Boost.Graph, Boost.DynamicBitset ---------------------------------------
# Boost.Graph for the frozen graph's CSR and the Hessian colouring, and
# Boost.DynamicBitset for the coupling rows.  Header-only, so the compiled Boost
# libraries are not needed.
macro(ddx_use_boost)
    if (NOT TARGET Boost::graph)
        set(BOOST_INCLUDE_LIBRARIES graph dynamic_bitset CACHE STRING "" FORCE)
        set(BOOST_ENABLE_CMAKE ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(Boost)
    endif ()
endmacro()

# --- Boost.Mp11 -------------------------------------------------------------

# Call this *after* ddx_use_boost() where both are wanted: Boost.Graph depends
# on Mp11, so the superproject has already defined Boost::mp11, and defining a
# target twice is an error rather than a merge.  The guard is on the target so
# that it holds however Mp11 arrived -- a parent project that brought its own
# Boost included.
macro(ddx_use_mp11)
    if (NOT TARGET Boost::mp11)
        FetchContent_MakeAvailable(mp11)
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
