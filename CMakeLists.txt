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

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-std=c++23")

# --- <expected> -------------------------------------------------------------
# ddx::result is std::expected, and libstdc++ offers <expected> only where
# __cpp_concepts is 202002L -- which Clang first defines in 19.  Unguarded, the
# toolchain floor shows up as a missing name partway down an include stack.
# MSVC is exempt because the probe's -std= is not its spelling.
check_cxx_source_compiles(
        "#include <expected>
         int main() { return std::expected<int, int>{0}.value(); }"
        DDX_HAS_STD_EXPECTED)
if (NOT MSVC AND NOT DDX_HAS_STD_EXPECTED)
    message(FATAL_ERROR
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} has no usable <expected>.  "
            "Build with GCC 14+, Clang 19+ over libstdc++, or Clang 17+ with -stdlib=libc++.")
endif ()

# --- accessor spelling ------------------------------------------------------
# include/util/config.hpp owns the decision; this only overrides it.
#   auto - the compiler's own answer (__cpp_explicit_this_parameter)
#   on   - deducing this even where the macro is absent
#   off  - the ref-qualified overload set
# Two probes: what `auto` picks up, and what `on` would require.
set(DDX_DEDUCING_THIS "auto" CACHE STRING "Accessor spelling: auto|on|off")
set_property(CACHE DDX_DEDUCING_THIS PROPERTY STRINGS auto on off)

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

if (DDX_DEDUCING_THIS STREQUAL "on" AND NOT DDX_DEDUCING_THIS_COMPILES)
    message(FATAL_ERROR
            "DDX_DEDUCING_THIS=on but ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} "
            "cannot compile an explicit object parameter.  Use auto or off.")
elseif (DDX_DEDUCING_THIS STREQUAL "on")
    set(DDX_DEDUCING_THIS_VALUE 1)
    set(DDX_ACCESSORS "deducing this, forced on")
elseif (DDX_DEDUCING_THIS STREQUAL "off")
    set(DDX_DEDUCING_THIS_VALUE 0)
    set(DDX_ACCESSORS "ref-qualified overload set, forced off")
elseif (NOT DDX_DEDUCING_THIS STREQUAL "auto")
    message(FATAL_ERROR "DDX_DEDUCING_THIS must be auto, on or off (got '${DDX_DEDUCING_THIS}').")
elseif (DDX_HAS_DEDUCING_THIS)
    # auto leaves the value undefined, so a consumer building without this file agrees.
    set(DDX_ACCESSORS "deducing this")
else ()
    set(DDX_ACCESSORS "ref-qualified overload set")
endif ()
message(STATUS "accessors: ${DDX_ACCESSORS} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")

# --- which mdspan include/md/md.hpp binds to --------------------------------
#   auto     - standard <mdspan> when complete enough, vendored otherwise
#   std      - require the standard one; hard error if incomplete
#   vendored - always the copy under include/md/third_party
set(DDX_MDSPAN_MODE "auto" CACHE STRING "Which mdspan md.hpp binds to: auto|std|vendored")
set_property(CACHE DDX_MDSPAN_MODE PROPERTY STRINGS auto std vendored)

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
            "has no complete <mdspan>/<submdspan>.  Use auto or vendored.")
elseif (DDX_MDSPAN_MODE STREQUAL "vendored" OR NOT DDX_HAS_STD_MDSPAN)
    set(DDX_MDSPAN_VENDORED ON)
    message(STATUS "mdspan: vendored (include/md/third_party/mdspan.hpp)")
else ()
    message(STATUS "mdspan: standard <mdspan>")
endif ()

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

option(ENABLE_NATIVE_ARCH "Build optimized for this machine" ON)
# Pin FP contraction, drop errno.  Neither flag is lossy.  Nothing that trades
# accuracy for speed belongs here -- in particular never -ffast-math (nor
# /fp:fast on MSVC, which needs no flags at all).
option(DDX_FP_FLAGS "Pin FP contraction and drop errno on libm calls" ON)
# The headers throw nothing, so a consumer is free to build without exceptions.
# This only turns the flag on for our own targets -- the INTERFACE library must
# never push it onto a consumer, which is why it is applied in
# ddx_target_flags() below and nowhere else.  Two tests that assert on the
# standard library's own throws compile out; see DDX_NO_EXCEPTIONS in the tests.
option(DDX_NO_EXCEPTIONS "Build our own targets without C++ exceptions" OFF)

if (MSVC)
    # /bigobj: one TU instantiates past the 2^16 COFF section limit.
    set(DDX_CODEGEN_FLAGS /arch:AVX2 /bigobj)
    # /Wall noise a header-only expression-template library emits by the
    # thousand.  C4365 and C5219 are deliberately left on.  C4866 fires on every
    # subscript of a range view -- an overloaded operator[] whose operands are a
    # view and a loop counter, so there is no evaluation order to get wrong.
    set(DDX_WARNINGS /Wall /wd4623 /wd4625 /wd4626 /wd5026 /wd5027 /wd4710
                     /wd4711 /wd4868 /wd4820 /wd5045 /wd5246 /wd4514 /wd4324
                     /wd5266 /wd4866)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if (DDX_NO_EXCEPTIONS)
        list(APPEND DDX_CODEGEN_FLAGS /EHs-c- /D_HAS_EXCEPTIONS=0)
    endif ()
else ()
    if (ENABLE_NATIVE_ARCH)
        set(DDX_CODEGEN_FLAGS -march=native)
    else ()
        set(DDX_CODEGEN_FLAGS -march=x86-64-v3)
    endif ()
    if (DDX_FP_FLAGS)
        list(APPEND DDX_CODEGEN_FLAGS -ffp-contract=fast -fno-math-errno)
    endif ()
    if (DDX_NO_EXCEPTIONS)
        # -funwind-tables with it, matching how LLVM itself is built: no
        # exceptions to raise or catch, but frames still unwindable for a
        # backtrace or a profiler.
        list(APPEND DDX_CODEGEN_FLAGS -fno-exceptions -funwind-tables)
    endif ()
    set(DDX_WARNINGS -Wall -Wextra -Wpedantic -Wfatal-errors)
endif ()

# Our own targets only -- never the INTERFACE library, which must not push flags
# onto a consumer.
function(ddx_target_flags target)
    target_compile_options(${target} PRIVATE ${DDX_CODEGEN_FLAGS} ${DDX_WARNINGS})
endfunction()

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

# Before include(FetchContent), which defines this itself and would make the
# guard below a no-op.
if (NOT DEFINED CACHE{FETCHCONTENT_BASE_DIR})
    set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.deps" CACHE PATH
            "Shared download cache for fetched dependencies")
endif ()
include(FetchContent)

# Pinned, not floating: "latest" that moves under you is not reproducible.
# Bump the one line.  Mp11 is fetched at the same release, so the core and the
# runtime graph never see two versions of the same headers.
set(DDX_BOOST_VERSION "1.92.0" CACHE STRING "Boost release to fetch")

if (DDX_BUILD_RT)

    # Boost.Graph for the frozen graph's CSR and the Hessian colouring, and
    # Boost.DynamicBitset for the coupling rows.  Header-only, so the compiled
    # Boost libraries are not needed.
    #
    # Fetched rather than taken from the system: a distro Boost is whatever it
    # is, and the colouring's quality is something this project asserts on.
    # BOOST_INCLUDE_LIBRARIES names the two used and the modular build resolves
    # their dependencies -- this is not a fetch of all of Boost.
    #
    # The -cmake archive is the whole Boost source (~700 MB extracted);
    # BOOST_INCLUDE_LIBRARIES limits what is *configured*, not what is
    # downloaded.  The shared cache above means one copy per checkout rather
    # than one per build tree -- five presets would otherwise be five copies.
    set(BOOST_INCLUDE_LIBRARIES graph dynamic_bitset CACHE STRING "" FORCE)
    set(BOOST_ENABLE_CMAKE ON CACHE BOOL "" FORCE)
    # SYSTEM: a fetched dependency's warnings are not ours to fix, and Boost's
    # smallest_last_ordering.hpp emits several under -Wall -Wextra.
    FetchContent_Declare(Boost
            URL https://github.com/boostorg/boost/releases/download/boost-${DDX_BOOST_VERSION}/boost-${DDX_BOOST_VERSION}-cmake.tar.xz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            SYSTEM
    )
    FetchContent_MakeAvailable(Boost)

    add_library(ddx_rt INTERFACE)
    add_library(ddx::rt ALIAS ddx_rt)
    # EXPORT_NAME so an installed ddx spells its targets the way an in-tree one
    # does: ddx::rt, not ddx::ddx_rt.  Same for ddx::jit below.
    set_target_properties(ddx_rt PROPERTIES EXPORT_NAME rt)
    # Boost is fetched for the build and found again at install time -- the
    # fetched targets belong to no export set of ours, so the installed
    # interface names them for ddx-config.cmake to resolve instead.
    target_link_libraries(ddx_rt INTERFACE ddx
            $<BUILD_INTERFACE:Boost::graph>
            $<BUILD_INTERFACE:Boost::dynamic_bitset>)
    add_subdirectory(include/rt)

    # Boost leaves boost::throw_exception undefined under -fno-exceptions and
    # makes it the program's to supply; see src/rt/boost_no_exceptions.cpp.
    # ddx_rt is INTERFACE, so this is the one compiled object it carries, and
    # only in this configuration.
    if (DDX_NO_EXCEPTIONS)
        add_library(ddx_rt_no_exceptions STATIC src/rt/boost_no_exceptions.cpp)
        # ddx_jit links this through ddx_rt and may itself be a shared library,
        # which cannot absorb a non-PIC archive.
        set_target_properties(ddx_rt_no_exceptions PROPERTIES
                POSITION_INDEPENDENT_CODE ON
                EXPORT_NAME rt_no_exceptions)
        # BUILD_INTERFACE for the same reason as ddx_rt above: PRIVATE on a static
        # library still lands in the exported interface as $<LINK_ONLY:>, and the
        # fetched Boost target is not ours to export.  Nothing is lost -- the
        # header is needed to compile this object, not to link against it.
        target_link_libraries(ddx_rt_no_exceptions PRIVATE $<BUILD_INTERFACE:Boost::throw_exception>)
        ddx_target_flags(ddx_rt_no_exceptions)
        target_link_libraries(ddx_rt INTERFACE ddx_rt_no_exceptions)
    endif ()
endif ()

# --- Boost.Mp11 -------------------------------------------------------------
# The type-list vocabulary the symbol lists are built out of, and the one
# third-party thing ddx::ddx itself needs.  Header-only, so this is a fetch and
# not a link.
#
# After the runtime graph rather than before it: Boost.Graph depends on Mp11, so
# the superproject fetched above has already defined Boost::mp11, and defining a
# target twice is an error rather than a merge.  The guard is on the target and
# not on DDX_BUILD_RT so that it holds however Mp11 arrived -- a parent project
# that brought its own Boost included.
if (NOT TARGET Boost::mp11)
    FetchContent_Declare(mp11
            URL https://github.com/boostorg/mp11/archive/refs/tags/boost-${DDX_BOOST_VERSION}.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            SYSTEM
    )
    FetchContent_MakeAvailable(mp11)
endif ()
# BUILD_INTERFACE for the same reason as Boost::graph on ddx_rt: the fetched
# target is not ours to export, so ddx-config.cmake finds Mp11 again instead.
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

    # LLVMConfig declares itself compatible only with an exact version request,
    # so a find_package version range never matches; check the major ourselves.
    # The ORC C++ API is not stable across releases, hence the range at all.
    find_package(LLVM REQUIRED CONFIG)
    if (LLVM_VERSION_MAJOR LESS 18 OR LLVM_VERSION_MAJOR GREATER 20)
        message(FATAL_ERROR
                "ddx::jit has been built against LLVM 18-20; found ${LLVM_PACKAGE_VERSION}. "
                "Point CMake at a supported one with -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm.")
    endif ()
    message(STATUS "LLVM ${LLVM_PACKAGE_VERSION} (${LLVM_INCLUDE_DIRS})")
    separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
    llvm_map_components_to_libnames(DDX_LLVM_LIBS
            core orcjit native passes support analysis target)
    add_subdirectory(src/jit)
endif ()

# A shared ddx::jit has to be findable at run time.  CMake's build RPATH covers
# the ELF build tree already; $ORIGIN is what survives the tree being moved, and
# Windows has no RPATH at all, so there the DLL is copied next to the .exe.
function(ddx_jit_runtime target)
    if (NOT DDX_BUILD_JIT OR DDX_JIT_STATIC)
        return ()
    endif ()
    set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "$ORIGIN")
    if (WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
                COMMAND_EXPAND_LISTS)
    endif ()
endfunction()

# --- install ----------------------------------------------------------------
# Consuming ddx by add_subdirectory() stays the documented route; this is what
# makes a built ddx::jit installable, which a shared library has to be to be
# worth building.
option(DDX_INSTALL "Generate install rules" ${PROJECT_IS_TOP_LEVEL})
if (DDX_INSTALL)
    include(CMakePackageConfigHelpers)

    set(DDX_EXPORT_TARGETS ddx)
    if (DDX_BUILD_RT)
        list(APPEND DDX_EXPORT_TARGETS ddx_rt)
        if (DDX_NO_EXCEPTIONS)
            list(APPEND DDX_EXPORT_TARGETS ddx_rt_no_exceptions)
        endif ()
    endif ()
    if (DDX_BUILD_JIT)
        list(APPEND DDX_EXPORT_TARGETS ddx_jit)
    endif ()

    install(TARGETS ${DDX_EXPORT_TARGETS}
            EXPORT ddxTargets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.hpp")
    install(EXPORT ddxTargets
            NAMESPACE ddx::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ddx)

    configure_package_config_file(
            cmake/ddx-config.cmake.in
            "${CMAKE_CURRENT_BINARY_DIR}/ddx-config.cmake"
            INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ddx)
    # SameMajorVersion: ddx::jit carries an SOVERSION of the major alone.
    write_basic_package_version_file(
            "${CMAKE_CURRENT_BINARY_DIR}/ddx-config-version.cmake"
            COMPATIBILITY SameMajorVersion
            ARCH_INDEPENDENT)
    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/ddx-config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/ddx-config-version.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ddx)
endif ()

# Everything below builds the executables, and only when this is the top-level
# project.  A consumer that add_subdirectory()s us gets the library and nothing else.
if (NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    return ()
endif ()

option(DDX_BUILD_BENCHMARKS "Build Google Benchmark benchmarks" ON)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE ddx)
ddx_target_flags(main)

# One TU per anonymous-namespace helper block, so `-j` can build them at once.
file(GLOB DDX_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
add_executable(tests ${DDX_TEST_SOURCES})
target_include_directories(tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
target_link_libraries(tests PRIVATE gtest_main ddx)
ddx_target_flags(tests)

enable_testing()
gtest_discover_tests(tests)

# The runtime-graph tests live in their own target: they need Boost.Graph, which
# the dependency-free suite above must not link.
if (DDX_BUILD_RT)
    file(GLOB DDX_RT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/rt/*.cpp")
    add_executable(tests_rt ${DDX_RT_TEST_SOURCES})
    # Against the JIT where it is built, the interpreter where it is not: the
    # batch tests are written once and have to give the same answers either way,
    # which is only checked if they actually run both.
    if (DDX_BUILD_JIT)
        target_link_libraries(tests_rt PRIVATE gtest_main ddx_jit)
    else ()
        target_link_libraries(tests_rt PRIVATE gtest_main ddx_rt)
    endif ()
    ddx_target_flags(tests_rt)
    ddx_jit_runtime(tests_rt)
    gtest_discover_tests(tests_rt)
endif ()

if (DDX_BUILD_JIT)
    file(GLOB DDX_JIT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/jit/*.cpp")
    add_executable(tests_jit ${DDX_JIT_TEST_SOURCES})
    target_link_libraries(tests_jit PRIVATE gtest_main ddx_jit)
    ddx_target_flags(tests_jit)
    ddx_jit_runtime(tests_jit)
    gtest_discover_tests(tests_jit)
endif ()

if (DDX_BUILD_BENCHMARKS)
    FetchContent_Declare(
            googlebenchmark
            URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            SYSTEM
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)

    add_executable(benchmarks benchmarks/benchmark.cpp)
    target_link_libraries(benchmarks PRIVATE benchmark::benchmark ddx)
    ddx_target_flags(benchmarks)

    if (DDX_BUILD_JIT)
        add_executable(benchmarks_jit benchmarks/jit/benchmark_jit.cpp)
        target_link_libraries(benchmarks_jit PRIVATE benchmark::benchmark ddx_jit)
        ddx_target_flags(benchmarks_jit)
        ddx_jit_runtime(benchmarks_jit)
    endif ()

    add_custom_target(benchmark_json
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/benchmark-results"
            COMMAND $<TARGET_FILE:benchmarks>
            --benchmark_min_time=0.05s
            --benchmark_out="${CMAKE_BINARY_DIR}/benchmark-results/benchmarks.json"
            --benchmark_out_format=json
            DEPENDS benchmarks
            USES_TERMINAL
            COMMENT "Running benchmarks and exporting JSON results")
endif ()

# Fetched dependencies are not ours to warn about.
if (MSVC)
    foreach (dep gtest gtest_main gmock gmock_main benchmark benchmark_main)
        if (TARGET ${dep})
            target_compile_options(${dep} PRIVATE /W0 /wd4668)
            set_target_properties(${dep} PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
        endif ()
    endforeach ()
endif ()
