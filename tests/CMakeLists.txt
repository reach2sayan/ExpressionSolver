# One target per dependency tier, so the dependency-free suite stays
# dependency-free: `tests` links nothing but GoogleTest, `tests_rt` adds
# Boost.Graph, `tests_jit` adds LLVM.
ddx_use_googletest()
include(GoogleTest)

# One TU per anonymous-namespace helper block, so `-j` can build them at once.
file(GLOB DDX_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_executable(tests ${DDX_TEST_SOURCES})
target_include_directories(tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(tests PRIVATE gtest_main ddx)
ddx_target_flags(tests)
gtest_discover_tests(tests)

if (DDX_BUILD_RT)
    file(GLOB DDX_RT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/rt/*.cpp")
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
    file(GLOB DDX_JIT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/jit/*.cpp")
    add_executable(tests_jit ${DDX_JIT_TEST_SOURCES})
    target_link_libraries(tests_jit PRIVATE gtest_main ddx_jit)
    ddx_target_flags(tests_jit)
    ddx_jit_runtime(tests_jit)
    gtest_discover_tests(tests_jit)
endif ()

# --- packaging ---------------------------------------------------------------
# Nothing else in the tree consumes ddx from outside, so nothing else can catch
# a broken export set, a config that resolves by path, or a component request
# that succeeds against a build which never built that component.  These
# configure and build tests/package as its own project against both trees.
#
# Opt-out rather than opt-in, but off by default under a parent: a nested CMake
# per case is slow, and it inherits neither our compiler pick nor our cache, so
# everything it needs is passed explicitly below.
option(DDX_TEST_PACKAGE "Build the out-of-tree consumer tests" ${PROJECT_IS_TOP_LEVEL})
if (DDX_TEST_PACKAGE AND DDX_INSTALL)
    set(DDX_PACKAGE_COMPONENTS "")
    if (DDX_BUILD_RT)
        list(APPEND DDX_PACKAGE_COMPONENTS rt)
    endif ()
    if (DDX_BUILD_JIT)
        list(APPEND DDX_PACKAGE_COMPONENTS jit)
    endif ()
    string(REPLACE ";" "\;" DDX_PACKAGE_COMPONENTS_ARG "${DDX_PACKAGE_COMPONENTS}")

    set(DDX_PACKAGE_COMMON_OPTIONS
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DDDX_CONSUMER_COMPONENTS=${DDX_PACKAGE_COMPONENTS_ARG})
    if (DDX_BUILD_JIT)
        list(APPEND DDX_PACKAGE_COMMON_OPTIONS -DLLVM_DIR=${LLVM_DIR})
    endif ()

    # 1. The build tree, with nothing installed anywhere.  ddx_DIR is the build
    #    directory itself, which is what export(EXPORT) makes possible.
    add_test(NAME package.build_tree
            COMMAND ${CMAKE_CTEST_COMMAND}
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/package"
                             "${CMAKE_CURRENT_BINARY_DIR}/package/build_tree"
            --build-generator ${CMAKE_GENERATOR}
            --build-options -Dddx_DIR=${PROJECT_BINARY_DIR} ${DDX_PACKAGE_COMMON_OPTIONS}
            --test-command ${CMAKE_CTEST_COMMAND} --output-on-failure)

    # 2. An install prefix, which is the other half: the build tree's exported
    #    targets point into the source and build directories, an installed one's
    #    point inside the prefix, and only the second exercises the install rules.
    set(DDX_PACKAGE_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package/prefix")
    add_test(NAME package.install
            COMMAND ${CMAKE_COMMAND} --install ${PROJECT_BINARY_DIR}
                    --prefix ${DDX_PACKAGE_PREFIX})
    set_tests_properties(package.install PROPERTIES FIXTURES_SETUP ddx_installed)

    add_test(NAME package.install_tree
            COMMAND ${CMAKE_CTEST_COMMAND}
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/package"
                             "${CMAKE_CURRENT_BINARY_DIR}/package/install_tree"
            --build-generator ${CMAKE_GENERATOR}
            --build-options -DCMAKE_PREFIX_PATH=${DDX_PACKAGE_PREFIX} ${DDX_PACKAGE_COMMON_OPTIONS}
            --test-command ${CMAKE_CTEST_COMMAND} --output-on-failure)
    set_tests_properties(package.install_tree PROPERTIES FIXTURES_REQUIRED ddx_installed)

    # 3. Asking for a component this build does not have must fail at configure
    #    time.  Only checkable where something is genuinely absent.
    if (NOT DDX_BUILD_JIT)
        add_test(NAME package.missing_component
                COMMAND ${CMAKE_CTEST_COMMAND}
                --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/package"
                                 "${CMAKE_CURRENT_BINARY_DIR}/package/missing_component"
                --build-generator ${CMAKE_GENERATOR}
                --build-options -Dddx_DIR=${PROJECT_BINARY_DIR}
                                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                                -DDDX_CONSUMER_COMPONENTS=jit)
        set_tests_properties(package.missing_component PROPERTIES WILL_FAIL TRUE)
    endif ()
endif ()
