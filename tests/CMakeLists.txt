# One target per dependency tier, so the dependency-free suite stays
# dependency-free: `tests` links nothing but GoogleTest, `tests_rt` adds
# Boost.Graph, `tests_jit` adds LLVM.
ddx_use_googletest()
include(GoogleTest)

# One TU per anonymous-namespace helper block, so `-j` can build them at once.
#
# `tests` is the compile-time suite with forward mode taken out of it, so it is
# what keeps the core honest about not needing one: every case in it compiles
# and runs without a dual in sight.  The cases that do want one live in
# tests/dual/, which is the same split the tiering above already makes for
# Boost.Graph and LLVM.
file(GLOB DDX_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_executable(tests ${DDX_TEST_SOURCES})
target_include_directories(tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(tests PRIVATE gtest_main ddx)
ddx_target_flags(tests)
gtest_discover_tests(tests)

file(GLOB DDX_DUAL_TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/dual/*.cpp")
add_executable(tests_dual ${DDX_DUAL_TEST_SOURCES})
target_include_directories(tests_dual PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(tests_dual PRIVATE gtest_main ddx)
ddx_target_flags(tests_dual)
gtest_discover_tests(tests_dual)

# The batch tests are written once and have to give the same answers whether
# they run interpreted or JIT-compiled, which is only checked if they actually
# run both.  One link line does that now: ddx::rt is libddx either way, and the
# JIT is inside it when it was built.
file(GLOB DDX_RT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/rt/*.cpp")
add_executable(tests_rt ${DDX_RT_TEST_SOURCES})
target_link_libraries(tests_rt PRIVATE gtest_main ddx::rt)
ddx_target_flags(tests_rt)
ddx_runtime_deps(tests_rt)
gtest_discover_tests(tests_rt)

if (DDX_BUILD_JIT)
    file(GLOB DDX_JIT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/jit/*.cpp")
    add_executable(tests_jit ${DDX_JIT_TEST_SOURCES})
    target_link_libraries(tests_jit PRIVATE gtest_main ddx::jit)
    # The partitioner is internal to the backend and names no LLVM type, so the
    # unit test compiles that translation unit rather than reaching into libddx
    # for a symbol hidden visibility keeps out of it.
    target_sources(tests_jit PRIVATE ${CMAKE_SOURCE_DIR}/src/jit/partition.cpp)
    target_include_directories(tests_jit PRIVATE ${CMAKE_SOURCE_DIR}/src/jit)
    ddx_target_flags(tests_jit)
    ddx_runtime_deps(tests_jit)
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
    # rt and dual are in every build; only the backend is a question.
    set(DDX_PACKAGE_COMPONENTS rt dual)
    if (DDX_BUILD_JIT)
        list(APPEND DDX_PACKAGE_COMPONENTS jit)
    endif ()
    # A list has to reach the nested cmake as one argument.  Escaping the
    # separator does not survive add_test()'s own list expansion -- the generated
    # CTestTestfile came out with "...COMPONENTS=rt" "dual", so every case but
    # the first component was dropped on the floor and handed to cmake as a
    # stray positional.  $<SEMICOLON> is written into the testfile literally and
    # is a list again by the time the consumer's cache sees it.
    string(JOIN "$<SEMICOLON>" DDX_PACKAGE_COMPONENTS_ARG ${DDX_PACKAGE_COMPONENTS})

    # Boost_DIR alongside the compiler, for the same reason LLVM_DIR goes below:
    # ddx-config.cmake re-finds Boost on the consumer's behalf, and the nested
    # project has no toolchain file and no cache to find it with.  Where the
    # parent was pointed at a Boost by one -- vcpkg on Windows -- an unhinted
    # find_package(Boost CONFIG) there fails, and the install_tree case cannot
    # even fall back on an inherited CMAKE_PREFIX_PATH because it sets its own.
    set(DDX_PACKAGE_COMMON_OPTIONS
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DBoost_DIR=${Boost_DIR}
            -DDDX_CONSUMER_COMPONENTS=${DDX_PACKAGE_COMPONENTS_ARG})
    if (DDX_BUILD_JIT)
        list(APPEND DDX_PACKAGE_COMMON_OPTIONS -DLLVM_DIR=${LLVM_DIR})
    endif ()

    # A multi-config generator carries the configuration on the command line and
    # not in the cache, and none of it reaches a nested cmake: left alone, the
    # consumer is built Debug against a Release ddx, its ctest finds a test it
    # has no configuration for, and the install has no configuration to install.
    # --build-project is the other half of the same gap -- Visual Studio is
    # driven through a solution file ctest names after the project, and without
    # a name it tries to build ".slnx".
    get_property(DDX_PACKAGE_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if (DDX_PACKAGE_MULTI_CONFIG)
        set(DDX_PACKAGE_CONFIG -C $<CONFIG>)
        set(DDX_PACKAGE_INSTALL_CONFIG --config $<CONFIG>)
    endif ()
    set(DDX_PACKAGE_PROJECT --build-project ddx_package_consumer)

    # 1. The build tree, with nothing installed anywhere.  ddx_DIR is the build
    #    directory itself, which is what export(EXPORT) makes possible.
    add_test(NAME package.build_tree
            COMMAND ${CMAKE_CTEST_COMMAND} ${DDX_PACKAGE_CONFIG}
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/package"
                             "${CMAKE_CURRENT_BINARY_DIR}/package/build_tree"
            --build-generator ${CMAKE_GENERATOR} ${DDX_PACKAGE_PROJECT}
            --build-options -Dddx_DIR=${PROJECT_BINARY_DIR} ${DDX_PACKAGE_COMMON_OPTIONS}
            --test-command ${CMAKE_CTEST_COMMAND} ${DDX_PACKAGE_CONFIG} --output-on-failure)

    # 2. An install prefix, which is the other half: the build tree's exported
    #    targets point into the source and build directories, an installed one's
    #    point inside the prefix, and only the second exercises the install rules.
    set(DDX_PACKAGE_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package/prefix")
    add_test(NAME package.install
            COMMAND ${CMAKE_COMMAND} --install ${PROJECT_BINARY_DIR}
                    ${DDX_PACKAGE_INSTALL_CONFIG} --prefix ${DDX_PACKAGE_PREFIX})
    set_tests_properties(package.install PROPERTIES FIXTURES_SETUP ddx_installed)

    add_test(NAME package.install_tree
            COMMAND ${CMAKE_CTEST_COMMAND} ${DDX_PACKAGE_CONFIG}
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/package"
                             "${CMAKE_CURRENT_BINARY_DIR}/package/install_tree"
            --build-generator ${CMAKE_GENERATOR} ${DDX_PACKAGE_PROJECT}
            --build-options -DCMAKE_PREFIX_PATH=${DDX_PACKAGE_PREFIX} ${DDX_PACKAGE_COMMON_OPTIONS}
            --test-command ${CMAKE_CTEST_COMMAND} ${DDX_PACKAGE_CONFIG} --output-on-failure)
    set_tests_properties(package.install_tree PROPERTIES FIXTURES_REQUIRED ddx_installed)

    # 3. Asking for a component this build does not have must fail at configure
    #    time.  Only checkable where something is genuinely absent.
    if (NOT DDX_BUILD_JIT)
        add_test(NAME package.missing_component
                COMMAND ${CMAKE_CTEST_COMMAND} ${DDX_PACKAGE_CONFIG}
                --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/package"
                                 "${CMAKE_CURRENT_BINARY_DIR}/package/missing_component"
                --build-generator ${CMAKE_GENERATOR} ${DDX_PACKAGE_PROJECT}
                --build-options -Dddx_DIR=${PROJECT_BINARY_DIR}
                                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                                -DBoost_DIR=${Boost_DIR}
                                -DDDX_CONSUMER_COMPONENTS=jit)
        set_tests_properties(package.missing_component PROPERTIES WILL_FAIL TRUE)
    endif ()
endif ()
