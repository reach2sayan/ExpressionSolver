# One target per dependency tier, so the dependency-free suite stays
# dependency-free: `tests` links nothing but GoogleTest, `tests_rt` adds
# Boost.Graph, `tests_jit` adds LLVM.
ddx_use_googletest()
include(GoogleTest)

# One TU per anonymous-namespace helper block, so `-j` can build them at once.
# `tests` is the compile-time suite with forward mode taken out, which is what
# keeps the core honest about not needing one; the cases that want a dual live
# in tests/dual/.
file(GLOB DDX_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_executable(tests ${DDX_TEST_SOURCES})
target_include_directories(tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(tests PRIVATE gtest_main ddx)
ddx_target_flags(tests)
# libLLVM will never be instrumented, so without ignore_noninstrumented_modules
# every ORC internal is a report rather than a finding.
set(DDX_TEST_ENV "")
if (DDX_SANITIZE STREQUAL "thread")
    set(DDX_TEST_ENV PROPERTIES ENVIRONMENT
        "TSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/tsan.supp:ignore_noninstrumented_modules=1")
endif ()

gtest_discover_tests(tests ${DDX_TEST_ENV})

file(GLOB DDX_DUAL_TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/dual/*.cpp")
add_executable(tests_dual ${DDX_DUAL_TEST_SOURCES})
target_include_directories(tests_dual PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(tests_dual PRIVATE gtest_main ddx)
ddx_target_flags(tests_dual)
gtest_discover_tests(tests_dual ${DDX_TEST_ENV})

# The batch tests are written once and must answer the same interpreted or
# compiled, which one link line gets: ddx::rt is libddx either way.
file(GLOB DDX_RT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/rt/*.cpp")
add_executable(tests_rt ${DDX_RT_TEST_SOURCES})
target_link_libraries(tests_rt PRIVATE gtest_main ddx::rt)
ddx_target_flags(tests_rt)
ddx_runtime_deps(tests_rt)
gtest_discover_tests(tests_rt ${DDX_TEST_ENV})

if (DDX_BUILD_JIT)
    file(GLOB DDX_JIT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/jit/*.cpp")
    add_executable(tests_jit ${DDX_JIT_TEST_SOURCES})
    target_link_libraries(tests_jit PRIVATE gtest_main ddx::jit)
    ddx_target_flags(tests_jit)
    ddx_runtime_deps(tests_jit)
    gtest_discover_tests(tests_jit ${DDX_TEST_ENV})

    # Its own process: the assertion is the exit status, after main returned.
    add_executable(tests_jit_exit_stress jit/stress/exit_stress.cpp)
    target_link_libraries(tests_jit_exit_stress PRIVATE ddx::jit)
    ddx_target_flags(tests_jit_exit_stress)
    ddx_runtime_deps(tests_jit_exit_stress)
    add_test(NAME jit.exit_stress COMMAND tests_jit_exit_stress)
    if (DDX_TEST_ENV)
        set_tests_properties(jit.exit_stress ${DDX_TEST_ENV})
    endif ()
endif ()

# --- packaging ---------------------------------------------------------------
# The only consumer of ddx from outside, so the only thing that can catch a
# broken export set, a config resolving by path, or a component request that
# succeeds against a build without it.  A nested CMake per case is slow and
# inherits neither our compiler pick nor our cache, hence everything below.
option(DDX_TEST_PACKAGE "Build the out-of-tree consumer tests" ${PROJECT_IS_TOP_LEVEL})
# Not under a sanitizer: these configure and build a consumer out of tree, and
# an uninstrumented consumer cannot link an instrumented libddx.
if (DDX_TEST_PACKAGE AND DDX_INSTALL AND DDX_SANITIZE STREQUAL "off")
    # rt and dual are in every build; only the backend is a question.
    set(DDX_PACKAGE_COMPONENTS rt dual)
    if (DDX_BUILD_JIT)
        list(APPEND DDX_PACKAGE_COMPONENTS jit)
    endif ()
    # A list has to reach the nested cmake as one argument, and an escaped
    # separator does not survive add_test()'s own expansion -- every component
    # but the first became a stray positional.  $<SEMICOLON> goes into the
    # testfile literally and is a list again by the time the cache sees it.
    string(JOIN "$<SEMICOLON>" DDX_PACKAGE_COMPONENTS_ARG ${DDX_PACKAGE_COMPONENTS})

    # No Boost is handed down: ddx-config carries its own include path now, so
    # a consumer needs nothing found.  What the nested project cannot work out
    # for itself is the compiler and the configuration, having no toolchain file
    # and no cache -- and install_tree cannot fall back on an inherited
    # CMAKE_PREFIX_PATH either, since it sets its own.
    set(DDX_PACKAGE_COMMON_OPTIONS
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DDDX_CONSUMER_COMPONENTS=${DDX_PACKAGE_COMPONENTS_ARG})
    if (DDX_BUILD_JIT)
        list(APPEND DDX_PACKAGE_COMMON_OPTIONS -DLLVM_DIR=${LLVM_DIR})
    endif ()

    # A multi-config generator carries the configuration on the command line,
    # and none of it reaches a nested cmake: left alone the consumer is built
    # Debug against a Release ddx.  --build-project is the other half -- ctest
    # names the solution file after the project, and otherwise tries ".slnx".
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

    # 2. An install prefix: the build tree's exported targets point into the
    #    source and build directories, an installed one's inside the prefix.
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
