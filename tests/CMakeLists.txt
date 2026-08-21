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
