add_library(minimizer SHARED src/minimizer.cpp)

target_include_directories(minimizer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(minimizer PUBLIC diff Eigen3::Eigen)

set_target_properties(minimizer PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

if (NOT MSVC)
    target_compile_options(minimizer PRIVATE -Wall -Wpedantic -Wfatal-errors -Wextra)
else()
    target_compile_options(minimizer PRIVATE /Wall)
    set_target_properties(minimizer PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_executable(minimizer_tests tests/minimizer_tests.cpp)
target_link_libraries(minimizer_tests PRIVATE minimizer gtest_main)
target_include_directories(minimizer_tests PRIVATE ${CMAKE_SOURCE_DIR})
set_target_properties(minimizer_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
add_test(NAME minimizer_tests COMMAND minimizer_tests)
