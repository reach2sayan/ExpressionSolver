# Header directories, in dependency order: util/ knows nothing about the rest,
# expr/ builds the symbolic layer on it, dual/ the numeric scalars, drivers/ the
# things that actually differentiate, and md/ the storage views they hand back.
#
# Each subdirectory adds its own headers to the `diff` INTERFACE target defined
# in the top-level CMakeLists.txt.
target_sources(diff INTERFACE ${CMAKE_CURRENT_LIST_DIR}/api.hpp)

add_subdirectory(util)
add_subdirectory(expr)
add_subdirectory(dual)
add_subdirectory(drivers)
add_subdirectory(md)
