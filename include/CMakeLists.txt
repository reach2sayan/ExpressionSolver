# Header lists for the IDE; nothing is compiled from here.
target_sources(ddx INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/ddx.hpp>)

add_subdirectory(util)
add_subdirectory(ops)
add_subdirectory(symbolic)
add_subdirectory(dual)
add_subdirectory(md)
