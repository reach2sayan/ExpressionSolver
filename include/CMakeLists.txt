# Header directories, in dependency order.  util/ knows nothing about the rest;
# ops/ is the operation vocabulary the compile-time graph, the runtime graph
# and the JIT all share; md/ is the storage the drivers hand back; symbolic/ is
# the compile-time graph; dual/ is forward mode, and the only optional one.
#
# Each subdirectory adds its own headers to the `ddx` INTERFACE target defined
# in the top-level CMakeLists.txt.  Nothing is compiled from here -- the lists
# exist so an IDE resolves include paths for every header, so each one has to
# be named.
target_sources(ddx INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/ddx.hpp>)

add_subdirectory(util)
add_subdirectory(ops)
add_subdirectory(symbolic)
if (DDX_BUILD_DUAL)
    add_subdirectory(dual)
endif ()
add_subdirectory(md)
