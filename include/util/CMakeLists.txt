# util/ — language configuration, version macros, compile-time strings, and the
# RAII seed toggle every driver borrows.
target_sources(ddx INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/config.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/version.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/fixed_string.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/pinned.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/scope_guard.hpp>
)
