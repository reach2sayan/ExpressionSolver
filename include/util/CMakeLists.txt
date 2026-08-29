target_sources(ddx_util INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/config.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/fixed_string.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/pinned.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/scope_guard.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/ranges.hpp>
)
