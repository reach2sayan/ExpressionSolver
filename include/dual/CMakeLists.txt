# dual/ — the numeric AD scalars the drivers propagate
target_sources(ddx INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/dual.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/taylor_dual.hpp>
)
