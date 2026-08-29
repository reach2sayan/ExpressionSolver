target_sources(ddx_rt INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/ast.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/lower.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/equation.hpp>
)
