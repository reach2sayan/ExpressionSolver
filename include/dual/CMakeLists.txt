target_sources(ddx_dual INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/dual.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/taylor_dual.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/workspace.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/numeric.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/hessian.hpp>
)
