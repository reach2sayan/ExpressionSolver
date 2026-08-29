target_sources(ddx_ops INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/numeric.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/algebra.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/scalar.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/mode.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/unary_math.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/operations.hpp>
)
