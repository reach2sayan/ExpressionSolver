# expr/ — the symbolic layer
target_sources(ddx INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/symbol.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/expressions.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/operations.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/unary_math.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/values.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/named_value.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/named_map.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/bound.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/equation.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/traits.hpp>
)
