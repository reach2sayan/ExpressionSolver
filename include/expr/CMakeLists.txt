# expr/ — the symbolic layer
target_sources(ddx INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/expressions.hpp
        ${CMAKE_CURRENT_LIST_DIR}/operations.hpp
        ${CMAKE_CURRENT_LIST_DIR}/unary_math.hpp
        ${CMAKE_CURRENT_LIST_DIR}/values.hpp
        ${CMAKE_CURRENT_LIST_DIR}/named_value.hpp
        ${CMAKE_CURRENT_LIST_DIR}/named_map.hpp
        ${CMAKE_CURRENT_LIST_DIR}/bound.hpp
        ${CMAKE_CURRENT_LIST_DIR}/equation.hpp
        ${CMAKE_CURRENT_LIST_DIR}/traits.hpp
)
