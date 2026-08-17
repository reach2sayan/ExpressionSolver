# expr/ — the symbolic layer: expression nodes and their concepts, the operator
# set, the leaves that carry values, the label-keyed binding of a point, and the
# vector-valued Equation wrapper.  All of it is constexpr-evaluable and free of
# third-party dependencies.
target_sources(diff INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/expressions.hpp
        ${CMAKE_CURRENT_LIST_DIR}/operations.hpp
        ${CMAKE_CURRENT_LIST_DIR}/unary_math.hpp
        ${CMAKE_CURRENT_LIST_DIR}/values.hpp
        ${CMAKE_CURRENT_LIST_DIR}/named_value.hpp
        ${CMAKE_CURRENT_LIST_DIR}/bound.hpp
        ${CMAKE_CURRENT_LIST_DIR}/equation.hpp
        ${CMAKE_CURRENT_LIST_DIR}/traits.hpp
)
