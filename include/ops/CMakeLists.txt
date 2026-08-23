# ops/ — the operation vocabulary, and the only thing the runtime graph and the
# JIT share with the compile-time one.  numeric.hpp is the leaf every layer is
# written against; operations.hpp and unary_math.hpp are the single source of
# the op set and its derivative rules.
target_sources(ddx_ops INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/numeric.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/scalar.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/mode.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/unary_math.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/operations.hpp>
)
