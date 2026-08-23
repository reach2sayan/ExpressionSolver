# dual/ — forward mode: the dual scalar, its operators, and the drivers that
# only it can answer.  ops/scalar.hpp forward-declares Dual and gives every
# trait a non-dual answer, which is what keeps ops/ and symbolic/ free of any
# dependency on this directory.
target_sources(ddx_dual INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/dual.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/taylor_dual.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/workspace.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/numeric.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/hessian.hpp>
)
