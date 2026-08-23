# dual/ — forward mode: the dual scalar, its operators, and the drivers that
# only it can answer.  Optional (DDX_BUILD_DUAL, on by default); ops/scalar.hpp
# forward-declares Dual so everything above compiles whether or not this is
# built, and the second-derivative entry points constrain themselves away.
target_sources(ddx_dual INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/dual.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/taylor_dual.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/workspace.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/numeric.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/hessian.hpp>
)
