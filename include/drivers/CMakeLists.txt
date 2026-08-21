# drivers/ — everything that actually computes a derivative,
# (a) `numeric` differentiates an opaque runtime callable,
# (b) `symbolic` an expression graph it can see inside and sweep backward.
# (c) `hessian` is the router that chooses between them,
# (d) `seeded_energy` the bridge that lets a graph reach the numeric side,
# (e) `coupling` the compile-time sparsity analysis that decides how few backward sweeps a graph Hessian needs.
# (d) `common` is the vocabulary the two driver files share so they need not know about each other.
target_sources(ddx INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/common.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/numeric.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/symbolic.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/hessian.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/seeded_energy.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/coupling.hpp>
)
