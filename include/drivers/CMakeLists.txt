# drivers/ — everything that actually computes a derivative, split by what it is
# handed: `numeric` differentiates an opaque runtime callable, `symbolic` an
# expression graph it can see inside and sweep backward.  `hessian` is the
# router that chooses between them, `seeded_energy` the bridge that lets a graph
# reach the numeric side, `coupling` the compile-time sparsity analysis that
# decides how few backward sweeps a graph Hessian needs.  `common` is the
# vocabulary the two driver files share so they need not know about each other.
target_sources(ddx INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/common.hpp
        ${CMAKE_CURRENT_LIST_DIR}/numeric.hpp
        ${CMAKE_CURRENT_LIST_DIR}/symbolic.hpp
        ${CMAKE_CURRENT_LIST_DIR}/hessian.hpp
        ${CMAKE_CURRENT_LIST_DIR}/seeded_energy.hpp
        ${CMAKE_CURRENT_LIST_DIR}/coupling.hpp
)
