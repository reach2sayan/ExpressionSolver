# The two headers a JIT consumer sees, named on ddx_rt because src/jit has no
# target of its own.  Added after src/jit: ddx_rt only carries DDX_HAS_JIT from
# there on, and an IDE resolving format.hpp without it sees the jit::Ir
# formatter compiled out.
target_sources(ddx_rt INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/kernel.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/format.hpp>
)
