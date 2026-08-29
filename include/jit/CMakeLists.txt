# After src/jit: ddx_rt carries DDX_HAS_JIT only from there on.
target_sources(ddx_rt INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/kernel.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/format.hpp>
)
