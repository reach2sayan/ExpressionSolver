# jit/ — the two headers a consumer of the JIT sees.  Named on ddx_rt, which is
# what ddx::jit aliases: the objects in src/jit are linked into libddx rather
# than into a library of their own, so there is no separate target to hang them
# on.
#
# Added only under DDX_BUILD_JIT, and after src/jit, because ddx_rt only carries
# DDX_HAS_JIT from that point on -- an IDE resolving format.hpp without it would
# see the jit::Ir formatter compiled out.
#
# Neither header names an LLVM type, which is why they can sit on a target that
# does not carry LLVM's include path.
target_sources(ddx_rt INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/kernel.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/format.hpp>
)
