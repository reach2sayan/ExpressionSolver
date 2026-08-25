# md/ — the multidimensional views derivatives are handed back in: the mdspan
# binding (std:: or vendored, see DDX_MDSPAN_MODE), the packed symmetric
# layouts, the tensor types built on them, and the scratch the sweeps run in.
target_sources(ddx_md INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/md.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/layouts.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/workspace.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/tensor.hpp>
        # Vendored reference mdspan — used only when the toolchain has no
        # <mdspan> of its own.  kokkos/mdspan, single-header branch, commit
        # c3fc07b6 (2026-07-26), run through our .clang-format.  The one
        # change to what it says is a guard around the at() overloads, which
        # throw (search LOCAL PATCH); everything else is upstream token for
        # token, which scripts/mdspan_vendor_diff.py re-checks.
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/third_party/mdspan.hpp>
)
