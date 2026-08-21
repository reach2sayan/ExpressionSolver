# md/ — the multidimensional views derivatives are handed back in: the mdspan
# binding (std:: or vendored, see DDX_MDSPAN_MODE), the packed symmetric
# layouts, and the tensor types built on them.
target_sources(ddx INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/md.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/layouts.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/tensor.hpp>
        # Vendored reference mdspan — used only when the toolchain has no
        # <mdspan> of its own.  Not ours; left unformatted and unedited.
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/third_party/mdspan.hpp>
)
