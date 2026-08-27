# The archive, split by concern: what a built equation is, how a value becomes
# bytes, what a ddx file looks like, the two graph keys, and the three entry
# points.  Listed like every other header directory -- an IDE resolves include
# paths from the target naming a header, and one left off gets none.
target_sources(ddx_rt INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/snapshot.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/codec.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/container.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/digest.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/sound.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/archive.hpp>
)
