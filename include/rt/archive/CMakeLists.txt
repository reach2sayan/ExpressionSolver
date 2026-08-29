target_sources(ddx_rt INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/snapshot.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/codec.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/container.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/digest.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/sound.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/archive.hpp>
)
