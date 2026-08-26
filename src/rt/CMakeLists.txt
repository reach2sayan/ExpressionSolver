# The runtime graph is almost all headers.  Compiled here is the part carrying
# no scalar at all -- the colouring, the graphviz writer, the archive prologue --
# which would otherwise be emitted once per scalar for byte-identical code, and
# would put Boost.Graph on the include path of every TU that only evaluates.
add_library(ddx_rt ${DDX_LIBRARY_TYPE} coupling.cpp dot.cpp archive.cpp)
add_library(ddx::rt ALIAS ddx_rt)

# One binary artifact, libddx, and EXPORT_NAME so an installed ddx spells its
# targets as an in-tree one does.  Hidden by default, so what leaves libddx is
# what DDX_API marks and not Boost.Graph's instantiations.
set_target_properties(ddx_rt PROPERTIES
        OUTPUT_NAME ddx
        EXPORT_NAME rt
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})

# DDX_BUILDING only while this library compiles -- see include/util/export.hpp.
target_compile_definitions(ddx_rt PRIVATE DDX_BUILDING)
if (NOT DDX_SHARED_LIBS)
    # A consumer's declarations have to agree about whether there is a boundary.
    target_compile_definitions(ddx_rt PUBLIC DDX_STATIC_LIB)
endif ()

# PUBLIC because Boost is named in include/rt, so a consumer of these headers
# needs the include path.  Header-only throughout, so nothing links a Boost
# library, and the target names a found package the export set carries as-is.
target_link_libraries(ddx_rt PUBLIC ddx Boost::headers Threads::Threads)
ddx_target_flags(ddx_rt)

# Under -fno-exceptions Boost leaves boost::throw_exception for the program to
# supply.  This tree always builds that way, so it is conditional on nothing.
target_sources(ddx_rt PRIVATE boost_no_exceptions.cpp)
