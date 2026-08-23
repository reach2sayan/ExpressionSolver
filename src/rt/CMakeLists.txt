# The runtime expression graph.  Almost all of it is headers under include/rt --
# the builder, the interpreter and the derivative sweeps are templates over the
# scalar, and stay constexpr up to the freeze.  What is compiled here is the
# part that carries no scalar at all:
#
#   coupling.cpp  the Hessian column colouring -- Boost.Graph's conflict graph,
#                 smallest-last ordering and sequential colouring
#   dot.cpp       the graphviz writer, which drags in the DOT reader with it
#
# Both would otherwise be emitted once per scalar type the graph is instantiated
# at, for byte-identical code, and would put Boost.Graph's algorithms on the
# include path of every translation unit that only wants to evaluate.
add_library(ddx_rt ${DDX_LIBRARY_TYPE} coupling.cpp dot.cpp)
add_library(ddx::rt ALIAS ddx_rt)

# OUTPUT_NAME ddx: one binary artifact, libddx, whatever is compiled into it.
# EXPORT_NAME so an installed ddx spells its targets the way an in-tree one
# does: ddx::rt, not ddx::ddx_rt.
#
# Hidden by default, so what leaves libddx is the list DDX_API marks and not the
# several thousand Boost.Graph template instantiations that would otherwise be
# exported from it.
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
    # A consumer's declarations have to agree with these about whether there is
    # a boundary at all.
    target_compile_definitions(ddx_rt PUBLIC DDX_STATIC_LIB)
endif ()

# Boost is fetched for the build and found again by the consumer -- the fetched
# targets belong to no export set of ours.  BUILD_LOCAL_INTERFACE, not
# BUILD_INTERFACE: both keep them out of an install, but only this one also
# keeps them out of export(), which retains BUILD_INTERFACE because the build
# tree is what it exports.  ddx-config.cmake resolves Boost for either tree.
#
# Both are PUBLIC because both are named in include/rt: the frozen graph is a
# compressed_sparse_row_graph and the coupling rows are dynamic_bitsets.  What
# moved into this library is Boost.Graph's *algorithms* -- the colouring and the
# graphviz writer -- which nothing outside these two .cpp files includes now.
# All of it is header-only, so none of it is a link-time dependency for anyone.
target_link_libraries(ddx_rt PUBLIC ddx
        $<BUILD_LOCAL_INTERFACE:Boost::graph>
        $<BUILD_LOCAL_INTERFACE:Boost::dynamic_bitset>)
ddx_target_flags(ddx_rt)

# Boost leaves boost::throw_exception undefined under -fno-exceptions and makes
# it the program's to supply; see boost_no_exceptions.cpp.  It compiles straight
# into the library now that there is one.
if (DDX_NO_EXCEPTIONS)
    target_sources(ddx_rt PRIVATE boost_no_exceptions.cpp)
    target_link_libraries(ddx_rt PRIVATE $<BUILD_LOCAL_INTERFACE:Boost::throw_exception>)
endif ()
