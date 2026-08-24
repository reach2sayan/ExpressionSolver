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

# PUBLIC because Boost is named in include/rt: the frozen graph is a
# compressed_sparse_row_graph and the coupling rows are dynamic_bitsets, so a
# consumer of these headers needs the include path too.  What moved into this
# library is Boost.Graph's *algorithms* -- the colouring and the graphviz writer
# -- which nothing outside these two .cpp files includes now.  All of it is
# header-only, so none of it is a link-time dependency for anyone, and the
# target names a found package, so the export set carries it as it stands.
target_link_libraries(ddx_rt PUBLIC ddx Boost::headers Threads::Threads)
ddx_target_flags(ddx_rt)

# Boost leaves boost::throw_exception undefined under -fno-exceptions and makes
# it the program's to supply.  The project builds that way always, so this is
# conditional on nothing.
target_sources(ddx_rt PRIVATE boost_no_exceptions.cpp)
