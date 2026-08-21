# The runtime expression graph.  INTERFACE like ddx::ddx -- the graph itself is
# headers under include/rt -- with one compiled object in the no-exceptions
# configuration below.
add_library(ddx_rt INTERFACE)
add_library(ddx::rt ALIAS ddx_rt)
# EXPORT_NAME so an installed ddx spells its targets the way an in-tree one
# does: ddx::rt, not ddx::ddx_rt.  Same for ddx::jit.
set_target_properties(ddx_rt PROPERTIES EXPORT_NAME rt)
# Boost is fetched for the build and found again at install time -- the fetched
# targets belong to no export set of ours, so the installed interface names them
# for ddx-config.cmake to resolve instead.
target_link_libraries(ddx_rt INTERFACE ddx
        $<BUILD_INTERFACE:Boost::graph>
        $<BUILD_INTERFACE:Boost::dynamic_bitset>)

# Boost leaves boost::throw_exception undefined under -fno-exceptions and makes
# it the program's to supply; see boost_no_exceptions.cpp.  ddx_rt is INTERFACE,
# so this is the one compiled object it carries, and only in this configuration.
if (DDX_NO_EXCEPTIONS)
    add_library(ddx_rt_no_exceptions STATIC boost_no_exceptions.cpp)
    # ddx_jit links this through ddx_rt and may itself be a shared library,
    # which cannot absorb a non-PIC archive.
    set_target_properties(ddx_rt_no_exceptions PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            EXPORT_NAME rt_no_exceptions)
    # BUILD_INTERFACE for the same reason as ddx_rt above: PRIVATE on a static
    # library still lands in the exported interface as $<LINK_ONLY:>, and the
    # fetched Boost target is not ours to export.  Nothing is lost -- the header
    # is needed to compile this object, not to link against it.
    target_link_libraries(ddx_rt_no_exceptions PRIVATE $<BUILD_INTERFACE:Boost::throw_exception>)
    ddx_target_flags(ddx_rt_no_exceptions)
    target_link_libraries(ddx_rt INTERFACE ddx_rt_no_exceptions)
endif ()
