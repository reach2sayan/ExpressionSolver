# Consuming ddx by add_subdirectory() stays the documented route; this is what
# makes a built ddx::jit installable,
#Included from the top level only once every target that can be exported exists.
include_guard(GLOBAL)

include(CMakePackageConfigHelpers)
# The Boost directories to ship; include() without OPTIONAL, so a missing list
# stops the build rather than quietly installing nothing.
include(DdxBoostHeaders)

# ddx_rt is libddx itself.  The JIT objects are inside it when built, so there
# is nothing separate to export for them -- ddx::jit aliases this same target.
set(DDX_EXPORT_TARGETS ddx ddx_util ddx_ops ddx_md ddx_symbolic ddx_dual ddx_rt)

install(TARGETS ${DDX_EXPORT_TARGETS}
        EXPORT ddxTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")

# Boost travels inside the prefix, so a consumer needs none of their own: the
# build never consults the machine's Boost and neither should what it installs.
#
# Only what the installed headers reach, as whole directories, listed by
# scripts/boost_headers.py in DdxBoostHeaders.cmake: 32 MB against 190 MB for
# the tree.  Directories rather than the reachable files because the files
# differ by toolchain -- config/compiler/gcc.hpp against clang.hpp -- and every
# such variant sits beside a header the closure already names.  config, mpl and
# preprocessor arrive entire, every variant of each: those three hide the
# toolchain's copy in a subdirectory a Linux closure never enters.
#
# Only the fetched one.  A DDX_BOOST_INCLUDEDIR build is a caller who keeps
# their own Boost, and copying it here would leave two of it -- theirs and a
# snapshot of theirs, ageing apart.  ddx-config points at their path instead,
# which is the same bargain the build struck: name a Boost and you own it.
if (NOT DDX_BOOST_INCLUDEDIR)
    # install(DIRECTORY) recurses, and `boost` itself is on the list -- 165 files
    # one level deep against 16413 below it -- so the directories are globbed and
    # installed as files instead.  LIST_DIRECTORIES false with a bare `*` rather
    # than *.hpp: the set carries three .h and two .ipp as well, and everything
    # that is a file cannot miss an extension Boost adds later.
    foreach (dir IN LISTS DDX_BOOST_HEADER_DIRS)
        file(GLOB headers LIST_DIRECTORIES false "${DDX_BOOST_ROOT}/${dir}/*")
        if (NOT headers)
            message(FATAL_ERROR
                    "DdxBoostHeaders names ${dir}, which holds no files under "
                    "${DDX_BOOST_ROOT}.  The list and the tree disagree; "
                    "regenerate it with `python3 scripts/boost_headers.py`.")
        endif ()
        install(FILES ${headers} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${dir}")
    endforeach ()
endif ()

install(EXPORT ddxTargets
        NAMESPACE ddx::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ddx)

# Two configs from one template, because the two trees hold Boost in different
# places.  One file serving both would have to ask the filesystem which tree it
# is in; each of these states its answer instead.
#
# The build tree names the Boost this build used.  PACKAGE_PREFIX_DIR is written
# through escaped, so it reaches the installed file as text and expands when
# that file runs -- a prefix that moves is still right.
set(DDX_BOOST_CONFIG_ROOT "${DDX_BOOST_ROOT}")
configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/ddx-config.cmake.in"
        "${PROJECT_BINARY_DIR}/ddx-config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ddx)

# The installed one, which finds Boost inside the prefix -- unless the caller
# supplied their own, in which case nothing was copied and their root stands.
if (NOT DDX_BOOST_INCLUDEDIR)
    set(DDX_BOOST_CONFIG_ROOT "\${PACKAGE_PREFIX_DIR}/${CMAKE_INSTALL_INCLUDEDIR}")
endif ()
configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/ddx-config.cmake.in"
        "${PROJECT_BINARY_DIR}/install/ddx-config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ddx)
# SameMajorVersion: ddx::jit carries an SOVERSION of the major alone.
write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/ddx-config-version.cmake"
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)
install(FILES
        "${PROJECT_BINARY_DIR}/install/ddx-config.cmake"
        "${PROJECT_BINARY_DIR}/ddx-config-version.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ddx)

export(EXPORT ddxTargets
        NAMESPACE ddx::
        FILE "${PROJECT_BINARY_DIR}/ddxTargets.cmake")
