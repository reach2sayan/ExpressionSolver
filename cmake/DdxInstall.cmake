# Consuming ddx by add_subdirectory() stays the documented route; this is what
# makes a built ddx::jit installable,
#Included from the top level only once every target that can be exported exists.
include_guard(GLOBAL)

include(CMakePackageConfigHelpers)

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
# The whole tree, not the part ddx reaches.  A curated list would be a third of
# the bytes and would go stale the first time a header named a library nobody
# added to it -- and the closure is the compiler's anyway, so a list right here
# would be short elsewhere.
#
# Only the fetched one.  A DDX_BOOST_INCLUDEDIR build is a caller who keeps
# their own Boost, and copying it here would leave two of it -- theirs and a
# snapshot of theirs, ageing apart.  ddx-config points at their path instead,
# which is the same bargain the build struck: name a Boost and you own it.
if (NOT DDX_BOOST_INCLUDEDIR)
    install(DIRECTORY "${DDX_BOOST_ROOT}/boost"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif ()

install(EXPORT ddxTargets
        NAMESPACE ddx::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ddx)

configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/ddx-config.cmake.in"
        "${PROJECT_BINARY_DIR}/ddx-config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ddx)
# SameMajorVersion: ddx::jit carries an SOVERSION of the major alone.
write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/ddx-config-version.cmake"
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)
install(FILES
        "${PROJECT_BINARY_DIR}/ddx-config.cmake"
        "${PROJECT_BINARY_DIR}/ddx-config-version.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ddx)

export(EXPORT ddxTargets
        NAMESPACE ddx::
        FILE "${PROJECT_BINARY_DIR}/ddxTargets.cmake")
