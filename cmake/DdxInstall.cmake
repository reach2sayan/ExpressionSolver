# Consuming ddx by add_subdirectory() stays the documented route; this is what
# makes a built ddx::jit installable,
#Included from the top level only once every target that can be exported exists.
include_guard(GLOBAL)

include(CMakePackageConfigHelpers)

set(DDX_EXPORT_TARGETS ddx ddx_util ddx_ops ddx_md ddx_symbolic)
if (DDX_BUILD_DUAL)
    list(APPEND DDX_EXPORT_TARGETS ddx_dual)
endif ()
# libddx itself.  The JIT objects are inside it when built, so there is nothing
# separate to export for them -- ddx::jit is an alias of this same target.
list(APPEND DDX_EXPORT_TARGETS ddx_rt)

install(TARGETS ${DDX_EXPORT_TARGETS}
        EXPORT ddxTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")
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
