# The extension module: a consumer of libddx like compare/, living in this tree
# so one `uv pip install .` builds the pair.  LLVM is inside libddx and Boost is
# headers, so with DDX_INSTALL off the wheel is the two libraries below and
# nothing else.
#
#   ddx/_ddx.cpython-*.so   these bindings
#   ddx/libddx.so.1         the library they link

pybind11_add_module(ddx_python src/module.cpp)
set_target_properties(ddx_python PROPERTIES OUTPUT_NAME _ddx)
target_link_libraries(ddx_python PRIVATE ddx::rt)
# Only for the LLJIT itself: jit::Options and jit::Kernel are header types, so
# the module presents one surface either way.
if (DDX_BUILD_JIT)
    target_link_libraries(ddx_python PRIVATE ddx::jit)
endif ()

# pybind11 translates a C++ exception into a Python one, so this module keeps them.
ddx_target_flags(ddx_python EXCEPTIONS)

ddx_runtime_deps(ddx_python)

# $ORIGIN -- @loader_path on macOS -- is how _ddx finds libddx once the wheel
# is unpacked somewhere CMake never saw.
set_target_properties(ddx_python PROPERTIES
        INSTALL_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path,$ORIGIN>")
install(TARGETS ddx_python LIBRARY DESTINATION ddx)
# RUNTIME for the Windows DLL, which is not a LIBRARY there; its import
# library is nothing a wheel loads, so it stays out.
install(TARGETS ddx_rt
        LIBRARY DESTINATION ddx NAMELINK_SKIP
        RUNTIME DESTINATION ddx
        ARCHIVE DESTINATION ddx EXCLUDE_FROM_ALL)
# A wheel holds no symlinks, so the soname link would be a second copy of a
# 40 MB library; the real file takes the one name _ddx asks for instead.
if (NOT WIN32)
    install(CODE "file(RENAME
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/ddx/$<TARGET_FILE_NAME:ddx_rt>\"
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/ddx/$<TARGET_SONAME_FILE_NAME:ddx_rt>\")")
endif ()

# --- the package, in the build tree ------------------------------------------
# The wheel gets this layout from install(); a preset build has to be given it,
# so that pytest imports the same `ddx` a user would rather than a loose .so.
set(DDX_PYTHON_PACKAGE "${CMAKE_CURRENT_BINARY_DIR}/ddx")
set_target_properties(ddx_python PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${DDX_PYTHON_PACKAGE}")

file(GLOB DDX_PYTHON_SOURCES CONFIGURE_DEPENDS "ddx/*.py" "ddx/py.typed" "ddx/*.pyi")
add_custom_target(ddx_python_package ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${DDX_PYTHON_PACKAGE}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DDX_PYTHON_SOURCES} "${DDX_PYTHON_PACKAGE}"
        DEPENDS ${DDX_PYTHON_SOURCES}
        COMMENT "Staging the ddx package beside _ddx")
add_dependencies(ddx_python_package ddx_python)

# --- pytest, as a ctest -------------------------------------------------------
# One test, not one per file: pytest is the runner.  The suite reads ddx.Options
# to decide what a build without the backend should do, so the same tests run
# under both python presets with no skip list.
add_test(NAME python.pytest
        COMMAND "${Python_EXECUTABLE}" -m pytest -q
        "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set_tests_properties(python.pytest PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};PYTHONDONTWRITEBYTECODE=1;DDX_PACKAGE_DIR=${CMAKE_CURRENT_BINARY_DIR}")

# --- stubs --------------------------------------------------------------------
# On request, not every build: the result is checked in, a type checker having to
# see it without configuring CMake.  Against the *bare* extension, since
# importing the package runs __init__.py and attaches `options` and `compile`,
# which belong to the hand-written layer's types.
add_custom_target(ddx_python_stubs
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${DDX_PYTHON_PACKAGE}"
        "${Python_EXECUTABLE}" -m pybind11_stubgen _ddx
        -o "${CMAKE_CURRENT_BINARY_DIR}/stubs"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_BINARY_DIR}/stubs/_ddx.pyi"
        "${CMAKE_CURRENT_SOURCE_DIR}/ddx/_ddx.pyi"
        DEPENDS ddx_python
        COMMENT "Regenerating python/ddx/_ddx.pyi")
