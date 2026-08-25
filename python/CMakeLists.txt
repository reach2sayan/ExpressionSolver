# The extension module.  A consumer of libddx like compare/ is, except that it
# lives in this tree so that one `uv pip install .` builds the pair.
#
#   ddx/_ddx.cpython-*.so   these bindings
#   ddx/libddx.so.1         the library they link
#
# LLVM and Boost are the machine's, exactly as for a C++ build; nothing is
# vendored into the wheel.

pybind11_add_module(ddx_python src/module.cpp)
set_target_properties(ddx_python PROPERTIES OUTPUT_NAME _ddx)
target_link_libraries(ddx_python PRIVATE ddx::rt)
# Only for the LLJIT itself.  jit::Options and jit::Kernel are header types, so
# the module presents one surface either way: without the backend nothing
# compiles, no kernel lands, and the sweep answers everything.
if (DDX_BUILD_JIT)
    target_link_libraries(ddx_python PRIVATE ddx::jit)
endif ()

# The flags ddx recorded, less the one that cannot come across -- the same trade
# compare/ makes.  pybind11 translates C++ exceptions into Python ones, so the
# module needs exception support; -march, -ffp-contract and -fno-math-errno are
# kept, because those change arithmetic and the kernel is emitted to match.
set(DDX_PYTHON_FLAGS ${DDX_CODEGEN_FLAGS})
list(REMOVE_ITEM DDX_PYTHON_FLAGS -fno-exceptions /EHs-c- /D_HAS_EXCEPTIONS=0)
target_compile_options(ddx_python PRIVATE ${DDX_PYTHON_FLAGS} ${DDX_WARNINGS})

ddx_runtime_deps(ddx_python)

# Both objects land beside __init__.py, so the package is self-contained but for
# LLVM and Boost.  $ORIGIN is how _ddx finds libddx once the wheel is unpacked
# somewhere CMake never saw; USE_LINK_PATH is how libddx keeps the LLVM runpath
# it was linked with, which install() would otherwise strip -- LLVM is not on
# the default loader path on any machine that has more than one.
set_target_properties(ddx_python PROPERTIES INSTALL_RPATH "$ORIGIN")
# And libddx has to keep finding LLVM.  CMake's build-tree RUNPATH is stripped on
# install, and ddx_rt is created in src/rt before this directory is read, so the
# property is set on it here rather than through CMAKE_INSTALL_RPATH_USE_LINK_PATH.
# Easy to miss: this machine also ships libLLVM.so.20.1 on the default loader
# path, so the wheel loads without it and only fails somewhere else.
if (DDX_BUILD_JIT)
    set_property(TARGET ddx_rt APPEND PROPERTY INSTALL_RPATH "${LLVM_LIBRARY_DIR}")
endif ()
install(TARGETS ddx_python LIBRARY DESTINATION ddx)
install(TARGETS ddx_rt LIBRARY DESTINATION ddx)

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
# One test rather than one per file: pytest is the runner, and ctest asking it
# to collect twice would just be slower.  The suite reads ddx.Options to decide
# what to expect of a build without the backend, so the same tests run under
# both python presets and neither needs a list of what to skip.
add_test(NAME python.pytest
        COMMAND "${Python_EXECUTABLE}" -m pytest -q
        "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set_tests_properties(python.pytest PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};PYTHONDONTWRITEBYTECODE=1;DDX_PACKAGE_DIR=${CMAKE_CURRENT_BINARY_DIR}")

# --- stubs --------------------------------------------------------------------
# Regenerated on request, not every build: the result is checked in, because a
# type checker has to see it without configuring CMake first.  Generated against
# the *bare* extension rather than `ddx._ddx` -- importing the package would run
# __init__.py, which attaches the pydantic-facing `options` and `compile` to the
# bound class, and those belong in the hand-written layer's types, not in the
# description of the C++ surface.
add_custom_target(ddx_python_stubs
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${DDX_PYTHON_PACKAGE}"
        "${Python_EXECUTABLE}" -m pybind11_stubgen _ddx
        -o "${CMAKE_CURRENT_BINARY_DIR}/stubs"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_BINARY_DIR}/stubs/_ddx.pyi"
        "${CMAKE_CURRENT_SOURCE_DIR}/ddx/_ddx.pyi"
        DEPENDS ddx_python
        COMMENT "Regenerating python/ddx/_ddx.pyi")
