# The only LLVM-aware code in the project.  include/jit/kernel.hpp names no LLVM
# type, so nothing here escapes into a consumer's translation unit.
#
# An OBJECT library rather than a second shared library: ddx produces one binary,
# libddx, and these objects are linked into it.  The indirection buys the one
# thing a plain target_sources() on ddx_rt could not -- these files need
# their own private include path (LLVM's), which should not reach the rest of
# the library.
add_library(ddx_jit_objs OBJECT codegen.cpp compiler.cpp partition.cpp)

# PIC unconditionally: required when libddx is shared, and it costs the static
# build nothing measurable on x86-64.  Hidden visibility so what leaves libddx
# is the list DDX_JIT_API marks and not whatever LLVM left visible.
set_target_properties(ddx_jit_objs PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

# DDX_JIT_BUILDING only while these objects compile -- see include/jit/export.hpp.
target_compile_definitions(ddx_jit_objs PRIVATE DDX_BUILDING ${LLVM_DEFINITIONS_LIST})
target_include_directories(ddx_jit_objs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ddx_jit_objs SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
# The same header dependencies ddx_rt has, not ddx_rt itself: these objects are
# linked *into* it, so naming it here would be circular.
target_link_libraries(ddx_jit_objs PRIVATE ddx Boost::headers)

# Deliberately no -fno-rtti, even against an LLVM built without it.  The flag is
# only needed by code that derives from an LLVM polymorphic class or applies
# typeid/dynamic_cast to one; isa<>/cast<> use LLVM's own RTTI, not the
# language's, and nothing here inherits from LLVM.
#
# Exceptions are off through ddx_target_flags() like everywhere else, which
# suits this file particularly: it is the one place the project meets LLVM, LLVM
# is itself built -fno-exceptions -funwind-tables (llvm-config --cxxflags), and
# nothing here throws -- llvm::Error comes out as jit::error instead.
ddx_target_flags(ddx_jit_objs)

# Into libddx, and ddx::jit is that same library under the name a consumer asks
# for the backend by.
target_sources(ddx_rt PRIVATE $<TARGET_OBJECTS:ddx_jit_objs>)
add_library(ddx::jit ALIAS ddx_rt)

# PUBLIC on ddx_rt: DDX_HAS_JIT unlocks jit/format.hpp's formatter for jit::Ir
# and rt/equation.hpp's compiled batch calls, and the two defaults are read by
# Options in a consumer's own translation unit.
target_compile_definitions(ddx_rt PUBLIC
        DDX_HAS_JIT
        DDX_JIT_DEFAULT_OPT=${DDX_JIT_DEFAULT_OPT}
        DDX_JIT_DEFAULT_CONTRACT=${DDX_JIT_DEFAULT_CONTRACT})
# The dylib where LLVM ships one, which is the difference between a library of a
# few hundred KB and one that has swallowed LLVM's codegen archives whole.
#
# The static fallback has to hide what it swallowed.  A process that already has
# LLVM loaded -- Mesa's llvmpipe and radeonsi both do -- would otherwise end up
# with two copies of LLVM's ManagedStatic and command-line registries, which is
# a crash rather than a diagnostic.
if (DDX_SHARED_LIBS AND LLVM_LINK_LLVM_DYLIB)
    target_link_libraries(ddx_rt PRIVATE LLVM)
else ()
    target_link_libraries(ddx_rt PRIVATE ${DDX_LLVM_LIBS})
    if (DDX_SHARED_LIBS AND NOT MSVC)
        target_link_options(ddx_rt PRIVATE -Wl,--exclude-libs,ALL)
    endif ()
endif ()
