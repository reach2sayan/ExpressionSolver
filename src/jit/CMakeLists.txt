# The only LLVM-aware code in the project.  include/jit/kernel.hpp names no
# LLVM type, so nothing here escapes into a consumer's translation unit.
# Shared unless asked otherwise: LLVM is the bulk of this library, and a static
# ddx::jit puts that bulk into every consumer that links it.
if (DDX_JIT_STATIC)
    set(DDX_JIT_LIBRARY_TYPE STATIC)
else ()
    set(DDX_JIT_LIBRARY_TYPE SHARED)
endif ()

add_library(ddx_jit ${DDX_JIT_LIBRARY_TYPE}
        codegen.cpp
        compiler.cpp
)
add_library(ddx::jit ALIAS ddx_jit)

# Hidden by default, so what leaves the library is the list DDX_JIT_API marks
# and not whatever LLVM happens to have left visible.  PIC unconditionally: it
# is required of the shared build and costs the static one nothing measurable on
# x86-64.
set_target_properties(ddx_jit PROPERTIES
        EXPORT_NAME jit
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})

# DDX_JIT_BUILDING only while this library compiles -- see include/jit/export.hpp.
# DDX_JIT_STATIC is PUBLIC because a consumer's declaration of Compiler has to
# agree with this one about whether there is a boundary at all.
target_compile_definitions(ddx_jit PRIVATE DDX_JIT_BUILDING)
if (DDX_JIT_STATIC)
    target_compile_definitions(ddx_jit PUBLIC DDX_JIT_STATIC)
endif ()

target_include_directories(ddx_jit
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ddx_jit SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(ddx_jit PRIVATE ${LLVM_DEFINITIONS_LIST})
target_link_libraries(ddx_jit PUBLIC ddx_rt)

# The dylib where LLVM ships one, which is the difference between a library of a
# few hundred KB and one that has swallowed LLVM's codegen archives whole.
#
# The static fallback has to hide what it swallowed.  A process that already has
# LLVM loaded -- Mesa's llvmpipe and radeonsi both do -- would otherwise end up
# with two copies of LLVM's ManagedStatic and command-line registries, which is
# a crash rather than a diagnostic.
if (NOT DDX_JIT_STATIC AND LLVM_LINK_LLVM_DYLIB)
    target_link_libraries(ddx_jit PRIVATE LLVM)
else ()
    target_link_libraries(ddx_jit PRIVATE ${DDX_LLVM_LIBS})
    if (NOT DDX_JIT_STATIC AND NOT MSVC)
        target_link_options(ddx_jit PRIVATE -Wl,--exclude-libs,ALL)
    endif ()
endif ()
# PUBLIC: DDX_HAS_JIT unlocks the std::formatter for jit::Ir in expr/format.hpp,
# and the two defaults are read by Options in a consumer's own translation unit.
target_compile_definitions(ddx_jit PUBLIC
        DDX_HAS_JIT
        DDX_JIT_DEFAULT_OPT=${DDX_JIT_DEFAULT_OPT}
        DDX_JIT_DEFAULT_CONTRACT=${DDX_JIT_DEFAULT_CONTRACT})

# Deliberately no -fno-rtti, even against an LLVM built without it.  The flag is
# only needed by code that derives from an LLVM polymorphic class or applies
# typeid/dynamic_cast to one; isa<>/cast<> use LLVM's own RTTI, not the
# language's, and nothing here inherits from LLVM.
#
# -fno-exceptions, on the other hand, unconditionally: this is the one place the
# project meets LLVM, LLVM is itself built -fno-exceptions -funwind-tables
# (llvm-config --cxxflags), and nothing here throws -- llvm::Error comes out as
# jit::error instead.  Independent of DDX_NO_EXCEPTIONS, which is about the rest
# of the tree.
ddx_target_flags(ddx_jit)
if (NOT MSVC)
    target_compile_options(ddx_jit PRIVATE -fno-exceptions -funwind-tables)
else ()
    target_compile_options(ddx_jit PRIVATE /EHs-c-)
endif ()
