# The only LLVM-aware code here; kernel.hpp names no LLVM type, so none escapes
# into a consumer's TU.  An OBJECT library, not a second shared one: these link
# into libddx, and the indirection is what gives them their own private include
# path, which a plain target_sources() on ddx_rt could not.
add_library(ddx_jit_objs OBJECT codegen.cpp compiler.cpp)

# PIC unconditionally: required when libddx is shared, free on x86-64 otherwise.
# Hidden, so what leaves libddx is DDX_JIT_API's list, not LLVM's.
set_target_properties(ddx_jit_objs PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

# DDX_JIT_BUILDING only while these objects compile -- see include/jit/export.hpp.
target_compile_definitions(ddx_jit_objs PRIVATE DDX_BUILDING ${LLVM_DEFINITIONS_LIST})
target_include_directories(ddx_jit_objs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ddx_jit_objs SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
# ddx_rt's header dependencies, not ddx_rt: these link *into* it.
target_link_libraries(ddx_jit_objs PRIVATE ddx Boost::headers)

# No -fno-rtti, even against an LLVM built without it: the flag matters only to
# code deriving from an LLVM polymorphic class, and isa<>/cast<> use LLVM's own
# RTTI.  Exceptions are off as everywhere -- LLVM is built the same way, and
# llvm::Error comes out as jit::error.
ddx_target_flags(ddx_jit_objs)

# Into libddx; ddx::jit is that library under the name the backend is asked for.
target_sources(ddx_rt PRIVATE $<TARGET_OBJECTS:ddx_jit_objs>)
add_library(ddx::jit ALIAS ddx_rt)

# PUBLIC: DDX_HAS_JIT unlocks the jit::Ir formatter and equation.hpp's compiled
# batch calls, and Options reads the two defaults in a consumer's own TU.
target_compile_definitions(ddx_rt PUBLIC
        DDX_HAS_JIT
        DDX_JIT_DEFAULT_OPT=${DDX_JIT_DEFAULT_OPT}
        DDX_JIT_DEFAULT_CONTRACT=${DDX_JIT_DEFAULT_CONTRACT})
# The dylib where LLVM ships one: the difference between a few hundred KB and
# having swallowed LLVM's codegen archives.  The static fallback has to hide what
# it swallowed -- a process that already has LLVM loaded (Mesa's llvmpipe does)
# would get two ManagedStatic registries, which crashes rather than diagnosing.
if (DDX_SHARED_LIBS AND LLVM_LINK_LLVM_DYLIB)
    target_link_libraries(ddx_rt PRIVATE LLVM)
else ()
    target_link_libraries(ddx_rt PRIVATE ${DDX_LLVM_LIBS})
    if (DDX_SHARED_LIBS AND NOT MSVC)
        target_link_options(ddx_rt PRIVATE -Wl,--exclude-libs,ALL)
    endif ()
endif ()
