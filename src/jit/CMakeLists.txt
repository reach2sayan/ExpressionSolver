# The only LLVM-aware code here; kernel.hpp names no LLVM type, so none escapes
# into a consumer's TU.  An OBJECT library, not a second shared one: these link
# into libddx, and the indirection is what gives them their own private include
# path, which a plain target_sources() on ddx_rt could not.
add_library(ddx_jit_objs OBJECT codegen.cpp compiler.cpp)

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
# LLVM's archives, and zlib and zstd with them, so a client loads nothing past
# the C runtime -- tests/jit/check_exports.cmake holds that line.  What is
# swallowed is hidden: a process that already has LLVM loaded is ordinary
# (Mesa's llvmpipe does), and two private copies coexist where one interposed
# over the other's ManagedStatic and cl::opt registries crashes rather than
# diagnosing.  --gc-sections because the archives are function-sectioned and it
# is a fifth of the file.  ld64 has neither flag; the jit preset is Linux.
target_link_libraries(ddx_rt PRIVATE ${DDX_LLVM_LIBS})
if (NOT MSVC AND NOT APPLE)
    target_link_options(ddx_rt PRIVATE -Wl,--exclude-libs,ALL -Wl,--gc-sections)
endif ()
