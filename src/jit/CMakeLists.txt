# The only LLVM-aware code.  An OBJECT library so it has its own include path.
add_library(ddx_jit_objs OBJECT codegen.cpp compiler.cpp)

set_target_properties(ddx_jit_objs PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

target_compile_definitions(ddx_jit_objs PRIVATE DDX_BUILDING ${LLVM_DEFINITIONS_LIST})
target_include_directories(ddx_jit_objs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ddx_jit_objs SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(ddx_jit_objs PRIVATE ddx Boost::headers)
ddx_target_flags(ddx_jit_objs)

# RTTI as LLVM has it: compiler.cpp derives from IRCompiler and ObjectCache.
if (NOT LLVM_ENABLE_RTTI)
    target_compile_options(ddx_jit_objs PRIVATE
            $<IF:$<CXX_COMPILER_ID:MSVC>,/GR-,-fno-rtti>)
endif ()

target_sources(ddx_rt PRIVATE $<TARGET_OBJECTS:ddx_jit_objs>)
add_library(ddx::jit ALIAS ddx_rt)

# PUBLIC: consumers' own TUs read these.
target_compile_definitions(ddx_rt PUBLIC
        DDX_HAS_JIT
        DDX_JIT_DEFAULT_OPT=${DDX_JIT_DEFAULT_OPT}
        DDX_JIT_DEFAULT_CONTRACT=${DDX_JIT_DEFAULT_CONTRACT})

# LLVM swallowed and hidden -- tests/jit/check_exports.cmake holds that line.
target_link_libraries(ddx_rt PRIVATE ${DDX_LLVM_LIBS})
if (NOT MSVC AND NOT APPLE)
    target_link_options(ddx_rt PRIVATE -Wl,--exclude-libs,ALL -Wl,--gc-sections)
endif ()
