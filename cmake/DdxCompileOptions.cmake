# The flags our own targets are built with, and the helpers that apply them.
# Nothing here touches the INTERFACE library: ddx::ddx must not push a flag onto
# a consumer, so DDX_CODEGEN_FLAGS and DDX_WARNINGS reach a target only through
# ddx_target_flags().
include_guard(GLOBAL)

option(ENABLE_NATIVE_ARCH "Build optimized for this machine" ON)
# Pin FP contraction, drop errno.  Neither flag is lossy.  Nothing that trades
# accuracy for speed belongs here -- in particular never -ffast-math (nor
# /fp:fast on MSVC, which needs no flags at all).
option(DDX_FP_FLAGS "Pin FP contraction and drop errno on libm calls" ON)
# The headers throw nothing, so a consumer is free to build without exceptions.
# This only turns the flag on for our own targets
option(DDX_NO_EXCEPTIONS "Build our own targets without C++ exceptions" OFF)

if (MSVC)
    # /bigobj: one TU instantiates past the 2^16 COFF section limit.
    set(DDX_CODEGEN_FLAGS /arch:AVX2 /bigobj)
    # /Wall noise a header-only expression-template library emits by the
    # thousand.  C4365 and C5219 are deliberately left on.  C4866 fires on every
    # subscript of a range view -- an overloaded operator[] whose operands are a
    # view and a loop counter, so there is no evaluation order to get wrong.
    set(DDX_WARNINGS /Wall /wd4623 /wd4625 /wd4626 /wd5026 /wd5027 /wd4710
                     /wd4711 /wd4868 /wd4820 /wd5045 /wd5246 /wd4514 /wd4324
                     /wd5266 /wd4866)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if (DDX_NO_EXCEPTIONS)
        list(APPEND DDX_CODEGEN_FLAGS /EHs-c- /D_HAS_EXCEPTIONS=0)
    endif ()
else ()
    if (ENABLE_NATIVE_ARCH)
        set(DDX_CODEGEN_FLAGS -march=native)
    else ()
        set(DDX_CODEGEN_FLAGS -march=x86-64-v3)
    endif ()
    if (DDX_FP_FLAGS)
        list(APPEND DDX_CODEGEN_FLAGS -ffp-contract=fast -fno-math-errno)
    endif ()
    if (DDX_NO_EXCEPTIONS)
        # -funwind-tables with it, matching how LLVM itself is built:
        list(APPEND DDX_CODEGEN_FLAGS -fno-exceptions -funwind-tables)
    endif ()
    set(DDX_WARNINGS -Wall -Wextra -Wpedantic -Wfatal-errors)
endif ()

# Our own targets only -- never the INTERFACE library, which must not push flags
# onto a consumer.
function(ddx_target_flags target)
    target_compile_options(${target} PRIVATE ${DDX_CODEGEN_FLAGS} ${DDX_WARNINGS})
endfunction()

# A shared ddx::jit has to be findable at run time.  CMake's build RPATH covers
# the ELF build tree already; $ORIGIN is what survives the tree being moved, and
# Windows has no RPATH at all, so there the DLL is copied next to the .exe.
function(ddx_jit_runtime target)
    if (NOT DDX_BUILD_JIT OR DDX_JIT_STATIC)
        return ()
    endif ()
    set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "$ORIGIN")
    if (WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
                COMMAND_EXPAND_LISTS)
    endif ()
endfunction()
