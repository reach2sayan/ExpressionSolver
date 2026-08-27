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
# ddx throws nothing -- errors come back as values -- so -fno-exceptions has no
# configuration to choose.  Our own targets only: what a consumer compiles with
# is theirs.

if (MSVC)
    # /bigobj: one TU instantiates past the 2^16 COFF section limit.
    set(DDX_CODEGEN_FLAGS /arch:AVX2 /bigobj)
    # /Wall noise an expression-template library emits by the thousand; C4365
    # and C5219 are deliberately left on.  C4866 fires on every range-view
    # subscript, where there is no evaluation order to get wrong, and C4061
    # wants a case per enumerator even where the switch has a default.
    set(DDX_WARNINGS /Wall /wd4061 /wd4623 /wd4625 /wd4626 /wd5026 /wd5027
                     /wd4710 /wd4711 /wd4868 /wd4820 /wd5045 /wd5246 /wd4514
                     /wd4324 /wd5266 /wd4866)
    # Boost arrives on an IMPORTED target, so CMake already spells its include
    # root /external:I and quiets it with /external:W0.  That much stops at a
    # template: a warning inside one Boost defines is charged to the code that
    # instantiates it, which is ours, and Boost.Parser is templates the whole
    # way down.  /external:templates- charges it to the header instead.
    list(APPEND DDX_WARNINGS /external:templates-)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    # C4577 sits with the flag that causes it, not among the warnings: it
    # reports `noexcept` under /EHs-c- and fires at /W1, so a consumer applying
    # the recorded flags without our /Wall set gets it too.
    list(APPEND DDX_CODEGEN_FLAGS /EHs-c- /D_HAS_EXCEPTIONS=0 /wd4577)
else ()
    if (ENABLE_NATIVE_ARCH)
        set(DDX_CODEGEN_FLAGS -march=native)
    else ()
        set(DDX_CODEGEN_FLAGS -march=x86-64-v3)
    endif ()
    if (DDX_FP_FLAGS)
        list(APPEND DDX_CODEGEN_FLAGS -ffp-contract=fast -fno-math-errno)
    endif ()
    # -funwind-tables with it, matching how LLVM itself is built:
    list(APPEND DDX_CODEGEN_FLAGS -fno-exceptions -funwind-tables)
    set(DDX_WARNINGS -Wall -Wextra -Wpedantic -Wfatal-errors)
endif ()

# Global rather than through ddx_target_flags(): a sanitizer has to instrument
# everything, gtest included.  A no-op on MSVC rather than an error, so a preset
# asking for one still configures on Windows.
set(DDX_SANITIZE "off" CACHE STRING "off | thread | address | undefined")
set_property(CACHE DDX_SANITIZE PROPERTY STRINGS off thread address undefined)
if (NOT DDX_SANITIZE STREQUAL "off")
    if (MSVC)
        message(WARNING "DDX_SANITIZE=${DDX_SANITIZE} ignored: MSVC has no such sanitizer")
    else ()
        add_compile_options(-fsanitize=${DDX_SANITIZE} -fno-omit-frame-pointer -g)
        add_link_options(-fsanitize=${DDX_SANITIZE})
    endif ()
endif ()

# Our own targets only -- never the INTERFACE library, which must not push flags
# onto a consumer.
#
#   ddx_target_flags(<target> [EXCEPTIONS])
#
# EXCEPTIONS keeps exceptions on for one target, where the rest of the tree
# compiles without them: pybind11 translates a throw into a Python exception,
# and Boost.Parser's entry points do not compile without one.  Filtered from the
# recorded list rather than respelled, so -march and -ffp-contract still reach
# those targets as they reach every other -- those change arithmetic, and the
# JIT emits code to match.
#
# Warnings are errors, through the property rather than a flag: it spells itself
# on every compiler, it reaches only what this function is called on -- never a
# fetched dependency -- and `cmake --compile-no-warning-as-error` is the way past
# it when a newer compiler invents a warning nobody has fixed yet.
function(ddx_target_flags target)
    cmake_parse_arguments(PARSE_ARGV 1 arg "EXCEPTIONS" "" "")
    set(flags ${DDX_CODEGEN_FLAGS})
    if (arg_EXCEPTIONS)
        list(REMOVE_ITEM flags -fno-exceptions /EHs-c- /D_HAS_EXCEPTIONS=0)
    endif ()
    target_compile_options(${target} PRIVATE ${flags} ${DDX_WARNINGS})
    set_property(TARGET ${target} PROPERTY COMPILE_WARNING_AS_ERROR ON)
endfunction()

# A shared libddx has to be findable at run time.  CMake's build RPATH covers
# the ELF build tree already; $ORIGIN is what survives the tree being moved, and
# Windows has no RPATH at all, so there the DLL is copied next to the .exe.
# Every executable linking ddx::rt needs it.
function(ddx_runtime_deps target)
    if (NOT DDX_SHARED_LIBS)
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
