# symbolic/ — the compile-time expression graph: the tree, the point it is
# evaluated at, and the symbolic and reverse sweeps that differentiate it.
# Header-only and fully constexpr, and it stays that way: an expression whose
# structure is known when the program is compiled should be a type.
target_sources(ddx_symbolic INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/symbol.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/workspace.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/expressions.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/simplify.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/traits.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/values.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/entry.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/record.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/format.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/bound.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/coupling.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/seeded_energy.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/sweep.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/equation.hpp>
)
