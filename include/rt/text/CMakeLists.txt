# The text surface: what a parsed expression is, how one becomes nodes in an
# arena, and the equation() that spells both at once.  Listed like every other
# header directory -- an IDE resolves include paths from the target naming a
# header, and one left off gets none.
target_sources(ddx_rt INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/ast.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/lower.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/equation.hpp>
)
