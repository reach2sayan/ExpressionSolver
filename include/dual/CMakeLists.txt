# dual/ — the numeric AD scalars the drivers propagate: the nestable first-order
# Dual, and the flat TaylorDual for univariate higher order.
target_sources(ddx INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/dual.hpp
        ${CMAKE_CURRENT_LIST_DIR}/taylor_dual.hpp
)
