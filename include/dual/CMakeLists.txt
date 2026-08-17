# dual/ — the numeric AD scalars the drivers propagate: the nestable first-order
# Dual, the flat TaylorDual for univariate higher order, and the VectorDual lane
# pack that turns one sweep into a whole Hessian row.
target_sources(diff INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/dual.hpp
        ${CMAKE_CURRENT_LIST_DIR}/taylor_dual.hpp
        ${CMAKE_CURRENT_LIST_DIR}/vector_dual.hpp
)
