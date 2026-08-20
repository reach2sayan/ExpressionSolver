# dual/ — the numeric AD scalars the drivers propagate
target_sources(ddx INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/dual.hpp
        ${CMAKE_CURRENT_LIST_DIR}/taylor_dual.hpp
)
