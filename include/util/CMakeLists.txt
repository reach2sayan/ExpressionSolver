# util/ — the layer with no knowledge of expressions or derivatives: language
# configuration, version macros, compile-time strings, type-list
# metaprogramming, and the RAII seed toggle every driver borrows.
target_sources(diff INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/config.hpp
        ${CMAKE_CURRENT_LIST_DIR}/version.hpp
        ${CMAKE_CURRENT_LIST_DIR}/fixed_string.hpp
        ${CMAKE_CURRENT_LIST_DIR}/mpl.hpp
        ${CMAKE_CURRENT_LIST_DIR}/scope_guard.hpp
)
