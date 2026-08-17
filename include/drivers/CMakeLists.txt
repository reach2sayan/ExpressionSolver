# drivers/ — everything that actually computes a derivative: the constexpr
# expression-graph sweeps, the runtime scalar and vector-forward Hessian
# drivers, the bridge that feeds a graph to a runtime driver, and the
# compile-time coupling analysis that decides how few sweeps are needed.
target_sources(diff INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/gradient.hpp
        ${CMAKE_CURRENT_LIST_DIR}/forward_driver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/vforward_driver.hpp
        ${CMAKE_CURRENT_LIST_DIR}/seeded_energy.hpp
        ${CMAKE_CURRENT_LIST_DIR}/coupling.hpp
)
