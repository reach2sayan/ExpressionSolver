# interop/ — the optional boundaries to third-party libraries.  Nothing in the
# AD core may include anything from here: these headers are where a result
# leaves the library, and they are the only place a third-party dependency is
# allowed to appear (CI enforces that for Eigen).
target_sources(diff INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/eigen_interop.hpp
)
