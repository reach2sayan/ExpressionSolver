#pragma once
// symbolic/workspace.hpp's SweepWorkspace pinned to the dual orders the drivers
// sweep at.  Only these aliases need Dual complete, which is why the workspace
// itself lives a layer down.

#include "dual/dual.hpp"
#include "symbolic/workspace.hpp"

namespace ddx::impl {

using JacobianWorkspace = SweepWorkspace<dual>;
using HessianWorkspace = SweepWorkspace<dual2nd>;

} // namespace ddx::impl
