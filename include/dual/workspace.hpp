#pragma once
// The forward-mode workspaces: the generic SweepWorkspace of
// symbolic/workspace.hpp pinned to the dual orders the drivers sweep at.
//
// Only these two aliases need Dual to be a complete type, which is why the
// workspace itself lives a layer down -- symbolic/sweep.hpp wants HessianStatic
// and symmetrize from it and has no business acquiring forward mode to get
// them.

#include "dual/dual.hpp"
#include "symbolic/workspace.hpp"

namespace ddx::impl {

using GradientWorkspace = SweepWorkspace<dual>;
using HessianWorkspace = SweepWorkspace<dual2nd>;

} // namespace ddx::impl
