#pragma once

#include "jit/kernel.hpp"
#include "partition.hpp"
#include "rt/graph.hpp"

#include <llvm/IR/Module.h>

namespace ddx::jit::detail {

// One graph as a single counted loop over n points.  The caller names the
// function: every module lands in one JITDylib, where a duplicate symbol is an
// error.
[[nodiscard]] std::unique_ptr<llvm::Module>
emit_module(llvm::LLVMContext &ctx, const rt::Graph<double> &g,
            const Options &opt, llvm::StringRef name);

// One slab of a partitioned graph, in a module of its own so that the slabs
// compile independently -- a context is not shared between them for the same
// reason.  A value crossing into or out of the slab travels in the caller's
// scratch at `slot * n + point`; everything else stays in a register, and the
// outputs a slab defines are stored by that slab.
[[nodiscard]] std::unique_ptr<llvm::Module>
emit_slab(llvm::LLVMContext &ctx, const rt::Graph<double> &g,
          const Partition &p, std::size_t slab, const Options &opt,
          llvm::StringRef name);

} // namespace ddx::jit::detail
