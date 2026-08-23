#pragma once

#include "jit/kernel.hpp"
#include "rt/graph.hpp"

#include <llvm/IR/Module.h>

namespace ddx::jit::detail {

// One graph as a single counted loop over n points.  The caller names the
// function: every module lands in one JITDylib, where a duplicate symbol is an
// error.
[[nodiscard]] std::unique_ptr<llvm::Module>
emit_module(llvm::LLVMContext &ctx, const rt::Graph<double> &g,
            const Options &opt, llvm::StringRef name);

} // namespace ddx::jit::detail
