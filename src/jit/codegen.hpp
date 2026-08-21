#pragma once

#include "jit/kernel.hpp"
#include "rt/graph.hpp"

#include <llvm/IR/Module.h>

namespace ddx::jit::detail {

// Emit one graph as a single counted loop over n points.  The caller names the
// function, because each compile adds its own module to one JITDylib and two
// definitions of the same symbol there are an error.
[[nodiscard]] std::unique_ptr<llvm::Module>
emit_module(llvm::LLVMContext &ctx, const rt::Graph<double> &g,
            const Options &opt, llvm::StringRef name);

} // namespace ddx::jit::detail
