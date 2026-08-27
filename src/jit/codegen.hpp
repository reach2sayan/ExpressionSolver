#pragma once

#include "jit/kernel.hpp"
#include "rt/graph.hpp"

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Module.h>

namespace ddx::jit::detail {

// One graph as a single counted loop over n points, `lanes` points per
// iteration: the body is emitted over <lanes x double>, or over double when
// lanes is 1.
[[nodiscard]] std::unique_ptr<llvm::Module>
emit_module(llvm::LLVMContext &ctx, const rt::Graph<double> &g,
            llvm::StringRef name, unsigned lanes,
            const llvm::DataLayout &layout, llvm::StringRef triple);

} // namespace ddx::jit::detail
