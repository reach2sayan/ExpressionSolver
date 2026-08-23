#include "codegen.hpp"

#include <llvm/ADT/Twine.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>

#include <algorithm>
#include <array>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

namespace ddx::jit::detail {
namespace {

// Which LLVM intrinsic, if any, covers an op; whatever is missing goes out as a
// libm call under label_of(op).  Everything is derived from the op's own label
// except three, where the label is not the intrinsic's name: llvm.abs is the
// *integer* intrinsic, and maximum/minimum are the NaN-propagating pair the
// interpreter matches where maxnum/minnum are not.
llvm::Intrinsic::ID intrinsic_for(rt::OpCode op) {
  switch (op) {
  case rt::OpCode::Abs:
    return llvm::Intrinsic::fabs;
  case rt::OpCode::Max:
    return llvm::Intrinsic::maximum;
  case rt::OpCode::Min:
    return llvm::Intrinsic::minimum;
  default:
    break;
  }
  return llvm::Intrinsic::lookupIntrinsicID(
      llvm::Twine("llvm.").concat(rt::label_of(op)).str());
}

// memory(none) is the IR spelling of -fno-math-errno: without it a call is
// assumed to write errno, which blocks hoisting and vectorisation.
llvm::Function *libm_decl(llvm::Module &m, std::string_view name,
                          unsigned args) {
  llvm::Type *f64 = llvm::Type::getDoubleTy(m.getContext());
  llvm::SmallVector<llvm::Type *, 2> params(args, f64);
  llvm::FunctionCallee c =
      m.getOrInsertFunction(name, llvm::FunctionType::get(f64, params, false));
  auto *fn = llvm::cast<llvm::Function>(c.getCallee());
  fn->setMemoryEffects(llvm::MemoryEffects::none());
  fn->setDoesNotThrow();
  fn->setWillReturn();
  fn->setDoesNotFreeMemory();
  return fn;
}

class Emitter {
public:
  Emitter(llvm::Module &m, llvm::IRBuilder<> &b) : m_(m), b_(b) {}

  llvm::Value *unary(rt::OpCode op, llvm::Value *u) const {
    if (op == rt::OpCode::Neg) {
      return b_.CreateFNeg(u);
    }
    if (op == rt::OpCode::Sign) {
      // u > 0 ? 1 : u < 0 ? -1 : u - u.  The last arm reaches only ±0 and
      // NaN, giving 0 and NaN as sign_impl does.
      llvm::Type *const f64 = b_.getDoubleTy();
      llvm::Value *const zero = llvm::ConstantFP::get(f64, 0.0);
      return b_.CreateSelect(b_.CreateFCmpOGT(u, zero),
                             llvm::ConstantFP::get(f64, 1.0),
                             b_.CreateSelect(b_.CreateFCmpOLT(u, zero),
                                             llvm::ConstantFP::get(f64, -1.0),
                                             b_.CreateFSub(u, u)));
    }
    return call(op, {u});
  }

  llvm::Value *binary(rt::OpCode op, llvm::Value *l, llvm::Value *r) const {
    switch (op) {
    case rt::OpCode::Add:
      return b_.CreateFAdd(l, r);
    case rt::OpCode::Mul:
      return b_.CreateFMul(l, r);
    case rt::OpCode::Div:
      return b_.CreateFDiv(l, r);
    default:
      return call(op, {l, r});
    }
  }

private:
  llvm::Value *call(rt::OpCode op, llvm::ArrayRef<llvm::Value *> args) const {
    const llvm::Intrinsic::ID id = intrinsic_for(op);
    if (id != llvm::Intrinsic::not_intrinsic) {
      return b_.CreateCall(
          llvm::Intrinsic::getOrInsertDeclaration(&m_, id, {b_.getDoubleTy()}),
          args);
    }
    return b_.CreateCall(
        libm_decl(m_, rt::label_of(op), static_cast<unsigned>(args.size())),
        args);
  }

  llvm::Module &m_;
  llvm::IRBuilder<> &b_;
};

llvm::Function *declare_kernel(llvm::Module &m, llvm::StringRef name) {
  llvm::LLVMContext &ctx = m.getContext();
  llvm::Type *const i64 = llvm::Type::getInt64Ty(ctx);
  llvm::PointerType *const ptr = llvm::PointerType::getUnqual(ctx);
  // xs, f, g, h, n -- four column arrays and the batch length.
  auto *const fty = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx),
                                            {ptr, ptr, ptr, ptr, i64}, false);
  auto *const fn =
      llvm::Function::Create(fty, llvm::Function::ExternalLinkage, name, m);

  static constexpr std::array names{"xs", "f", "g", "h", "n"};
  for (const auto [i, arg_name] : names | std::views::enumerate) {
    fn->getArg(static_cast<unsigned>(i))->setName(arg_name);
  }
  // Saying the columns never alias is what lets the vectoriser skip its
  // runtime overlap check.
  for (const unsigned i : std::views::iota(0u, 4u)) {
    fn->addParamAttr(i, llvm::Attribute::NoAlias);
    fn->addParamAttr(i, llvm::Attribute::NoCapture);
  }
  fn->addParamAttr(0, llvm::Attribute::ReadOnly);

  // nounwind: without it the optimiser cannot move code across the libm
  // calls.  Kernel::operator() matches.
  fn->setDoesNotThrow();
  fn->setWillReturn();
  fn->setMemoryEffects(llvm::MemoryEffects::argMemOnly());
  return fn;
}

llvm::FastMathFlags flags_for(const Options &opt) {
  // Contraction only, matching -ffp-contract=fast: reassociation would change
  // derivative values.
  llvm::FastMathFlags fmf;
  if (opt.contract) {
    fmf.setAllowContract();
  }
  return fmf;
}

// Loaded once in the entry block, so they are loop-invariant.
struct Columns {
  std::vector<llvm::Value *> inputs;
  std::vector<llvm::Value *> values;   // f[k]
  std::vector<llvm::Value *> jacobian; // g[k*n + j]
  std::vector<llvm::Value *> hessian;  // h[c*n + i]
};

Columns hoist_columns(llvm::IRBuilder<> &b, llvm::Function &fn,
                      const rt::Graph<double> &g) {
  llvm::PointerType *const ptr = llvm::PointerType::getUnqual(fn.getContext());

  const auto load_columns = [&](unsigned arg, std::size_t count,
                                const char *stem) {
    return std::ranges::to<std::vector<llvm::Value *>>(
        std::views::iota(0uz, count) |
        std::views::transform([&](std::size_t j) {
          llvm::Value *const slot =
              b.CreateConstInBoundsGEP1_64(ptr, fn.getArg(arg), j);
          return b.CreateLoad(ptr, slot, stem + std::to_string(j));
        }));
  };

  const auto &layout = g.layout();
  return {.inputs = load_columns(0, g.symbols().size(), "col"),
          .values = load_columns(1, layout.values, "f"),
          .jacobian = load_columns(2, layout.jacobian, "g"),
          .hessian = load_columns(3, layout.hessian, "h")};
}

// Ids are topological, so one pass needs no worklist: every operand is already
// a llvm::Value when read.  `value` is carried in rather than returned so that
// a slab can seed it with what it loads from scratch first.
void emit_nodes_into(const Emitter &emit, llvm::IRBuilder<> &b,
                     const rt::Graph<double> &g,
                     std::span<const rt::NodeId> nodes, const Columns &cols,
                     llvm::Value *index, std::vector<llvm::Value *> &value) {
  llvm::Type *const f64 = b.getDoubleTy();
  for (const rt::NodeId v : nodes) {
    const auto &p = g[v];
    const auto operands = g.operands(v);
    switch (rt::arity_of(p.op)) {
    case 0:
      value[v] =
          p.op == rt::OpCode::Const
              ? llvm::cast<llvm::Value>(llvm::ConstantFP::get(f64, p.value))
              : b.CreateLoad(
                    f64, b.CreateInBoundsGEP(f64, cols.inputs[p.slot], index));
      break;
    case 1:
      value[v] = emit.unary(p.op, value[operands[0]]);
      break;
    default:
      value[v] = emit.binary(p.op, value[operands[0]], value[operands[1]]);
      break;
    }
  }
}

[[nodiscard]] std::vector<llvm::Value *>
emit_nodes(const Emitter &emit, llvm::IRBuilder<> &b,
           const rt::Graph<double> &g, const Columns &cols,
           llvm::Value *index) {
  std::vector<llvm::Value *> value(g.size(), nullptr);
  emit_nodes_into(emit, b, g, g.live_order(), cols, index, value);
  return value;
}

void emit_stores(llvm::IRBuilder<> &b, const rt::Graph<double> &g,
                 const Columns &cols, std::span<llvm::Value *const> value,
                 llvm::Value *index) {
  llvm::Type *const f64 = b.getDoubleTy();
  const auto blocks = g.output_blocks();

  const auto store_block = [&](const std::vector<llvm::Value *> &columns,
                               std::span<const rt::NodeId> block) {
    for (const auto [column, o] : std::views::zip(columns, block)) {
      b.CreateStore(value[o], b.CreateInBoundsGEP(f64, column, index));
    }
  };
  store_block(cols.values, blocks.values);
  store_block(cols.jacobian, blocks.jacobian);
  store_block(cols.hessian, blocks.hessian);
}

// xs, f, g, h, scratch, offset, n.  A slab runs over a *block* of the batch:
// `offset` is where the block starts in the caller's columns, and `n` is how
// long it is, so scratch is sized by the block rather than by the batch and
// stays in cache.  That is the difference between a split kernel that is worth
// compiling and one that streams tens of megabytes past every cache.
llvm::Function *declare_slab(llvm::Module &m, llvm::StringRef name) {
  llvm::LLVMContext &ctx = m.getContext();
  llvm::Type *const i64 = llvm::Type::getInt64Ty(ctx);
  llvm::PointerType *const ptr = llvm::PointerType::getUnqual(ctx);
  auto *const fty = llvm::FunctionType::get(
      llvm::Type::getVoidTy(ctx), {ptr, ptr, ptr, ptr, ptr, i64, i64}, false);
  auto *const fn =
      llvm::Function::Create(fty, llvm::Function::ExternalLinkage, name, m);

  static constexpr std::array names{"xs", "f",      "g", "h",
                                    "scratch", "offset", "n"};
  for (const auto [i, arg_name] : names | std::views::enumerate) {
    fn->getArg(static_cast<unsigned>(i))->setName(arg_name);
  }
  for (const unsigned i : std::views::iota(0u, 5u)) {
    fn->addParamAttr(i, llvm::Attribute::NoAlias);
    fn->addParamAttr(i, llvm::Attribute::NoCapture);
  }
  fn->addParamAttr(0, llvm::Attribute::ReadOnly);
  fn->setDoesNotThrow();
  fn->setWillReturn();
  fn->setMemoryEffects(llvm::MemoryEffects::argMemOnly());
  return fn;
}

// scratch[slot * n + point]: slot-major, so one slot's points are contiguous
// and a slab reads and writes them the way it walks them.
llvm::Value *scratch_at(llvm::IRBuilder<> &b, llvm::Value *scratch,
                        std::size_t slot, llvm::Value *count,
                        llvm::Value *index) {
  llvm::Type *const i64 = b.getInt64Ty();
  llvm::Value *const base =
      b.CreateMul(llvm::ConstantInt::get(i64, slot), count);
  return b.CreateInBoundsGEP(b.getDoubleTy(), scratch,
                             b.CreateAdd(base, index));
}

} // namespace

std::unique_ptr<llvm::Module> emit_slab(llvm::LLVMContext &ctx,
                                        const rt::Graph<double> &g,
                                        const Partition &p, std::size_t slab,
                                        const Options &opt,
                                        llvm::StringRef name) {
  auto m = std::make_unique<llvm::Module>("ddx.jit", ctx);
  llvm::Function *const fn = declare_slab(*m, name);
  llvm::Type *const i64 = llvm::Type::getInt64Ty(ctx);
  llvm::Type *const f64 = llvm::Type::getDoubleTy(ctx);
  llvm::Argument *const scratch = fn->getArg(4);
  llvm::Argument *const offset = fn->getArg(5);
  llvm::Argument *const count = fn->getArg(6);
  const Slab &s = p.slabs[slab];

  auto *const entry = llvm::BasicBlock::Create(ctx, "entry", fn);
  auto *const loop = llvm::BasicBlock::Create(ctx, "loop", fn);
  auto *const exit = llvm::BasicBlock::Create(ctx, "exit", fn);
  const llvm::FastMathFlags fmf = flags_for(opt);

  llvm::IRBuilder<> b(entry);
  b.setFastMathFlags(fmf);
  const Columns cols = hoist_columns(b, *fn, g);
  b.CreateCondBr(b.CreateICmpEQ(count, llvm::ConstantInt::get(i64, 0)), exit,
                 loop);

  b.SetInsertPoint(loop);
  b.setFastMathFlags(fmf);
  llvm::PHINode *const index = b.CreatePHI(i64, 2, "i");
  index->addIncoming(llvm::ConstantInt::get(i64, 0), entry);

  // What earlier slabs left, before anything reads it.
  std::vector<llvm::Value *> value(g.size(), nullptr);
  for (const rt::NodeId v : s.live_in) {
    value[v] = b.CreateLoad(f64, scratch_at(b, scratch, p.slot[v], count, index));
  }

  llvm::Value *const column_index = b.CreateAdd(offset, index);
  const Emitter emit(*m, b);
  emit_nodes_into(emit, b, g, s.nodes, cols, column_index, value);

  for (const rt::NodeId v : s.live_out) {
    b.CreateStore(value[v], scratch_at(b, scratch, p.slot[v], count, index));
  }

  // Only the columns this slab defines: every output is stored exactly once,
  // by whichever slab computes it.
  const auto mine = [&](rt::NodeId v) { return value[v] != nullptr; };
  const auto blocks = g.output_blocks();
  const auto store_block = [&](const std::vector<llvm::Value *> &columns,
                               std::span<const rt::NodeId> block) {
    for (const auto [column, o] : std::views::zip(columns, block)) {
      if (mine(o)) {
        b.CreateStore(value[o],
                      b.CreateInBoundsGEP(f64, column, column_index));
      }
    }
  };
  store_block(cols.values, blocks.values);
  store_block(cols.jacobian, blocks.jacobian);
  store_block(cols.hessian, blocks.hessian);

  llvm::Value *const next =
      b.CreateAdd(index, llvm::ConstantInt::get(i64, 1), "i.next");
  index->addIncoming(next, loop);
  b.CreateCondBr(b.CreateICmpEQ(next, count), exit, loop);

  b.SetInsertPoint(exit);
  b.CreateRetVoid();

  return llvm::verifyFunction(*fn, &llvm::errs()) ? nullptr : std::move(m);
}

std::unique_ptr<llvm::Module> emit_module(llvm::LLVMContext &ctx,
                                          const rt::Graph<double> &g,
                                          const Options &opt,
                                          llvm::StringRef name) {
  auto m = std::make_unique<llvm::Module>("ddx.jit", ctx);
  llvm::Function *const fn = declare_kernel(*m, name);
  llvm::Type *const i64 = llvm::Type::getInt64Ty(ctx);
  llvm::Argument *const count = fn->getArg(4);

  auto *const entry = llvm::BasicBlock::Create(ctx, "entry", fn);
  auto *const loop = llvm::BasicBlock::Create(ctx, "loop", fn);
  auto *const exit = llvm::BasicBlock::Create(ctx, "exit", fn);
  const llvm::FastMathFlags fmf = flags_for(opt);

  llvm::IRBuilder<> b(entry);
  b.setFastMathFlags(fmf);
  const Columns cols = hoist_columns(b, *fn, g);
  b.CreateCondBr(b.CreateICmpEQ(count, llvm::ConstantInt::get(i64, 0)), exit,
                 loop);

  b.SetInsertPoint(loop);
  b.setFastMathFlags(fmf);
  llvm::PHINode *const index = b.CreatePHI(i64, 2, "i");
  index->addIncoming(llvm::ConstantInt::get(i64, 0), entry);

  const Emitter emit(*m, b);
  const std::vector<llvm::Value *> value = emit_nodes(emit, b, g, cols, index);
  emit_stores(b, g, cols, value, index);

  llvm::Value *const next =
      b.CreateAdd(index, llvm::ConstantInt::get(i64, 1), "i.next");
  index->addIncoming(next, loop);
  b.CreateCondBr(b.CreateICmpEQ(next, count), exit, loop);

  b.SetInsertPoint(exit);
  b.CreateRetVoid();

  return llvm::verifyFunction(*fn, &llvm::errs()) ? nullptr : std::move(m);
}

} // namespace ddx::jit::detail
