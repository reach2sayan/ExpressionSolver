#include "codegen.hpp"

#include "util/ranges.hpp"

#include <llvm/ADT/Twine.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Alignment.h>

#include <algorithm>
#include <array>
#include <ranges>
#include <span>
#include <string_view>
#include <vector>

namespace ddx::jit::detail {
namespace {

// Which intrinsic covers an op; the rest go out as libm calls.  Derived from the
// label except three: llvm.abs is the *integer* one, and maximum/minimum are the
// NaN-propagating pair the interpreter matches where maxnum/minnum are not.
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

// The IR spelling of -fno-math-errno: without it a call is assumed to write
// errno, which blocks hoisting across it.
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

// W points per iteration, as a double at W == 1 and <W x double> otherwise.
// Every IRBuilder operation is typed on `ty`, so only the memory accesses and
// the calls with no vector intrinsic tell the two apart.
struct Lanes {
  unsigned width;
  llvm::Type *ty;
  [[nodiscard]] constexpr bool vector() const noexcept { return width > 1; }
};

class Emitter {
public:
  Emitter(llvm::Module &m, llvm::IRBuilder<> &b, Lanes lanes)
      : m_(m), b_(b), lanes_(lanes) {}

  // A splat where the lane type is a vector.
  [[nodiscard]] llvm::Value *constant(double v) const {
    return llvm::ConstantFP::get(lanes_.ty, v);
  }

  // Columns are plain double*, so no alignment past the element's is claimed.
  // A load past the batch reads inactive lanes as 1.0: never stored, and inside
  // every op's domain, so a scalarised libm call cannot hit a pole.
  [[nodiscard]] llvm::Value *load(llvm::Value *column, llvm::Value *index,
                                  llvm::Value *mask,
                                  const llvm::Twine &name) const {
    llvm::Value *const at =
        b_.CreateInBoundsGEP(b_.getDoubleTy(), column, index);
    if (!lanes_.vector()) {
      return b_.CreateLoad(lanes_.ty, at, name);
    }
    return b_.CreateMaskedLoad(lanes_.ty, at, llvm::Align(alignof(double)),
                               mask, constant(1.0), name);
  }

  void store(llvm::Value *v, llvm::Value *column, llvm::Value *index,
             llvm::Value *mask) const {
    llvm::Value *const at =
        b_.CreateInBoundsGEP(b_.getDoubleTy(), column, index);
    if (!lanes_.vector()) {
      b_.CreateStore(v, at);
      return;
    }
    b_.CreateMaskedStore(v, at, llvm::Align(alignof(double)), mask);
  }

  llvm::Value *unary(rt::OpCode op, llvm::Value *u) const {
    if (op == rt::OpCode::Neg) {
      return b_.CreateFNeg(u);
    }
    if (op == rt::OpCode::Sign) {
      // u > 0 ? 1 : u < 0 ? -1 : u - u -- the last arm reaches only ±0 and NaN.
      llvm::Value *const zero = constant(0.0);
      return b_.CreateSelect(b_.CreateFCmpOGT(u, zero), constant(1.0),
                             b_.CreateSelect(b_.CreateFCmpOLT(u, zero),
                                             constant(-1.0),
                                             b_.CreateFSub(u, u)));
    }
    return call(op, {u});
  }

  // Not llvm.fmuladd, which is only *permission* to fuse and may be declined
  // where the sweep has committed to one rounding.  With no host FMA this
  // lowers to libm's fma(): slow, and the same bits.
  llvm::Value *fma(const rt::Contraction &c, llvm::Value *x, llvm::Value *y,
                   llvm::Value *z) const {
    return b_.CreateIntrinsic(llvm::Intrinsic::fma, {lanes_.ty},
                              {c.negated ? b_.CreateFNeg(x) : x, y, z});
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
  // An intrinsic is declared on the lane type and left to the backend to unroll;
  // a libm function has only a scalar entry point, so it is unrolled here.
  llvm::Value *call(rt::OpCode op, llvm::ArrayRef<llvm::Value *> args) const {
    const llvm::Intrinsic::ID id = intrinsic_for(op);
    if (id != llvm::Intrinsic::not_intrinsic) {
      return b_.CreateCall(
          llvm::Intrinsic::getOrInsertDeclaration(&m_, id, {lanes_.ty}), args);
    }
    llvm::Function *const fn =
        libm_decl(m_, rt::label_of(op), static_cast<unsigned>(args.size()));
    return lanes_.vector() ? scalarised(fn, args) : b_.CreateCall(fn, args);
  }

  llvm::Value *scalarised(llvm::Function *fn,
                          llvm::ArrayRef<llvm::Value *> args) const {
    llvm::Value *out = llvm::PoisonValue::get(lanes_.ty);
    for (const unsigned lane : std::views::iota(0u, lanes_.width)) {
      llvm::SmallVector<llvm::Value *, 2> scalar;
      for (llvm::Value *const a : args) {
        scalar.push_back(b_.CreateExtractElement(a, lane));
      }
      out = b_.CreateInsertElement(out, b_.CreateCall(fn, scalar), lane);
    }
    return out;
  }

  llvm::Module &m_;
  llvm::IRBuilder<> &b_;
  Lanes lanes_;
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
  // The four pointer *arrays*; the columns they hold are reached through a load,
  // so nothing here says anything about those -- and this is not argmemonly.
  for (const unsigned i : std::views::iota(0u, 4u)) {
    fn->addParamAttr(i, llvm::Attribute::NoAlias);
    fn->addParamAttr(i, llvm::Attribute::NoCapture);
  }
  fn->addParamAttr(0, llvm::Attribute::ReadOnly);

  // nounwind, or the optimiser cannot move code across the libm calls.
  fn->setDoesNotThrow();
  fn->setWillReturn();
  return fn;
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
    return std::views::iota(0uz, count) |
           std::views::transform([&](std::size_t j) {
             llvm::Value *const slot =
                 b.CreateConstInBoundsGEP1_64(ptr, fn.getArg(arg), j);
             return b.CreateLoad(ptr, slot, stem + std::to_string(j));
           }) |
           impl::to<std::vector<llvm::Value *>>();
  };

  const auto &layout = g.layout();
  return {.inputs = load_columns(0, g.symbols().size(), "col"),
          .values = load_columns(1, layout.values, "f"),
          .jacobian = load_columns(2, layout.jacobian, "g"),
          .hessian = load_columns(3, layout.hessian, "h")};
}

// Ids are topological, so one pass needs no worklist.  Under contraction the
// adds carry their product and the multiplies they swallowed are never reached.
[[nodiscard]] std::vector<llvm::Value *>
emit_nodes(const Emitter &emit, const rt::Graph<double> &g, const Columns &cols,
           llvm::Value *index, llvm::Value *mask, bool contract) {
  std::vector<llvm::Value *> value(g.size(), nullptr);
  for (const rt::NodeId v : contract ? g.contracted_order() : g.live_order()) {
    const auto &p = g[v];
    if (contract) {
      if (const rt::Contraction c = rt::contraction_at(g, v)) {
        value[v] = emit.fma(c, value[c.x], value[c.y], value[c.z]);
        continue;
      }
    }
    const auto operands = g.operands(v);
    switch (rt::arity_of(p.op)) {
    case 0:
      value[v] = p.op == rt::OpCode::Const
                     ? emit.constant(p.value)
                     : emit.load(cols.inputs[p.slot], index, mask, "");
      break;
    case 1:
      value[v] = emit.unary(p.op, value[operands[0]]);
      break;
    default:
      value[v] = emit.binary(p.op, value[operands[0]], value[operands[1]]);
      break;
    }
  }
  return value;
}

void emit_stores(const Emitter &emit, const rt::Graph<double> &g,
                 const Columns &cols, std::span<llvm::Value *const> value,
                 llvm::Value *index, llvm::Value *mask) {
  const auto blocks = g.output_blocks();
  const auto store_block = [&](const std::vector<llvm::Value *> &columns,
                               std::span<const rt::NodeId> block) {
    for (const auto [column, o] : std::views::zip(columns, block)) {
      emit.store(value[o], column, index, mask);
    }
  };
  store_block(cols.values, blocks.values);
  store_block(cols.jacobian, blocks.jacobian);
  store_block(cols.hessian, blocks.hessian);
}

} // namespace

std::unique_ptr<llvm::Module>
emit_module(llvm::LLVMContext &ctx, const rt::Graph<double> &g,
            const Options &opt, llvm::StringRef name, const unsigned width) {
  auto m = std::make_unique<llvm::Module>("ddx.jit", ctx);
  llvm::Function *const fn = declare_kernel(*m, name);
  llvm::Type *const i64 = llvm::Type::getInt64Ty(ctx);
  llvm::Type *const f64 = llvm::Type::getDoubleTy(ctx);
  llvm::Argument *const count = fn->getArg(4);
  const Lanes lanes{.width = width,
                    .ty = width > 1 ? llvm::FixedVectorType::get(f64, width)
                                    : f64};

  auto *const entry = llvm::BasicBlock::Create(ctx, "entry", fn);
  auto *const loop = llvm::BasicBlock::Create(ctx, "loop", fn);
  auto *const exit = llvm::BasicBlock::Create(ctx, "exit", fn);

  // None at all: contraction is decided in the graph and spelled llvm.fma, so
  // allowContract would fuse a *second* set the sweep computed separately.
  llvm::IRBuilder<> b(entry);
  const Columns cols = hoist_columns(b, *fn, g);
  b.CreateCondBr(b.CreateICmpEQ(count, llvm::ConstantInt::get(i64, 0)), exit,
                 loop);

  b.SetInsertPoint(loop);
  llvm::PHINode *const index = b.CreatePHI(i64, 2, "i");
  index->addIncoming(llvm::ConstantInt::get(i64, 0), entry);

  // Lane k is live while k < n - i.  Signed on purpose: both sides are
  // non-negative, and it is one instruction where the saturating
  // llvm.get.active.lane.mask is a dozen on AVX2.
  llvm::Value *mask = nullptr;
  if (lanes.vector()) {
    llvm::Value *const remaining = b.CreateSub(count, index, "remaining");
    llvm::Value *const step =
        b.CreateStepVector(llvm::FixedVectorType::get(i64, width));
    mask = b.CreateICmpSLT(step, b.CreateVectorSplat(width, remaining), "mask");
  }

  const Emitter emit(*m, b, lanes);
  const std::vector<llvm::Value *> value =
      emit_nodes(emit, g, cols, index, mask, opt.contract);
  emit_stores(emit, g, cols, value, index, mask);

  llvm::Value *const next =
      b.CreateAdd(index, llvm::ConstantInt::get(i64, width), "i.next");
  index->addIncoming(next, loop);
  b.CreateCondBr(b.CreateICmpUGE(next, count), exit, loop);

  b.SetInsertPoint(exit);
  b.CreateRetVoid();

  return llvm::verifyFunction(*fn, &llvm::errs()) ? nullptr : std::move(m);
}

} // namespace ddx::jit::detail
