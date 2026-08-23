#include "codegen.hpp"

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/Mangling.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>

#include <atomic>
#include <cmath>
#include <string>
#include <utility>

namespace ddx::jit {
namespace {

// Global LLVM state, and a process may hold more than one Compiler.
void init_native_target_once() {
  [[maybe_unused]] static const auto ready = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return 42;
  }();
}

// The one seam between LLVM's error model and ours; its text is the whole of
// what a caller can act on, so it is kept.
[[nodiscard]] error as_error(const errc code, const llvm::Twine &what,
                             llvm::Error e) {
  return error{.code = code,
               .detail = (what + ": " + llvm::toString(std::move(e))).str()};
}

// Every libm entry point an emitted kernel can call, defined outright:
// GetForCurrentProcess resolves only what a loaded module exports, and nothing
// says that is the libm this TU was compiled against.  Defining them pins the
// kernel and the interpreter to the same one.
[[nodiscard]] llvm::Error define_libm(llvm::orc::ExecutionSession &es,
                                      llvm::orc::JITDylib &jd,
                                      const llvm::DataLayout &dl) {
  llvm::orc::MangleAndInterner mangle(es, dl);
  llvm::orc::SymbolMap syms;
  // The unary + turns each lambda into a plain function pointer.
  const auto def = [&](const char *name, auto *fn) {
    syms[mangle(name)] = {llvm::orc::ExecutorAddr::fromPtr(fn),
                          llvm::JITSymbolFlags::Exported |
                              llvm::JITSymbolFlags::Callable};
  };
#define DDX_JIT_LIBM(fn, Op, label, ...)                                       \
  def(label, +[](double u) noexcept { return std::fn(u); });
  DDX_UNARY_MATH_TABLE(DDX_JIT_LIBM)
#undef DDX_JIT_LIBM
  def("fabs", +[](double u) noexcept { return std::fabs(u); });
  def("pow", +[](double l, double r) noexcept { return std::pow(l, r); });
  def("atan2", +[](double l, double r) noexcept { return std::atan2(l, r); });
  def("hypot", +[](double l, double r) noexcept { return std::hypot(l, r); });
  return jd.define(llvm::orc::absoluteSymbols(std::move(syms)));
}

// Whether libmvec resolves in *this process*, not whether the target could have
// one: promising the vectoriser a vector sin it cannot call fails at link.
llvm::TargetLibraryInfoImpl target_library_info(const llvm::Triple &triple,
                                                const Options &opt,
                                                const bool have) {
  llvm::TargetLibraryInfoImpl tlii(triple);
  const bool want =
      have && (opt.veclib == VecLib::Libmvec ||
               (opt.veclib == VecLib::Auto && triple.isOSLinux() &&
                triple.getArch() == llvm::Triple::x86_64));
  if (want) {
    // Without this the vectoriser bails out of any loop with a sin/exp in it.
    tlii.addVectorizableFunctionsFromVecLib(
        llvm::TargetLibraryInfoImpl::LIBMVEC_X86, triple);
  }
  return tlii;
}

void optimize(llvm::Module &m, llvm::TargetMachine &tm, const Options &opt,
              const bool have_libmvec) {
  const llvm::Triple triple(m.getTargetTriple());
  llvm::TargetLibraryInfoImpl tlii =
      target_library_info(triple, opt, have_libmvec);

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PassBuilder pb(&tm);
  // Before registerFunctionAnalyses, which would otherwise install the default
  // llvm::TargetLibraryAnalysis and with it an empty vector-function table.
  fam.registerPass([&] { return llvm::TargetLibraryAnalysis(tlii); });
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::OptimizationLevel level = llvm::OptimizationLevel::O2;
  switch (opt.opt_level) {
  case 0:
    level = llvm::OptimizationLevel::O0;
    break;
  case 1:
    level = llvm::OptimizationLevel::O1;
    break;
  case 3:
    level = llvm::OptimizationLevel::O3;
    break;
  default:
    break;
  }
  llvm::ModulePassManager mpm = level == llvm::OptimizationLevel::O0
                                    ? pb.buildO0DefaultPipeline(level)
                                    : pb.buildPerModuleDefaultPipeline(level);
  mpm.run(m, mam);
}

} // namespace

struct Compiler::Impl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  std::unique_ptr<llvm::TargetMachine> tm;
  // The only shared mutable state: LLJIT is internally synchronised, but two
  // threads naming a module alike would hand it a duplicate symbol.
  std::atomic<unsigned> counter{0};
  bool libmvec = false; // Whether the vector forms resolve here

  [[nodiscard]] static std::expected<std::shared_ptr<Impl>, error> bring_up() {
    init_native_target_once();
    auto impl = std::make_shared<Impl>();

    auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jtmb) {
      return std::unexpected{
          as_error(errc::jit_target, "detecting the host", jtmb.takeError())};
    } else {
      // Parity with the project's -march=native.
      jtmb->setCPU(llvm::sys::getHostCPUName().str());
      for (const auto &[feature, enabled] : llvm::sys::getHostCPUFeatures()) {
        jtmb->getFeatures().AddFeature(feature, enabled);
      }
    }

    if (auto tm = jtmb->createTargetMachine(); !tm) {
      return std::unexpected{as_error(
          errc::jit_target, "creating a target machine", tm.takeError())};
    } else {
      impl->tm = std::move(*tm);
    }

    if (auto jit = llvm::orc::LLJITBuilder()
                       .setJITTargetMachineBuilder(std::move(*jtmb))
                       .create();
        !jit) {
      return std::unexpected{
          as_error(errc::jit_target, "creating the JIT", jit.takeError())};
    } else {
      impl->jit = std::move(*jit);
    }

    llvm::orc::JITDylib &jd = impl->jit->getMainJITDylib();
    const llvm::DataLayout &dl = impl->jit->getDataLayout();
    const char prefix = dl.getGlobalPrefix();

    // Before the generators: a definition here beats anything they would find.
    if (auto e = define_libm(impl->jit->getExecutionSession(), jd, dl)) {
      return std::unexpected{
          as_error(errc::jit_target, "defining libm", std::move(e))};
    }

    if (auto procs =
            llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
                prefix);
        !procs) {
      return std::unexpected{as_error(
          errc::jit_target, "opening the process symbols", procs.takeError())};
    } else {
      jd.addGenerator(std::move(*procs));
    }

    // Not loaded in a program that never called it, so the generator above
    // cannot see it.
    if (auto vec = llvm::orc::DynamicLibrarySearchGenerator::Load(
            "libmvec.so.1", prefix)) {
      jd.addGenerator(std::move(*vec));
      impl->libmvec = true;
    } else {
      // Not an error: it leaves the transcendentals scalar, nothing more.
      llvm::consumeError(vec.takeError());
    }
    return impl;
  }

  [[nodiscard]] result<std::unique_ptr<llvm::Module>>
  build(llvm::LLVMContext &ctx, const rt::Graph<double> &g, const Options &opt,
        const std::string &name) const {
    auto m = detail::emit_module(ctx, g, opt, name);
    if (!m) {
      // emit_module has already written the verifier's own diagnosis to stderr.
      return std::unexpected{
          error{errc::jit_verify, "the emitted module failed verification"}};
    }
    m->setDataLayout(jit->getDataLayout());
    m->setTargetTriple(tm->getTargetTriple().str());
    optimize(*m, *tm, opt, libmvec);
    return m;
  }
};

Compiler::Compiler(std::shared_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

result<Compiler> Compiler::create() {
  return Impl::bring_up().transform(
      [](std::shared_ptr<Impl> impl) { return Compiler{std::move(impl)}; });
}
Compiler::~Compiler() = default;
Compiler::Compiler(Compiler &&) noexcept = default;
Compiler &Compiler::operator=(Compiler &&) noexcept = default;

result<Kernel> Compiler::compile(const rt::Graph<double> &g,
                                 const Options &opt) {
  const std::string name = "ddx_kernel_" + std::to_string(impl_->counter++);
  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto m = impl_->build(*ctx, g, opt, name);
  if (!m) {
    return std::unexpected{m.error()};
  }

  if (auto e = impl_->jit->addIRModule(
          llvm::orc::ThreadSafeModule(std::move(*m), std::move(ctx)))) {
    return std::unexpected{
        as_error(errc::jit_module, "adding the module", std::move(e))};
  }

  auto sym = impl_->jit->lookup(name);
  if (!sym) {
    return std::unexpected{
        as_error(errc::jit_lookup, "looking up " + name, sym.takeError())};
  }

  const auto &layout = g.layout();
  // The Impl owns the code, so the Kernel holds a share of it.
  return Kernel{sym->toPtr<Kernel::function_type>(),
                g.symbols().size(),
                layout.values,
                layout.jacobian,
                layout.hessian,
                impl_};
}

result<std::string> Compiler::render_ir(const rt::Graph<double> &g,
                                        const Options &opt) const {
  auto ctx = std::make_unique<llvm::LLVMContext>();
  return impl_->build(*ctx, g, opt, "ddx_kernel_dump")
      .transform([](const std::unique_ptr<llvm::Module> &m) {
        std::string out;
        llvm::raw_string_ostream os(out);
        m->print(os, nullptr);
        return out;
      });
}

} // namespace ddx::jit
