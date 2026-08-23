#include "codegen.hpp"

#include "util/ranges.hpp"

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/Mangling.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/PassTimingInfo.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>

#include <llvm/ExecutionEngine/Orc/Core.h>

#include <boost/hof/pipable.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <concepts>
#include <string>
#include <thread>
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
// Nothing is registered unless asked for -- see VecLib in kernel.hpp for why
// the default declines.
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

// Decided here: the count it turns on is a property of the emitted module.
[[nodiscard]] bool lean_for(const llvm::Module &m, const Options &opt) {
  switch (opt.pipeline) {
  case Pipeline::Lean:
    return true;
  case Pipeline::Default:
    return false;
  default:
    return m.getInstructionCount() > opt.lean_above;
  }
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

  // A disabled handler registers nothing, so this is the same pipeline.
  llvm::PassInstrumentationCallbacks pic;
  llvm::TimePassesHandler timers(opt.time_passes);
  timers.setOutStream(llvm::errs());
  timers.registerCallbacks(pic);

  llvm::PipelineTuningOptions pto;
  if (lean_for(m, opt)) {
    pto.LoopVectorization = false;
    pto.SLPVectorization = false;
  }

  llvm::PassBuilder pb(&tm, pto, std::nullopt, &pic);
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

// A phase is In -> result<Out>; `stage` lifts one into a pipe step that a
// failure upstream skips.  Phases being values is what makes render_ir this
// pipeline minus its last step rather than a second copy of it.
template <std::invocable F>
[[nodiscard]] auto measure(std::chrono::nanoseconds &into, F step) {
  const auto start = std::chrono::steady_clock::now();
  auto out = step();
  into = std::chrono::steady_clock::now() - start;
  return out;
}

template <std::move_constructible F>
[[nodiscard]] auto stage(std::chrono::nanoseconds &into, F step) {
  // The pipe's input is whatever the stage above returned, so what is required
  // of it is spelled here rather than named: an expected-like carrying either.
  return boost::hof::pipable(
      [&into, step](auto up)
        requires requires {
          *up;
          up.error();
        }
      {
        using Out = decltype(step(std::move(*up)));
        if (!up) {
          return Out{std::unexpected{std::move(up.error())}};
        }
        return measure(into, [&] { return step(std::move(*up)); });
      })();
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

    // Without a pool the slabs compile one after another on the lookup thread.
    const auto threads = std::min(std::thread::hardware_concurrency(), 8u);
    if (auto jit = llvm::orc::LLJITBuilder()
                       .setJITTargetMachineBuilder(std::move(*jtmb))
                       .setNumCompileThreads(threads)
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
    // cannot see it.  Loaded even though the default declines to use it: a
    // caller that opts in must find the symbols already resolvable.
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

  // One module per slab, each with its own context: not sharing one is what
  // lets the slabs compile on different threads.
  struct Staged {
    std::vector<llvm::orc::ThreadSafeModule> modules;
    std::vector<std::string> names;
    std::size_t scratch = 0; // slots crossing a boundary
    bool split = false;      // emitted as slabs, which is a different signature
  };

  // Sized against the wavefront, not a constant, since that is what a cut costs.
  [[nodiscard]] static std::size_t slab_target(const detail::Partition &probe,
                                               const Options &opt) {
    return std::max(opt.min_slab, opt.slab_factor * probe.peak_wavefront);
  }

  [[nodiscard]] result<Staged> emit(const rt::Graph<double> &g,
                                    const Options &opt, const std::string &name,
                                    CompileReport &rep) const {
    Staged out;
    const bool big = opt.split_above != 0 && g.live_count() > opt.split_above;

    // Read only for its wavefront, which is one linear pass.
    const auto p =
        big ? detail::partition(g, ~std::size_t{0}) : detail::Partition{};
    const auto cut =
        big ? detail::partition(g, slab_target(p, opt)) : detail::Partition{};

    // One slab covering the graph means it will not be cut cheaply; emit it
    // whole rather than as a slab of one carrying the slab signature.
    const bool split = big && cut.slabs.size() > 1;
    const std::size_t count = split ? cut.slabs.size() : 1;
    rep.slabs = count;
    rep.scratch_slots = split ? cut.scratch : 0;

    for (const std::size_t i : std::views::iota(std::size_t{0}, count)) {
      auto ctx = std::make_unique<llvm::LLVMContext>();
      std::string slab_name =
          count == 1 ? name : name + "_s" + std::to_string(i);
      auto m = split ? detail::emit_slab(*ctx, g, cut, i, opt, slab_name)
                     : detail::emit_module(*ctx, g, opt, slab_name);
      if (!m) {
        // The emitter has already written the verifier's own diagnosis.
        return std::unexpected{
            error{errc::jit_verify, "the emitted module failed verification"}};
      }
      m->setDataLayout(jit->getDataLayout());
      m->setTargetTriple(tm->getTargetTriple().str());
      rep.instructions += m->getInstructionCount();
      out.modules.emplace_back(std::move(m),
                               llvm::orc::ThreadSafeContext{std::move(ctx)});
      out.names.push_back(std::move(slab_name));
    }
    out.scratch = split ? cut.scratch : 0;
    out.split = split;
    return out;
  }

  [[nodiscard]] result<Staged> optimized(Staged s, const Options &opt) const {
    for (auto &m : s.modules) {
      m.withModuleDo(
          [&](llvm::Module &mod) { optimize(mod, *tm, opt, libmvec); });
    }
    return s;
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

result<Kernel> Compiler::compile(const rt::Graph<double> &g, const Options &opt,
                                 CompileReport *report) {
  CompileReport ignored;
  CompileReport &rep = report != nullptr ? *report : ignored;
  rep.nodes = g.live_count();

  const std::string name = "ddx_kernel_" + std::to_string(impl_->counter++);
  Impl &impl = *impl_;

  // The backend runs on materialisation, not on addIRModule, so the lookup is
  // inside the codegen stage or its number would be the wrong one.
  const auto lower = [&](Impl::Staged s) -> result<Kernel> {
    for (auto &m : s.modules) {
      if (auto e = impl.jit->addIRModule(std::move(m))) {
        return std::unexpected{
            as_error(errc::jit_module, "adding the module", std::move(e))};
      }
    }

    // One lookup naming every slab, so the compile threads run them together.
    llvm::orc::SymbolLookupSet wanted;
    for (const auto &n : s.names) {
      wanted.add(impl.jit->mangleAndIntern(n));
    }
    auto syms = impl.jit->getExecutionSession().lookup(
        llvm::orc::makeJITDylibSearchOrder(&impl.jit->getMainJITDylib()),
        std::move(wanted));
    if (!syms) {
      return std::unexpected{
          as_error(errc::jit_lookup, "looking up " + name, syms.takeError())};
    }

    const auto &layout = g.layout();
    // The Impl owns the code, so the Kernel holds a share of it.
    if (!s.split) {
      return Kernel{syms->at(impl.jit->mangleAndIntern(s.names.front()))
                        .getAddress()
                        .toPtr<Kernel::function_type>(),
                    g.symbols().size(),
                    layout.values,
                    layout.jacobian,
                    layout.hessian,
                    impl_};
    }
    const auto slabs = s.names |
                       std::views::transform([&](const std::string &n) {
                         return syms->at(impl.jit->mangleAndIntern(n))
                             .getAddress()
                             .toPtr<Kernel::slab_type>();
                       }) |
                       impl::to<std::vector<Kernel::slab_type>>();
    return Kernel{slabs,         s.scratch,       g.symbols().size(),
                  layout.values, layout.jacobian, layout.hessian,
                  impl_};
  };

  return measure(rep.emit, [&] { return impl.emit(g, opt, name, rep); }) |
         stage(rep.optimize,
               [&](Impl::Staged m) {
                 return impl.optimized(std::move(m), opt);
               }) |
         stage(rep.codegen, lower);
}

result<std::string> Compiler::render_ir(const rt::Graph<double> &g,
                                        const Options &opt) const {
  CompileReport rep;
  const Impl &impl = *impl_;
  return (measure(rep.emit,
                  [&] { return impl.emit(g, opt, "ddx_kernel_dump", rep); }) |
          stage(rep.optimize,
                [&](Impl::Staged m) {
                  return impl.optimized(std::move(m), opt);
                }))
      .transform([](Impl::Staged s) {
        std::string out;
        llvm::raw_string_ostream os(out);
        for (auto &m : s.modules) {
          m.withModuleDo([&](llvm::Module &mod) { mod.print(os, nullptr); });
        }
        return out;
      });
}

} // namespace ddx::jit
