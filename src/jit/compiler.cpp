#include "codegen.hpp"

#include "rt/derivative.hpp"
#include "rt/expressions.hpp"
#include "util/scope_guard.hpp"

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/CodeGen/ReplaceWithVeclib.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/Mangling.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/PassTimingInfo.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/ThreadPool.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <concepts>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace ddx::jit {
namespace {

// Global LLVM state, and a process may hold more than one Compiler.
void init_native_target_once() {
  static std::once_flag once;
  std::call_once(once, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  });
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
// one: promising a vector sin that cannot be called fails at link.  Nothing is
// registered unless asked for -- see VecLib in kernel.hpp for why the default
// declines.
[[nodiscard]] bool want_veclib(const llvm::Triple &triple, const Options &opt,
                               const bool have) {
  return have && (opt.veclib == VecLib::Libmvec ||
                  (opt.veclib == VecLib::Auto && triple.isOSLinux() &&
                   triple.getArch() == llvm::Triple::x86_64));
}

llvm::TargetLibraryInfoImpl target_library_info(const llvm::Triple &triple,
                                                const Options &opt,
                                                const bool have) {
  llvm::TargetLibraryInfoImpl tlii(triple);
  if (want_veclib(triple, opt, have)) {
    tlii.addVectorizableFunctionsFromVecLib(
        llvm::TargetLibraryInfoImpl::LIBMVEC_X86, triple);
  }
  return tlii;
}

// The codegen level rides on the module as a flag, so it is a property of the
// compile that asked and not of the JIT, which is what lets one JIT serve
// compiles at different levels.
constexpr const char *kCodegenFlag = "ddx.codegen";

llvm::CodeGenOptLevel codegen_level_of(const llvm::Module &m) {
  const auto *const flag = llvm::mdconst::extract_or_null<llvm::ConstantInt>(
      m.getModuleFlag(kCodegenFlag));
  switch (flag != nullptr ? flag->getZExtValue() : 2) {
  case 0:
    return llvm::CodeGenOptLevel::None;
  case 1:
    return llvm::CodeGenOptLevel::Less;
  case 3:
    return llvm::CodeGenOptLevel::Aggressive;
  default:
    return llvm::CodeGenOptLevel::Default;
  }
}

// ConcurrentIRCompiler, with the level read off the module: one machine per
// module, which is what makes two compiles at once safe -- the default
// SimpleCompiler shares one whose subtarget cache is unsynchronised.
class LevelledCompiler final : public llvm::orc::IRCompileLayer::IRCompiler {
public:
  explicit LevelledCompiler(llvm::orc::JITTargetMachineBuilder machine)
      : IRCompiler(llvm::orc::irManglingOptionsFromTargetOptions(
            machine.getOptions())),
        machine_(std::move(machine)) {}

  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  operator()(llvm::Module &m) override {
    auto machine = machine_;
    machine.setCodeGenOptLevel(codegen_level_of(m));
    auto tm = machine.createTargetMachine();
    return (!tm) ? tm.takeError() : llvm::orc::SimpleCompiler(**tm)(m);
  }

private:
  llvm::orc::JITTargetMachineBuilder machine_;
};

// The widest fixed vector the host holds, in doubles.  Asked of the target's
// own cost model rather than read off a feature list, so prefer-256-bit parts
// answer 4 as the backend would split them anyway.
[[nodiscard]] unsigned host_lanes(llvm::orc::JITTargetMachineBuilder machine) {
  auto tm = machine.createTargetMachine();
  if (!tm) {
    llvm::consumeError(tm.takeError());
    return 1;
  }
  llvm::LLVMContext ctx;
  llvm::Module probe("ddx.probe", ctx);
  auto *const fn = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), false),
      llvm::Function::ExternalLinkage, "probe", probe);
  const llvm::TypeSize bits =
      (*tm)->getTargetTransformInfo(*fn).getRegisterBitWidth(
          llvm::TargetTransformInfo::RGK_FixedWidthVector);
  return std::max(1u, static_cast<unsigned>(bits.getFixedValue() / 64));
}

// The body is one straight-line block the graph has already shared, so what
// the middle end can do for it is fold: no loop passes, no vectoriser, no
// inliner.  Under a vector library the intrinsics over lanes are rewritten to
// its entry points here; with none they stay intrinsics, which the backend
// unrolls into the scalar libm calls the interpreter makes.
[[nodiscard]] llvm::Error optimize(llvm::Module &m, llvm::TargetMachine &tm,
                                   const Options &opt,
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

  // The two vectorisers the default pipeline would otherwise bring.  Both are
  // wrong here by default: the loop is already emitted `lanes` wide, and the
  // loop vectoriser in particular pays for an alias check between every pair
  // of columns and then declines.
  llvm::PipelineTuningOptions pto;
  pto.SLPVectorization = opt.slp;
  pto.LoopVectorization = opt.loop_vectorize;

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
  if (want_veclib(triple, opt, have_libmvec)) {
    llvm::FunctionPassManager fpm;
    fpm.addPass(llvm::ReplaceWithVeclib());
    mpm.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));
  }
  mpm.run(m, mam);
  return llvm::Error::success();
}

using Clock = std::chrono::steady_clock;

// What a compile needs of the JIT it is going into.  `code` is what a Kernel
// holds a share of, so the code outlives every Compiler that named it.
//
// The machine is a *recipe*, not an instance: llvm::TargetMachine caches its
// subtargets in an unsynchronised map, so two compiles running at once must not
// share one.  This is the same reason ORC's own ConcurrentIRCompiler builds one
// per module.
struct Host {
  llvm::orc::LLJIT &jit;
  llvm::orc::JITTargetMachineBuilder machine;
  std::string triple;
  bool libmvec;
  unsigned lanes; // What Options::lanes == 0 means here
  std::shared_ptr<void> code;
};

// One graph compiled once, filling the report it was handed as it goes.
class Compilation : private impl::pinned {
public:
  Compilation(Host host, const rt::Graph<double> &g, const Options &opt,
              std::string name, CompileReport &into)
      : host_(std::move(host)), g_(g), opt_(opt), name_(std::move(name)),
        rep_(into) {}

  [[nodiscard]] result<llvm::orc::ThreadSafeModule> prepared() {
    return emitted().and_then([this](llvm::orc::ThreadSafeModule m) {
      return optimized(std::move(m));
    });
  }

  [[nodiscard]] result<Kernel> operator()() {
    return prepared().and_then([this](llvm::orc::ThreadSafeModule m) {
      return materialised(std::move(m));
    });
  }

private:
  [[nodiscard]] result<llvm::orc::ThreadSafeModule> emitted() {
    const impl::scope_exit clock{
        [this, start = Clock::now()] { rep_.emit = Clock::now() - start; }};
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto m = detail::emit_module(*ctx, g_, opt_, name_,
                                 opt_.lanes != 0 ? opt_.lanes : host_.lanes);
    if (!m) {
      // The emitter has already written the verifier's own diagnosis.
      return std::unexpected{
          error{errc::jit_verify, "the emitted module failed verification"}};
    }
    m->setDataLayout(host_.jit.getDataLayout());
    m->setTargetTriple(host_.triple);
    m->addModuleFlag(llvm::Module::Error, kCodegenFlag, opt_.codegen_level);
    rep_.nodes = g_.live_count();
    rep_.instructions = m->getInstructionCount();
    return llvm::orc::ThreadSafeModule{
        std::move(m), llvm::orc::ThreadSafeContext{std::move(ctx)}};
  }

  [[nodiscard]] result<llvm::orc::ThreadSafeModule>
  optimized(llvm::orc::ThreadSafeModule m) {
    const impl::scope_exit clock{
        [&, start = Clock::now()] { rep_.optimize = Clock::now() - start; }};
    // One machine per compile -- see Host.
    auto tm = host_.machine.createTargetMachine();
    if (!tm) {
      return std::unexpected{as_error(
          errc::jit_target, "creating a target machine", tm.takeError())};
    }
    std::optional<error> refused;
    m.withModuleDo([&](llvm::Module &mod) {
      if (auto e = optimize(mod, **tm, opt_, host_.libmvec)) {
        refused = as_error(errc::jit_module, "building the pass pipeline",
                           std::move(e));
      }
    });

    if (refused) {
      return std::unexpected{*refused};
    }
    return m;
  }

  // The backend runs on materialisation, not on addIRModule, so the lookup is
  // inside this phase or its number would be the wrong one.
  [[nodiscard]] result<Kernel> materialised(llvm::orc::ThreadSafeModule m) {
    const impl::scope_exit clock{
        [this, start = Clock::now()] { rep_.codegen = Clock::now() - start; }};
    if (auto e = host_.jit.addIRModule(std::move(m))) {
      return std::unexpected{
          as_error(errc::jit_module, "adding the module", std::move(e))};
    }
    auto sym = host_.jit.lookup(name_);
    if (!sym) {
      return std::unexpected{
          as_error(errc::jit_lookup, "looking up " + name_, sym.takeError())};
    }
    const auto &layout = g_.layout();
    return Kernel{sym->toPtr<Kernel::function_type>(),
                  g_.symbols().size(),
                  layout.values,
                  layout.jacobian,
                  layout.hessian,
                  std::move(host_.code)};
  }

  Host host_;
  const rt::Graph<double> &g_;
  const Options &opt_;
  std::string name_;
  CompileReport &rep_;
};

// Where a background compile runs.
//
// Deliberately not a member of Impl: a Kernel holds a share of the Impl it came
// from, so the last share can be dropped on a worker -- and a pool joined from
// its own thread deadlocks.
//
// `warm_` being a *member* is the whole point, and it is what makes the order a
// class invariant rather than a rule about where a call gets written.  LLVM
// registers atexit entries lazily, *while it compiles*; anything registered
// after this object is destroyed before it, so the pool's join at exit would
// run its workers against freed LLVM state -- which faults, or hangs in the
// join. A member completes before the enclosing object does, so compiling
// inside one puts LLVM's entries strictly below ours however the members are
// ordered, and the pool cannot be reached without having gone through it.
// Templated only so it need not name Compiler::Impl, which is private -- so
// what it wants of `I` is spelled here rather than named: the one thing it does
// with one is compile through it.
template <typename I>
  requires requires(const std::shared_ptr<I> &impl, const rt::Graph<double> &g,
                    CompileReport &rep) {
    { I::run(impl, g, Options{}, rep) } -> std::same_as<result<Kernel>>;
  }
class Compiles : private impl::pinned {
public:
  // The pool every background compile runs on, brought up once and never torn
  // down before the LLVM statics its workers touch.
  [[nodiscard]] static Compiles &shared(const std::shared_ptr<I> &impl) {
    static Compiles instance{impl};
    return instance;
  }

  [[nodiscard]] llvm::DefaultThreadPool &threads() noexcept { return threads_; }

private:
  explicit Compiles(const std::shared_ptr<I> &impl) : warm_(impl) {}

  // Constructing one compiles something, which is what makes LLVM register.
  // Any live Impl will do: what is being ordered is LLVM's statics, not this.
  struct Warm {
    explicit Warm(const std::shared_ptr<I> &impl) {
      rt::Builder<double> b;
      const auto x = rt::var(b, "x");
      const auto y = rt::var(b, "y");
      // Every op the emitter can produce, off the same table it emits from:
      // what has to be registered before the pool is whatever LLVM touches
      // lazily, and that follows the ops, so enumerate them rather than hope.
      auto e = x / (y + 1.0) - x * y;
#define DDX_JIT_WARM(fn, Op, label, ...) e += fn(x);
      DDX_UNARY_MATH_TABLE(DDX_JIT_WARM)
#undef DDX_JIT_WARM
      e += pow(x, y) + atan2(x, y) + hypot(x, y);
      e += abs(x) + max(x, y) + min(x, y) + sign(x) + (-x);
      // Jacobian too: reverse mode emits ops the value alone never reaches.
      const auto row = rt::build_jacobian_impl<impl::DiffMode::Reverse>(b, e.id(b));
      const auto g = rt::GraphBuilder<double>{b}
                         .value(e)
                         .jacobian_from(row.partial)
                         .build();
      CompileReport discard;
      (void)I::run(impl, g, Options{}, discard);
    }
  };

  Warm warm_; // LLVM registers while this runs, so before we do
  llvm::DefaultThreadPool threads_;
};

} // namespace

struct Compiler::Impl {
  std::unique_ptr<llvm::orc::LLJIT> jit;
  // Copied before the builder is handed to LLJIT: every compile stamps out its
  // own machine from it.
  std::optional<llvm::orc::JITTargetMachineBuilder> machine;
  std::string triple;
  // The only shared mutable state: LLJIT is internally synchronised, but two
  // threads naming a module alike would hand it a duplicate symbol.
  std::atomic<unsigned> counter{0};
  bool libmvec = false; // Whether the vector forms resolve here
  unsigned lanes = 1;   // The host's vector width in doubles

  [[nodiscard]] static std::expected<std::shared_ptr<Impl>, error> bring_up() {
    init_native_target_once();
    auto impl = std::make_shared<Impl>();

    // detectHost pins the host CPU and every feature it has: parity with the
    // project's -march=native.
    auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jtmb) {
      return std::unexpected{
          as_error(errc::jit_target, "detecting the host", jtmb.takeError())};
    }
    impl->machine = *jtmb;
    impl->triple = jtmb->getTargetTriple().str();
    impl->lanes = host_lanes(*jtmb);

    // Named outright rather than reached through setNumCompileThreads, so the
    // backend runs on the thread that asked for the kernel and no idle pool is
    // spawned.
    auto compiler_factory = [](llvm::orc::JITTargetMachineBuilder machine)
        -> llvm::Expected<
            std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
      return std::make_unique<LevelledCompiler>(std::move(machine));
    };
    if (auto jit = llvm::orc::LLJITBuilder()
                       .setJITTargetMachineBuilder(std::move(*jtmb))
                       .setCompileFunctionCreator(std::move(compiler_factory))
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

  // Named on Impl rather than on Compiler so a queued compile needs only a
  // share of this, not a Compiler that may have been moved from or gone.
  [[nodiscard]] static result<Kernel> run(const std::shared_ptr<Impl> &self,
                                          const rt::Graph<double> &g,
                                          const Options &opt,
                                          CompileReport &rep) {
    Compilation work{Host{*self->jit, *self->machine, self->triple,
                          self->libmvec, self->lanes, self},
                     g, opt, "ddx_kernel_" + std::to_string(self->counter++),
                     rep};
    return work();
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
  CompileReport discard;
  return Impl::run(impl_, g, opt, report != nullptr ? *report : discard);
}

// The future is a packaged_task's, never std::async's: only the latter has a
// destructor that joins, which would put a compile on the critical path of
// whatever dropped the last handle.
std::shared_future<result<Kernel>>
Compiler::compile_async(std::shared_ptr<const rt::Graph<double>> g,
                        Options opt) {
  auto task = std::make_shared<std::packaged_task<result<Kernel>()>>(
      [self = impl_, graph = std::move(g), opt] {
        CompileReport discard;
        return Impl::run(self, *graph, opt, discard);
      });
  auto landing = task->get_future().share();
  // The task holds the future as well as the promise, so a caller that drops
  // its copy mid-compile is never the one that tears the shared state down --
  // doing that under a running compile crashes inside LLVM.  A std::function
  // has to be copyable, which a packaged_task is not, hence the shared_ptr.
  Compiles<Impl>::shared(impl_).threads().async(
      [task, landing] { std::invoke(*task); });
  return landing;
}

result<std::string> Compiler::render_ir(const rt::Graph<double> &g,
                                        const Options &opt) const {
  CompileReport discard;
  Compilation run{Host{*impl_->jit, *impl_->machine, impl_->triple,
                       impl_->libmvec, impl_->lanes, impl_},
                  g, opt, "ddx_kernel_dump", discard};
  return run.prepared().transform([](llvm::orc::ThreadSafeModule m) {
    std::string out;
    llvm::raw_string_ostream os(out);
    m.withModuleDo([&](llvm::Module &mod) { mod.print(os, nullptr); });
    return out;
  });
}

} // namespace ddx::jit
