#include "columns.hpp"
#include "equation.hpp"
#include "error.hpp"
#include "expression.hpp"

#include "rt/builder.hpp"
#include "rt/expressions.hpp"
#include "util/ranges.hpp"
#include "util/scope_guard.hpp"

#include "jit/kernel.hpp"

#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <exception>
#include <memory>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ddx::py {
namespace {

// The arena a model is assembled in, for the duration of the call.  Ours rather
// than the one rt::var(name) reads, because what has to survive is the shared
// ownership: every symbol below goes through the two-argument rt::var.
thread_local std::shared_ptr<rt::Builder<double>> current_arena;

// The exception type Python sees.  Deliberately leaked: it is a type object
// that lives as long as the module, and a translator is a plain function
// pointer, so it cannot capture.
pyb::handle error_class;

[[nodiscard]] PyExpression make_var(std::string_view name) {
  if (!current_arena) {
    // C++ carries a poison instead, unable to afford a branch per symbol.
    // Python already pays more per call than the branch costs.
    fail_with(errc::no_arena);
  }
  return {rt::var(*current_arena, name), current_arena};
}

// One root, or m of them from a tuple or list.  The count is a runtime value
// throughout, which is why these bindings sit below impl::Equation.
[[nodiscard]] std::vector<rt::NodeId> roots_of(const pyb::object &built,
                                               rt::Builder<double> &arena) {
  const bool many =
      pyb::isinstance<pyb::tuple>(built) || pyb::isinstance<pyb::list>(built);
  const auto returned =
      many ? pyb::reinterpret_borrow<pyb::sequence>(built)
           : pyb::reinterpret_borrow<pyb::sequence>(pyb::make_tuple(built));
  if (returned.empty()) {
    fail_with(errc::no_graph); // a system needs at least one function
  }

  return returned | std::views::transform([&arena](const pyb::handle item) {
           // By value, not by reference: a bare number is a pending literal,
           // and implicitly_convertible is what lets `lambda: 2.0` name a
           // constant.
           const auto e = pyb::cast<PyExpression>(item);
           if (e.poisoned()) {
             fail_with(errc::no_arena);
           }
           return e.root_in(arena);
         }) |
         impl::to<std::vector<rt::NodeId>>();
}

// The model as built, or what a cache file holds if it still describes it.
// Best effort throughout: an absent file is the first run and a stale one is a
// rebuild, so neither raises -- Equation.loaded is how a caller tells.
[[nodiscard]] pyb::object
built_or_loaded(const std::optional<std::filesystem::path> &cache,
                std::shared_ptr<rt::Builder<double>> arena,
                std::vector<rt::NodeId> roots) {
  if (cache) {
    if (auto snap = rt::load_snapshot<double>(*cache);
        snap && snap->roots.size() == roots.size() &&
        std::ranges::equal(snap->roots, roots) &&
        rt::digest<double>(snap->symbols, snap->nodes, snap->model_nodes) ==
            rt::digest<double>(arena->symbols(), arena->nodes(),
                               arena->size())) {
      return pyb::cast(PyEquation{std::move(*snap)});
    }
  }
  pyb::object eq =
      pyb::cast(PyEquation{std::move(arena), std::move(roots)});
  if (cache) {
    // A cache that cannot be written is still a working equation.
    (void)rt::save(eq.cast<PyEquation &>().snapshot(), *cache);
  }
  return eq;
}

// An equation from a file and nothing else: no model runs and nothing is
// swept.  The output count is a number here, so unlike the C++ side there is
// nothing to state and nothing to refuse.
[[nodiscard]] pyb::object load_equation(const std::filesystem::path &path) {
  return pyb::cast(PyEquation{unwrap(rt::load_snapshot<double>(path))});
}

// The whole of ddx.equation: install an arena, run the model in it, take what
// comes back -- rt::equation(assemble)'s three steps, with the output count a
// number rather than a template parameter.
[[nodiscard]] pyb::object make_equation(const pyb::object &model,
                                        const std::optional<std::filesystem::path> &cache) {
  auto arena = std::make_shared<rt::Builder<double>>();
  pyb::object built;
  {
    // RAII, and load-bearing: if the model raises, pybind11 unwinds through
    // here as py::error_already_set and this is what puts the previous arena
    // back.  Without it a raising model would poison every one after it.
    const impl::scoped_value scope{current_arena, arena};
    built = model();
  }

  auto roots = roots_of(built, *arena);
  pyb::object eq = built_or_loaded(cache, std::move(arena), std::move(roots));

  // What functools.wraps would do, without a Python wrapper to do it: the
  // equation answers help() and repr() with the model's own name and docstring.
  static constexpr std::array kWrapped{"__name__", "__qualname__", "__doc__",
                                       "__module__"};
  for (const char *attr :
       kWrapped | std::views::filter([&model](const char *a) {
         return pyb::hasattr(model, a);
       })) {
    eq.attr(attr) = model.attr(attr);
  }
  return eq;
}

} // namespace
} // namespace ddx::py

PYBIND11_MODULE(_ddx, m) {
  namespace pyb = pybind11;
  using namespace ddx;
  using ddx::py::PyEquation;
  using ddx::py::PyExpression;

  m.doc() = "ddx's runtime expression graph and LLVM JIT";

  // Whether a kernel can ever land.  Not a shape: Options, Backend and
  // wait_for_kernel are here either way -- what a build without the backend
  // takes away is the LLJIT, so nothing compiles and the sweep answers.
#ifdef DDX_HAS_JIT
  m.attr("has_jit") = true;
#else
  m.attr("has_jit") = false;
#endif

  // The class first, then a translator that puts the code on the instance --
  // register_exception would install one that carries only the message.
  ddx::py::error_class =
      pyb::exception<ddx::py::PyError>(m, "Error", PyExc_RuntimeError)
          .release();
  pyb::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const ddx::py::PyError &e) {
      pyb::object raised = ddx::py::error_class(e.what());
      raised.attr("code") = pyb::cast(e.code);
      PyErr_SetObject(ddx::py::error_class.ptr(), raised.ptr());
    }
  });

  // Off the library's own table, so an errc added there reaches Python without
  // this file changing, each member carrying that table's text as its
  // docstring.  The literal rather than detail::kMessages[...].data(): a
  // string_view need not be null-terminated.
  pyb::native_enum<errc> errors(m, "errc", "enum.IntEnum",
                                "Why ddx refused; Error.code carries one.");
#define DDX_PY_ERRC(name, text) errors.value(#name, errc::name, text);
  DDX_ERRC_TABLE(DDX_PY_ERRC)
#undef DDX_PY_ERRC
  errors.finalize();

  // Each operator twice: once against another expression, once against a plain
  // number.  implicitly_convertible already makes the second work at run time,
  // but only a real overload puts `float` in the signature -- and the signature
  // is what reaches the generated stubs, so without it a type checker rejects
  // `x * 2.0` while the interpreter accepts it.
#define DDX_PY_BINOP(name, body)                                               \
  def(                                                                         \
      name, [](const PyExpression &l, const PyExpression &r) { return body; }, \
      pyb::is_operator())                                                      \
      .def(                                                                    \
          name,                                                                \
          [](const PyExpression &l, double v) {                                \
            const PyExpression r{v};                                           \
            return body;                                                       \
          },                                                                   \
          pyb::is_operator())

  pyb::class_<PyExpression>(m, "Expression")
      .def(pyb::init<double>(), pyb::arg("value"))
      .def("__repr__", &PyExpression::repr)
      .DDX_PY_BINOP("__add__", l + r)
      .DDX_PY_BINOP("__radd__", r + l)
      .DDX_PY_BINOP("__sub__", l - r)
      .DDX_PY_BINOP("__rsub__", r - l)
      .DDX_PY_BINOP("__mul__", l * r)
      .DDX_PY_BINOP("__rmul__", r * l)
      .DDX_PY_BINOP("__truediv__", l / r)
      .DDX_PY_BINOP("__rtruediv__", r / l)
      .DDX_PY_BINOP("__pow__", ddx::py::pow(l, r))
      .DDX_PY_BINOP("__rpow__", ddx::py::pow(r, l))
      .def(
          "__neg__", [](const PyExpression &u) { return -u; },
          pyb::is_operator())
      .def(
          "__abs__", [](const PyExpression &u) { return ddx::py::abs(u); },
          pyb::is_operator());
#undef DDX_PY_BINOP

  pyb::implicitly_convertible<double, PyExpression>();

  // After the class, not before: a signature naming a type pybind11 has not
  // been told about yet renders as the raw C++ spelling, which is what reaches
  // the generated stubs.
  m.def("equation", &ddx::py::make_equation, pyb::arg("model"),
        pyb::kw_only(), pyb::arg("cache") = pyb::none());
  m.def("load", &ddx::py::load_equation, pyb::arg("path"));
  m.def("var", &ddx::py::make_var, pyb::arg("name"));

#define DDX_PY_DEF_UN(fn, Op, label, ...)                                      \
  m.def(                                                                       \
      #fn, [](const PyExpression &u) { return ddx::py::fn(u); },               \
      pyb::arg("x"));
  DDX_UNARY_MATH_TABLE(DDX_PY_DEF_UN)
  DDX_RT_UNARY_TABLE(DDX_PY_DEF_UN)
#undef DDX_PY_DEF_UN

#define DDX_PY_DEF_BIN(fn, Op, label, ...)                                     \
  m.def(                                                                       \
      #fn,                                                                     \
      [](const PyExpression &l, const PyExpression &r) {                       \
        return ddx::py::fn(l, r);                                              \
      },                                                                       \
      pyb::arg("x"), pyb::arg("y"));
  DDX_RT_BINARY_TABLE(DDX_PY_DEF_BIN)
#undef DDX_PY_DEF_BIN

  pyb::native_enum<jit::Backend>(m, "Backend", "enum.IntEnum",
                                 "Whether an equation compiles its graph.")
      .value("INTERPRET", jit::Backend::Interpret)
      .value("COMPILE", jit::Backend::Compile)
      .value("ADAPT", jit::Backend::Adapt)
      .finalize();

  pyb::native_enum<jit::VecLib>(m, "VecLib", "enum.IntEnum",
                                "Which vector math library a lane may call.")
      .value("NONE", jit::VecLib::None)
      .value("AUTO", jit::VecLib::Auto)
      .value("LIBMVEC", jit::VecLib::Libmvec)
      .finalize();

  // Internal.  ddx.Options is the pydantic model over this, which is where the
  // ranges are checked; the defaults stay here, because they follow how the
  // library was built.
  pyb::class_<jit::Options>(m, "_Options")
      .def(pyb::init<>())
      .def_readwrite("backend", &jit::Options::backend)
      .def_readwrite("points", &jit::Options::points)
      .def_readwrite("lanes", &jit::Options::lanes)
      .def_readwrite("opt_level", &jit::Options::opt_level)
      .def_readwrite("codegen_level", &jit::Options::codegen_level)
      .def_readwrite("warm_points", &jit::Options::warm_points)
      .def_readwrite("hot_points", &jit::Options::hot_points)
      .def_readwrite("slp", &jit::Options::slp)
      .def_readwrite("loop_vectorize", &jit::Options::loop_vectorize)
      .def_readwrite("veclib", &jit::Options::veclib)
      .def_readwrite("contract", &jit::Options::contract)
      .def_readwrite("time_passes", &jit::Options::time_passes)
      // Policy rather than identity: neither changes what a compile emits, so
      // neither belongs in a key that decides whether stored code may be run.
      .def_readwrite("retain_object", &jit::Options::retain_object)
      .def_readwrite("cache_dir", &jit::Options::cache_dir)
      .def(pyb::self == pyb::self);

  pyb::class_<PyEquation>(m, "Equation", pyb::dynamic_attr())
      .def("__repr__", &PyEquation::repr)
      .def("__call__", &PyEquation::evaluate, pyb::arg("x"))
      .def("evaluate", &PyEquation::evaluate, pyb::arg("x"))
      .def("jacobian", &PyEquation::jacobian, pyb::arg("x"))
      .def("hessian", &PyEquation::hessian, pyb::arg("x"))
      .def("to_dot", &PyEquation::to_dot, pyb::kw_only(),
           pyb::arg("all") = false)
      .def("save", &PyEquation::save, pyb::arg("path"))
      .def("verify", &PyEquation::verify, pyb::arg("path"))
      .def_property_readonly("loaded", &PyEquation::loaded)
      .def_property_readonly("symbols", &PyEquation::symbols)
      .def_property_readonly("arity", &PyEquation::arity)
      .def_property_readonly("outputs", &PyEquation::outputs)
      .def_property_readonly("uses_kernel", &PyEquation::uses_kernel)
      .def_property_readonly("hessian_colors", &PyEquation::hessian_colors)
      .def("wait_for_kernel", &PyEquation::wait_for_kernel)
      .def_property("_options", &PyEquation::options, &PyEquation::set_options);
}
