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
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
// than the one rt::var(name) reads, because what has to survive here is the
// shared ownership, not a raw pointer -- which also means every symbol below
// goes through the plain two-argument rt::var, and nothing reaches into rt's
// own thread-local.
thread_local std::shared_ptr<rt::Builder<double>> current_arena;

// The exception type Python sees.  Deliberately leaked: it is a type object
// that lives as long as the module, and a translator is a plain function
// pointer, so it cannot capture.
pyb::handle error_class;

[[nodiscard]] PyExpression make_var(std::string_view name) {
  if (!current_arena) {
    // C++ carries a poison instead, because it cannot afford a branch per
    // symbol.  Python already pays far more per call than the branch costs, and
    // an error where the mistake is beats one at the end of the model.
    fail_with(errc::no_arena);
  }
  return {rt::var(*current_arena, name), current_arena};
}

// One root, or m of them from a tuple or list.  The count is a runtime value
// throughout, which is the whole reason these bindings sit below impl::Equation.
[[nodiscard]] std::vector<rt::NodeId>
roots_of(const pyb::object &built, rt::Builder<double> &arena) {
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
           if (e.arena() && e.arena().get() != &arena) {
             // Its ids mean nothing here, and id() would hand back one that
             // indexes this arena at random rather than refusing.
             fail_with(errc::no_graph);
           }
           return e.id(arena);
         }) |
         impl::to<std::vector<rt::NodeId>>();
}

// The whole of ddx.equation: install an arena, run the model in it, take what
// comes back.  The same three steps rt::equation(assemble) makes, with the
// output count a number rather than a template parameter.
[[nodiscard]] pyb::object make_equation(const pyb::object &model) {
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
  pyb::object eq =
      pyb::cast(PyEquation{std::move(arena), std::move(roots)});

  // What functools.wraps would do, without a Python wrapper to do it: the
  // equation answers help() and repr() with the model's own name and docstring.
  for (const char *attr : {"__name__", "__qualname__", "__doc__", "__module__"}) {
    if (pyb::hasattr(model, attr)) {
      eq.attr(attr) = model.attr(attr);
    }
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
  // wait_for_kernel are here either way, because jit::Options and jit::Kernel
  // are header types -- what a build without the backend takes away is the
  // LLJIT, so nothing compiles and the sweep answers everything.
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
  // this file changing -- and each member carries that table's text as its
  // docstring.  The literal rather than detail::kMessages[...] .data(): a
  // string_view is not required to be null-terminated, and this one only is
  // because of how it happens to be built.
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
  def(name, [](const PyExpression &l, const PyExpression &r) { return body; },  \
      pyb::is_operator())                                                      \
      .def(name, [](const PyExpression &l, double v) {                         \
        const PyExpression r{v};                                               \
        return body;                                                           \
      }, pyb::is_operator())

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
      .def("__neg__", [](const PyExpression &u) { return -u; }, pyb::is_operator())
      .def("__abs__", [](const PyExpression &u) { return ddx::py::abs(u); },
           pyb::is_operator());
#undef DDX_PY_BINOP

  pyb::implicitly_convertible<double, PyExpression>();

  // After the class, not before: a signature naming a type pybind11 has not
  // been told about yet renders as the raw C++ spelling, which is what reaches
  // the generated stubs.
  m.def("equation", &ddx::py::make_equation, pyb::arg("model"));
  m.def("var", &ddx::py::make_var, pyb::arg("name"));

  // The free functions, off the same tables expression.hpp generated them with:
  // an opcode added to the library reaches Python without this file changing.
#define DDX_PY_DEF_UN(fn, Op, label, ...)                                      \
  m.def(#fn, [](const PyExpression &u) { return ddx::py::fn(u); }, pyb::arg("x"));
  DDX_UNARY_MATH_TABLE(DDX_PY_DEF_UN)
  DDX_RT_UNARY_TABLE(DDX_PY_DEF_UN)
#undef DDX_PY_DEF_UN

#define DDX_PY_DEF_BIN(fn, Op, label, ...)                                     \
  m.def(#fn, [](const PyExpression &l, const PyExpression &r) { return ddx::py::fn(l, r); }, \
        pyb::arg("x"), pyb::arg("y"));
  DDX_RT_BINARY_TABLE(DDX_PY_DEF_BIN)
#undef DDX_PY_DEF_BIN

  pyb::native_enum<jit::Backend>(m, "Backend", "enum.IntEnum",
                                 "Whether an equation compiles its graph.")
      .value("INTERPRET", jit::Backend::Interpret)
      .value("COMPILE", jit::Backend::Compile)
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
      .def_readwrite("slp", &jit::Options::slp)
      .def_readwrite("loop_vectorize", &jit::Options::loop_vectorize)
      .def_readwrite("veclib", &jit::Options::veclib)
      .def_readwrite("contract", &jit::Options::contract)
      .def_readwrite("time_passes", &jit::Options::time_passes)
      .def(pyb::self == pyb::self);

  pyb::class_<PyEquation>(m, "Equation", pyb::dynamic_attr())
      .def("__repr__", &PyEquation::repr)
      .def("__call__", &PyEquation::evaluate, pyb::arg("x"))
      .def("evaluate", &PyEquation::evaluate, pyb::arg("x"))
      .def("jacobian", &PyEquation::jacobian, pyb::arg("x"))
      .def("hessian", &PyEquation::hessian, pyb::arg("x"))
      .def("to_dot", &PyEquation::to_dot, pyb::kw_only(), pyb::arg("all") = false)
      .def_property_readonly("symbols", &PyEquation::symbols)
      .def_property_readonly("arity", &PyEquation::arity)
      .def_property_readonly("outputs", &PyEquation::outputs)
      .def_property_readonly("uses_kernel", &PyEquation::uses_kernel)
      .def_property_readonly("hessian_colors", &PyEquation::hessian_colors)
      .def("wait_for_kernel", &PyEquation::wait_for_kernel)
      .def_property("_options", &PyEquation::options, &PyEquation::set_options)
      ;
}
