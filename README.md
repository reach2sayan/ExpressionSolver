[![CMake](https://github.com/reach2sayan/ddx/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/reach2sayan/ddx/actions/workflows/cmake-multi-platform.yml)
[![C++](https://img.shields.io/badge/C++-%2300599C.svg?logo=c%2B%2B&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3776AB.svg?logo=python&logoColor=white)](#python)

# ddx

A C++23 library for differentiating expressions. Build a function over named
symbols while the program runs — terms looped over a data file, a model read
from configuration — and ask it for values, gradients, Jacobians and Hessians.
One point at a time, or a batch of thousands in a single call, interpreted or
compiled to machine code through LLVM. There are [Python bindings](#python)
over the same runtime.

```cpp
#include "ddx.hpp"
#include "rt/equation.hpp"
using namespace ddx;

struct Sample { double t, y; };
const std::vector<Sample> data{{0.0, 2.00}, {1.0, 1.21}, {2.0, 0.74}, {3.0, 0.45}};

// residual sum of squares of the fit m(t) = a·exp(−b·t)
const auto eq = rt::equation([&] {
  const auto a = rt::var("a");
  const auto b = rt::var("b");

  rt::RTExpression<double> rss = 0.0;
  for (const auto &s : data) {
    const auto r = s.y - a * exp(-b * s.t);
    rss += r * r;
  }
  return rss;
});

*eq.evaluate(2.0, 0.5);     // 4.1343959887e-05
*eq.jacobian(2.0, 0.5);     // {-0.0010757424682, 0.0150678475595}
*eq.hessian(2.0, 0.5);      // 2 × 2, row-major
```

**Contents** — [Requirements](#requirements) · [Using it](#using-it) ·
[Building](#building) · [Equations](#equations) · [Expressions](#expressions) ·
[Points](#points) · [Values and derivatives](#values-and-derivatives) ·
[Batches](#batches) · [Errors](#errors) ·
[Ownership and threads](#ownership-and-threads) · [The JIT](#the-jit) ·
[Reference](#reference) · [Python](#python) ·
[Compile-time expressions](#compile-time-expressions) · [Printing](#printing)

---

## Requirements

- A C++23 compiler and standard library: **GCC 14+**, **Clang 19+** over
  libstdc++ 14+, **Clang 17+** over libc++ 17+, or MSVC (VS 2022,
  `/std:c++latest`).
- **CMake 3.26+**.
- **Boost's headers**, on the machine that builds ddx and on any machine that
  compiles against it — `libboost-dev`, `boost-devel`, `brew install boost`.
  All of it is header-only, so no compiled Boost library is ever linked.
- GoogleTest and Google Benchmark are fetched at configure time, so a first
  configure that builds the tests or the benchmarks wants a network.
- `-DDDX_BUILD_JIT=ON` additionally needs an LLVM 20 installation, pointed at
  with `LLVM_DIR`.
- `-DDDX_BUILD_PYTHON=ON` additionally needs **Python 3.11+** and pybind11;
  the module itself imports NumPy 1.23+ and pydantic 2.7+.

## Using it

In a project that vendors ddx:

```cmake
add_subdirectory(ddx)
target_link_libraries(my_app PRIVATE ddx::rt)
```

Or against an installed or built copy:

```cmake
find_package(ddx CONFIG REQUIRED COMPONENTS rt jit)
target_link_libraries(my_app PRIVATE ddx::jit)
```

```sh
cmake --install build/release_jit --prefix /opt/ddx
```

A build tree is consumable without installing:

```cmake
find_package(ddx CONFIG REQUIRED PATHS /path/to/ddx/build/release NO_DEFAULT_PATH)
```

Asking for a component the build does not have fails at configure time and says
which option turns it on.

### Targets

| Target | Is |
|---|---|
| `ddx::ddx` | the header-only library — expressions, `Equation`, forward mode |
| `ddx::rt` | that plus runtime expressions |
| `ddx::jit` | `ddx::rt` with the LLVM backend compiled in |

Components for `find_package`: `ddx`, `util`, `ops`, `md`, `symbolic`, `dual`,
`rt`, `jit`.

## Building

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Or through the presets:

| Preset | Build type | JIT |
|---|---|---|
| `debug` | Debug | — |
| `release` | Release | — |
| `debug_jit` | Debug | yes |
| `release_jit` | Release | yes |
| `debug_jit_tsan` | Debug | yes — under ThreadSanitizer |
| `python` | Release | yes — and the extension module |
| `python_no_jit` | Release | — the extension module alone |

```sh
cmake --preset release_jit
cmake --build --preset release_jit
ctest --preset release_jit
```

The JIT presets point `LLVM_DIR` at the Debian/Ubuntu `llvm-20` layout;
override it on the command line, which wins over the preset:

```sh
cmake --preset release_jit -DLLVM_DIR=/opt/llvm-20/lib/cmake/llvm
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `DDX_BUILD_JIT` | `OFF` | compile the LLVM backend into the library |
| `DDX_SHARED_LIBS` | `ON` | build the library shared rather than static |
| `DDX_BUILD_PYTHON` | `OFF` | build the pybind11 extension module |
| `DDX_BUILD_BENCHMARKS` | `ON` | build the benchmark targets |
| `DDX_SANITIZE` | `off` | `thread`, `address` or `undefined` — instrument the build |
| `DDX_INSTALL` | on if top-level | generate the install and `find_package` rules |
| `ENABLE_NATIVE_ARCH` | `ON` | `-march=native`, falling back to `x86-64-v3` |
| `DDX_FP_FLAGS` | `ON` | `-ffp-contract=fast -fno-math-errno` |
| `DDX_MDSPAN_MODE` | `auto` | `auto` / `std` / `vendored` — which `mdspan` to bind to |
| `DDX_DEDUCING_THIS` | `auto` | `auto` / `on` / `off` — accessor spelling (P0847) |

`-ffast-math` is not used and not recommended: it changes derivative values.

ddx's own targets compile `-fno-exceptions`. Nothing in the library throws —
failures come back as values — so what you compile your own code with is your
business.

---

## Equations

```cpp
#include "ddx.hpp"
#include "rt/equation.hpp"
using namespace ddx;
```

Link `ddx::rt`, or `ddx::jit` for the compiled batch calls.

`rt::equation` takes a callback, runs it, and hands back the equation. Symbols
are named inside the callback with `rt::var(name)` — a string, not a template
argument — and the callback returns the expression it built:

```cpp
const auto eq = rt::equation([] {
  const auto x = rt::var("x");
  const auto y = rt::var("y");
  return exp(x) * sin(y);
});
```

Return a `std::array` of expressions for a system, f: ℝⁿ → ℝᵐ:

```cpp
const auto sys = rt::equation([] {
  const auto x = rt::var("x");
  const auto y = rt::var("y");
  return std::array{x * y + sin(x), exp(x) - y * y};
});

*sys.jacobian(1.3, 0.7);      // row-major, m × n
```

Another scalar type is a template argument on both: `rt::var<T>(name)` inside
`rt::equation<T>(...)`.

Symbols are ordered canonically — **alphabetically by name** — which is the
order positional points are read in and the order every result is laid out in.
`symbols()` lists them in that order, `arity()` counts them:

```cpp
*eq.arity();                  // 2
*eq.symbols();                // {"x", "y"}
```

## Expressions

`rt::RTExpression<T>` is the expression type. A bare number converts to one, so
an accumulator starts at `0.0` and a scalar mixes into arithmetic without
wrapping:

```cpp
rt::RTExpression<double> acc = 0.0;
for (const auto &s : data) {
  acc += (s.y - a * exp(-b * s.t)) * 2.0;
}
```

| Kind | Available |
|---|---|
| Arithmetic | `+` `-` `*` `/` unary `-` `+=` `-=` `*=` `/=` |
| Unary | `sin` `cos` `tan` `exp` `log` `log10` `sqrt` `cbrt` `abs` `sign` `asin` `acos` `atan` `sinh` `cosh` `tanh` `asinh` `acosh` `atanh` `erf` |
| Binary | `pow` `atan2` `hypot` `max` `min` |

They are found by argument-dependent lookup, so they work whether or not `rt`
is in scope — the compound forms are members, and rebind the handle rather than
mutating a node. Identical subexpressions are shared as they are built, and
literal arithmetic folds before it reaches the graph.

## Points

Every call that needs numbers takes them in four interchangeable spellings:

```cpp
*eq.jacobian(1.3, 0.7);                             // positional
*eq.jacobian(std::array{1.3, 0.7});                 // any range
*eq.jacobian(named<"y">(0.7), named<"x">(1.3));     // by name, any order
*eq.jacobian("y"_s = 0.7, "x"_s = 1.3);             // by name, assignment spelling
```

**Positional order is alphabetical by symbol name**, not the order the symbols
appear in the expression.

`point(args…)` builds the point vector on its own, in canonical order, so a
point assembled once can be reused:

```cpp
const auto at = *eq.point(named<"y">(0.7), named<"x">(1.3));
*eq.evaluate(at);
```

## Values and derivatives

Every per-point call answers `result<T>` — `std::expected<T, ddx::error>`. The
`*` in these examples is that check skipped.

| Call | Answers |
|---|---|
| `evaluate(point)` | `result<T>`, or `result<std::vector<T>>` for a system |
| `jacobian(point)` | `result<std::vector<T>>`, row-major m × n — so `n` long when m == 1, which is the gradient |
| `hessian(point)` | `result<std::vector<T>>`, dense row-major m × n × n |
| `univariate_derivative<K>(x0)` | `result<T>` — the K-th derivative, one symbol and one output only |

```cpp
const auto g = *eq.jacobian(2.0, 0.5);     // {∂f/∂a, ∂f/∂b}
const auto H = *eq.hessian(2.0, 0.5);      // H[i * n + j]
H[0];                                      //  3.1060035856
H[1];                                      // -3.1441109272
```

```cpp
const auto in_b = rt::equation([] { return sin(rt::var("b")); });
*in_b.univariate_derivative<4>(0.5);       //  sin⁗(0.5)
```

Values, gradient and Hessian are three separate graphs, each built the first
time it is asked for. A caller who only ever evaluates never builds a gradient.

## Batches

The batch calls take columns: one input column per symbol, one output pointer
per output column, every column `n` points long. They fill the outputs in place
and answer `result<void>`.

```cpp
result<void> evaluate(xs, values, n);
result<void> jacobian(xs, values, partials, n);
result<void> hessian (xs, values, partials, hessians, n);
```

Ask the equation how many output columns each block wants, and size the buffers
by that:

```cpp
const std::vector<double> as{1.8, 1.9, 2.0, 2.1};        // candidate a values
const std::vector<double> bs{0.45, 0.48, 0.50, 0.52};    // candidate b values
const std::size_t n = as.size();

std::vector<double> f(n * *eq.value_columns());
std::vector<double> j(n * *eq.jacobian_columns());

const std::vector<const double *> xs{as.data(), bs.data()};
const std::vector<double *> values{f.data()};
const std::vector<double *> partials{j.data(), j.data() + n};

if (const auto ok = eq.jacobian(xs, values, partials, n); !ok) {
  std::println("{}", ok.error());
}

f[2];                    // 4.1343959887e-05  — RSS at (2.0, 0.50)
j[2];                    // ∂RSS/∂a there
j[n + 2];                // ∂RSS/∂b there
```

A column count that does not match answers `errc::wrong_column_count`, and
nothing is written.

The Hessian takes a fourth block. Its columns are compressed by colour, so
`hessian_columns()` is what sizes them — asking builds nothing:

```cpp
std::vector<double> h(n * *eq.hessian_columns());
std::vector<double *> hessians;
for (std::size_t k = 0; k < *eq.hessian_columns(); ++k) {
  hessians.push_back(h.data() + k * n);
}

const auto ok = eq.hessian(xs, values, partials, hessians, n);
```

`evaluate` has its own graph rather than the Jacobian's with the partials
thrown away: columns nobody asks for are work nobody does.

## Errors

`rt::equation` always hands back an `Equation`. When the callback could not
produce one — a symbol named with no arena, an expression naming no graph — that
equation is *poisoned*: it carries the error rather than a graph, and every call
on it answers with that error.

```cpp
const auto eq = rt::equation([] { return rt::var("x") * 2.0; });

if (const auto bad = eq.status()) {
  std::println("{}", *bad);      // errc::no_arena
  return;
}
```

`status()` is a `std::optional<ddx::error>`; `poisoned()` is the same question
without the reason.

Calls answer `result<T>` on top of that: the symbol list exists only at run
time, so an arity mismatch or an unknown name is a genuine runtime failure.

```cpp
const auto j = eq.jacobian(1.0, 2.0);      // two values, one symbol
if (!j) {
  std::println("{}", j.error());  // errc::wrong_arity
}
```

| `errc` | Means |
|---|---|
| `no_arena` | a symbol was named outside an `rt::equation` callback |
| `no_graph` | the expression names no graph — a bare literal, for instance |
| `wrong_arity` | the point does not supply one value per symbol |
| `unknown_symbol` | a named point uses a name the equation does not have |
| `short_point` | a range point is shorter than the symbol list |
| `wrong_column_count` | a batch block has the wrong number of columns |
| `not_univariate` | `univariate_derivative` on more than one symbol |
| `jit_target`, `jit_module`, `jit_verify`, `jit_lookup` | the JIT could not compile the graph |

An `errc` and an `error` both format and stream as their text, so
`std::println("{}", j.error())` is the whole of reporting one. Nothing here
throws.

The accessors that answer no `result` answer `std::optional` instead —
`arity()`, `symbols()`, `value_columns()`, `jacobian_columns()`,
`hessian_columns()` and `hessian_colors()` are `nullopt` on a poisoned
equation. A bare count could not say it: an equation over a literal-only graph
legitimately has no symbols and no output columns.

## Ownership and threads

**The equation owns everything it needs.** It is move-only: return it from a
function, store it in a class, keep it as long as you like.

```cpp
auto make_model(const std::vector<Sample> &data) {
  return rt::equation([&] {
    const auto a = rt::var("a");
    const auto b = rt::var("b");
    rt::RTExpression<double> rss = 0.0;
    for (const auto &s : data) {
      const auto r = s.y - a * exp(-b * s.t);
      rss += r * r;
    }
    return rss;
  });
}

const auto eq = make_model(load());     // `data` is gone; `eq` is complete
```

**Symbols belong to the callback.** `rt::var` is meaningful while the callback
that named it runs. Build the expression there and return it; do not store an
`RTExpression` and use it afterwards. A symbol named outside a callback reaches
`rt::equation` with no arena behind it, and the equation that comes back is
poisoned with `errc::no_arena`.

**Batch columns are yours.** The equation writes through the pointers you give
it and keeps none of them; every buffer must be alive for the duration of the
call, and each column must hold at least `n` elements.

Building is independent per thread: two threads may run `rt::equation`
callbacks at the same time. An equation fills internal state on the first call
of each kind — make that first call before handing one to several threads, or
give each thread its own. A batch call splits across threads by slicing the
columns, each thread getting its own offset pointers and its own `n`:

```cpp
const std::size_t chunk = (n + threads - 1) / threads;
std::vector<std::jthread> pool;
for (std::size_t t = 0; t < n; t += chunk) {
  const std::size_t m = std::min(chunk, n - t);
  pool.emplace_back([&, t, m] {
    const std::vector<const double *> xs{as.data() + t, bs.data() + t};
    const std::vector<double *> values{f.data() + t};
    const std::vector<double *> partials{j.data() + t, j.data() + n + t};
    (void)eq.jacobian(xs, values, partials, m);
  });
}
```

## The JIT

Configure with `-DDDX_BUILD_JIT=ON` and link `ddx::jit`, and the batch calls can
run compiled code. The spellings do not change, the answers do not change, and
there is no compiler or kernel object to hold.

Nothing compiles unless you ask. An equation left alone interprets, which costs
no compiler and runs within ~1.4x of a kernel. Asking is what starts the build,
so it overlaps whatever you do next, and **no call ever waits for it**:

```cpp
auto eq = rt::equation([] {
  const auto x = rt::var("x");
  const auto y = rt::var("y");
  return x * log(x) + y * exp(x * y);
});

eq.options({.backend = rt::Backend::Compile, .points = 4096});
// Starts the gradient graph's compile here and now.  Returns immediately;
// nothing has been compiled yet.

eq.jacobian(xs, f, g, n);      // compile in flight -> interpreted
// ... the compile lands about here ...
eq.jacobian(xs, f, g, n);      // kernel, and from here on
eq.uses_kernel();              // true
```

Values, gradient and Hessian compile separately, each launched the moment its
graph is built — so a caller who only evaluates never compiles a gradient.

Results either side of the switchover agree **to the bit**. A multiply feeding
an add is one rounding in both, because the contraction is decided in the graph
rather than by the backend: the sweep calls `std::fma` where the kernel emits
`llvm.fma`. A loop running across the switchover sees no movement at all.

`wait_for_kernel()` is the only call that blocks, and only because it was asked
to:

```cpp
eq.wait_for_kernel();          // true once it lands
eq.jacobian(xs, f, g, n);      // kernel, guaranteed
```

Setting `Options` to the value it already holds does nothing. Changing it
discards every graph and abandons a compile still in flight rather than waiting
for it — including a change back to `Backend::Interpret`, after which no
compiler is asked at all.

### `jit::Options`

| Field | Default | Is |
|---|---|---|
| `backend` | `Interpret` | `Interpret` or `Compile`; choosing `Compile` starts the build |
| `points` | `1` | the batch you intend to hand one call — stated, since the kernel is built before any call exists to infer it from |
| `lanes` | `0` | points per loop iteration; `0` derives it from `points`, `1` is scalar |
| `opt_level` | follows the build type | LLVM's IR pipeline, 0–3 — 3 in a Release build, 1 in a Debug one |
| `codegen_level` | `1` | LLVM's codegen, 0–3 — the knob that trades kernel speed for compile time |
| `slp` | `false` | SLP vectorisation: packs subexpressions within one point |
| `loop_vectorize` | `false` | loop vectorisation, on a loop already emitted `lanes` wide |
| `veclib` | `None` | vector math library for transcendentals; `Libmvec` trades ~0.5 ULP for ~4 |
| `contract` | follows `DDX_FP_FLAGS` | fold a multiply feeding an add into one rounding, in both the kernel and the sweep |
| `time_passes` | `false` | per-pass timing to stderr |

`points` decides the lane width and nothing else: a call carrying some other
number is answered correctly, just not by the kernel that number would have
built. At one point a scalar kernel is 1.2x to 4.1x quicker than the host's
vector width, and over a batch the reverse by 3x.

```cpp
eq.options({.backend = rt::Backend::Compile, .points = 1});     // a gradient per step
eq.options({.backend = rt::Backend::Compile, .points = 4096});  // a batch at a time
```

The compile is paid once and nothing waits for it: calls are interpreted until
it lands. Codegen — instruction selection and register allocation — is 68–92% of
it, which is why `codegen_level` and not `opt_level` is the knob that moves a
compile time.

## Saving a built equation

Building an equation sweeps it: one reverse pass per function for the Jacobian,
and for the Hessian a coupling pattern that is quadratic in the symbols followed
by a colouring. `save` writes that work to a file, and `load` reads it back
without a model:

```cpp
const auto eq = rt::equation([] { return exp(rt::var("x")) * rt::var("y"); });
eq.save("f.ddx");

const auto same = *rt::load("f.ddx");            // no model, nothing swept
const auto sys  = *rt::load<double, 3>("s.ddx"); // three functions
```

The output count is a template parameter because it is part of the type, and a
file holding some other number is refused rather than adapted to.

Pair a model with a file and it becomes a cache — the lambda is the model as it
would be written anyway, and the path is only where the result is kept:

```cpp
const auto eq = rt::equation("f.ddx", [] {
  const auto x = rt::var("x");
  return exp(x) * x;
});
```

The first run builds, sweeps and writes the file. A later one rebuilds the arena
— interning, which is cheap — finds the file still describes it, and takes the
sweeps off disk. Editing the model changes the key, so the file is rebuilt and
overwritten rather than trusted. It never refuses: an absent file is the first
run and a stale one is a rebuild, and `loaded()` says which happened.

Two questions about a file, and they are different:

```cpp
rt::verify("f.ddx");     // result<void> — can this build read it at all?
eq.verify("f.ddx");      // result<void> — ...and is it *this* equation?
```

A file is binary, little-endian, and carries a checksum plus the schema of every
struct in it, so one written by a different version of ddx is refused rather than
misread. Nothing in it is trusted before the checksum and the structural
invariants have passed. What travels is the arena and the sweeps; a lane's
frozen graph does not, because freezing one is a single linear pass over an
arena the file already has.

Loading instead of sweeping is about 1.7×. Under `Interpret` — the default —
that is 1.7× off the whole cost of having an equation. Under a compiling
backend the sweeps are a few milliseconds against a compile of hundreds or
thousands — so there the saving worth having is the machine code, which the
file also carries:

```cpp
eq.options({.backend = rt::Backend::Compile, .retain_object = true});
eq.wait_for_kernel();
eq.save("f.ddx");                 // the kernel travels with the graph

const auto warm = *rt::load("f.ddx");
warm.uses_kernel();               // true, with nothing having compiled
```

Measured at 64, 128 and 256 variables: 38/79/179 ms to build and compile, against
0.40/0.63/1.12 ms to load — 96× to 160×, with emission, the pass pipeline and
codegen not run at all. `retain_object` is off by default, since a kernel does
not keep a megabyte of machine code alive on the chance that someone saves it;
an equation compiled without it saves its graph and no code.

A stored kernel is run only where the graph, the host and the options it was
emitted under all still agree. Anything else compiles instead — a mismatch is
answered by compiling, never by running the wrong code.

| Member | Answers |
|---|---|
| `save(path)` | `result<void>` — write this equation |
| `verify(path)` | `result<void>` — does `path` hold *this* equation? |
| `Equation::load(path)` | `result<Equation>` — read one, sweeping nothing |
| `loaded()` | whether this equation came off disk rather than being swept |
| `rt::verify<T>(path)` | `result<void>` — is `path` readable by this build? |

---

## Reference

`ddx::rt::equation(callback)` → `Equation`. All of it is `const` and
thread-safe except `options()`.

| Member | Answers |
|---|---|
| `poisoned()`, `status()` | whether the build failed, and with what |
| `arity()` | `std::optional<std::size_t>` — symbol count, `n` |
| `symbols()` | `std::optional<std::span<const std::string>>` — canonical order |
| `point(args…)` | `result<std::vector<T>>` — a point in canonical order |
| `evaluate(point)` | `result<T>`, or `result<std::vector<T>>` for a system |
| `jacobian(point)` | `result<std::vector<T>>`, row-major m × n |
| `hessian(point)` | `result<std::vector<T>>`, dense row-major m × n × n |
| `univariate_derivative<K>(x0)` | `result<T>`, one symbol and one output only |
| `evaluate(xs, f, n)` | `result<void>` — a batch of `n` |
| `jacobian(xs, f, g, n)` | `result<void>` |
| `hessian(xs, f, g, h, n)` | `result<void>`, one output only |
| `value_columns()`, `jacobian_columns()`, `hessian_columns()` | `std::optional<std::size_t>` — what sizes the batch buffers |
| `hessian_colors()` | `std::optional<std::size_t>` — colours in the Hessian's compression |
| `options(opts)`, `options()` | set or read the JIT options; setting returns `*this` |
| `uses_kernel()` | whether a batch call goes through compiled code |
| `wait_for_kernel()` | block until a compile in flight has landed |
| `save(path)`, `verify(path)` | write this equation; ask whether a file holds it |
| `load(path)`, `loaded()` | read one, sweeping nothing; whether this one was read |

---

## Python

The same runtime, as an extension module. Build it with a preset, or install the
package — scikit-build-core drives CMake, so `pip install .` configures and
builds a wheel in one step:

```sh
pip install .                       # wheel; turns the JIT on, so LLVM 20 is needed
cmake --preset python               # in-tree, JIT
cmake --preset python_no_jit        # in-tree, no LLVM
```

`ddx.has_jit` says whether the copy you have was built with the LLVM backend.
Everything below works either way; without it, calls interpret.

`equation` takes a model — a callable of no arguments returning one expression,
or a tuple of them for a system — as a call or a decorator:

```python
import ddx

@ddx.equation
def f():
    x = ddx.var("x")
    y = ddx.var("y")
    return ddx.exp(x) * ddx.sin(y)
```

Symbols are named inside the model with `ddx.var(name)`, and they order
alphabetically as they do in C++. `f.symbols` lists them, `f.arity` counts
them, `f.outputs` counts the model's outputs. Arithmetic operators work on
`Expression`, and a bare number mixes in; the free functions are the same set
the C++ side has.

### Points

A point is a sequence, or a dict keyed by symbol name. A 2-D array is a
**batch** — shape `(symbols, points)`, one row per symbol:

```python
f.jacobian([2.0, 3.0])                      # positional, alphabetical
f.jacobian({"x": 2.0, "y": 3.0})            # by name
f.jacobian(np.array([[2.0], [3.0]]))        # a batch of one
f.jacobian(np.array([[2.0, 2.5], [3.0, 3.5]]))   # a batch of two
```

### Values and derivatives

Each call returns everything up to and including what it names, so a gradient
never costs a second evaluation:

| Call | Returns |
|---|---|
| `evaluate(x)` | `f` |
| `jacobian(x)` | `(f, J)` |
| `hessian(x)` | `(f, J, H)` |

```python
value, gradient = f.jacobian([2.0, 3.0])
value, gradient, hessian = f.hessian([2.0, 3.0])
hessian.shape                               # (2, 2)
```

Shapes follow the point: a single point gives a scalar `f`, an `(n,)` gradient
and an `(n, n)` Hessian; a batch of `p` appends that axis, giving `(p,)`,
`(n, p)` and `(n, n, p)`. A system prepends its output axis. Unlike the C++
batch calls, the Hessian arrives **dense** — the colour compression is undone
on the way out, so nothing here needs `hessian_columns`.

### Errors

Python raises where C++ returns `result<T>`. `ddx.Error` is a `RuntimeError`
carrying the same code:

```python
try:
    f.jacobian({"z": 1.0})
except ddx.Error as e:
    print(e)                                # "no symbol of that name"
    e.code is ddx.errc.unknown_symbol       # True
```

`errc` is an `IntEnum`, so it compares and formats as its number — `e.code.name`
is the spelling. The codes are the ones in the [table above](#errors); the
exception's own message is the text.

### The JIT

`Options` is a frozen pydantic model of the same fields as `jit::Options`,
validated on the way in. `eq.options` reads and assigns it; `eq.compile()` sets
`backend=COMPILE`, waits for the kernel, and returns the equation, so a
configure-and-use reads in one line:

```python
f.compile(points=batch.shape[1]).jacobian(batch)
f.uses_kernel                               # True, once it has landed
f.options = ddx.Options(backend=ddx.Backend.INTERPRET)   # discards the kernel
```

`compile()` blocks by construction — it is `wait_for_kernel()` with the options
set first. Assigning `options` does not: calls interpret until the kernel lands
and switch over when it does, as in C++.

Equations save and load here too, over the same file format and the same
serialiser — a file written by C++ loads in Python and the other way round:

```python
eq.save("f.ddx")
same = ddx.load("f.ddx")          # no model runs, nothing is swept

@ddx.equation                      # or pair a model with a file, as a cache
def model() -> ddx.Expression:
    x, y = ddx.var("x"), ddx.var("y")
    return ddx.exp(x) * y

cached = ddx.equation(model, cache="f.ddx")
cached.loaded                      # False the first run, True after
```

`save`, `load` and `verify` raise `ddx.Error` rather than answering `False`:
unreadable, unloadable and "a different equation" are three different `errc`
values, and only the code says which.

### Reference

| Member | Is |
|---|---|
| `arity`, `outputs`, `symbols` | properties — symbol count, output count, canonical names |
| `evaluate(x)`, `__call__(x)` | `f` at the point or batch |
| `jacobian(x)` | `(f, J)` |
| `hessian(x)` | `(f, J, H)`, dense |
| `options` | property — read or assign an `Options` |
| `compile(**fields)` | set `Options`, block for the kernel, return self |
| `uses_kernel`, `wait_for_kernel()` | whether a call goes through compiled code, and blocking for it |
| `hessian_colors` | colours in the Hessian's compression |
| `to_dot(*, all=False)` | the graph in Graphviz form; `all=True` draws the pruned nodes too |
| `save(path)`, `verify(path)` | write this equation; raise unless `path` holds it |
| `loaded` | property — whether this equation was read rather than built |

`ddx.load(path)` reads one, and `ddx.equation(model, cache=path)` builds or reads
as the file allows.

---

## Compile-time expressions

The same derivatives are available on expressions whose shape is known when you
compile, header-only and `constexpr` throughout — `#include "ddx.hpp"`, link
`ddx::ddx`. Symbols are named by template argument, and `Equation` carries the
whole API:

```cpp
constexpr auto x = var<"x">;
constexpr auto y = var<"y">;

constexpr auto eq = Equation{x * y + 2.0 * x};
constexpr auto g = eq.jacobian(std::array{2.0, 3.0});
static_assert(g[0] == 5.0);        // ∂f/∂x
static_assert(g[1] == 2.0);        // ∂f/∂y
```

Points come in the same four spellings, and results are `std::array` or a
tensor rather than `result<std::vector<T>>` — the shapes are in the type, so
nothing has to be checked at run time.

| Member | Answers |
|---|---|
| `evaluate(point)` | f at the point |
| `jacobian(point)` | J, `J[i, j] = ∂fᵢ/∂xⱼ`; ∇f itself when m == 1 |
| `hessian(point)` | ∇²f — on symbols declared `var<"x", dual>` |
| `derivative_tensor<K>(point)` | all K-th order partials, any `K ≥ 1` |
| `univariate_derivative<K>(x0)` | the K-th derivative of a one-symbol function |

```cpp
constexpr auto H = Equation{var<"a", dual> * var<"b", dual>}.hessian(2.0, 3.0);
H[0, 1];                                          // 1.0

Equation{x * y}.derivative_tensor<3>(1.0, 2.0);   // rank 3
```

`record(named<"n">(3), named<"x">(1.5))` collects keyword arguments into a
value with the keys in its type, readable with `m.get<"n">()` or `m["x"_s]`.

## Printing

Compile-time expressions and `Equation`s are formattable and streamable:

```cpp
std::format("{}", x * y + sin(x));      // "x * y + sin(x)"
std::format("{::.3f}", 2.0 * x);        // "2.000 * x" — the spec applies to every number
std::cout << (x - y * x) << '\n';       // "x - y * x"
```

An `Equation` prints each output with its derivative row underneath. A symbol
held constant for one partial prints with a `_c` suffix:

```
f0: x * y
  grad: y_c, x_c
```

Slot 0 of an `Equation` is the function and slot `k > 0` is ∂f/∂xₖ in canonical
order, each an expression you can print or store:

```cpp
auto eq = Equation{x * y};

std::format("{}", eq[idx<0>()]);        // "x * y"
std::format("{}", eq[idx<1>()]);        // "y_c" — ∂f/∂x
```

## License

See [LICENSE.txt](LICENSE.txt). Suggestions and pull requests are welcome.
