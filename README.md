[![CMake](https://github.com/reach2sayan/ddx/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/reach2sayan/ddx/actions/workflows/cmake-multi-platform.yml)
[![C++](https://img.shields.io/badge/C++-%2300599C.svg?logo=c%2B%2B&logoColor=white)](#)

# ddx

A C++23 library for differentiating expressions. Write a function over named
symbols and ask it for values, Jacobians — the gradient, when there is one
output — Hessians or higher-order derivative tensors, at run time, in a
`static_assert`, or over a batch of thousands of points.

```cpp
#include "ddx.hpp"
using namespace ddx;

// residuals of the fit m(t) = a·exp(−b·t) against four measurements
constexpr auto residual(auto a, auto b, double t, double y) {
  return y - a * exp(-b * t);
}

constexpr auto rss(auto a, auto b) {
  const auto r0 = residual(a, b, 0.0, 2.00);
  const auto r1 = residual(a, b, 1.0, 1.21);
  const auto r2 = residual(a, b, 2.0, 0.74);
  const auto r3 = residual(a, b, 3.0, 0.45);
  return r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3;
}

const auto eq = Equation{rss(var<"a">, var<"b">)};

eq.evaluate(2.0, 0.5);      // 4.1343959887e-05
eq.jacobian(2.0, 0.5);      // {-0.0010757424682, 0.0150678475595}
```

`residual` and `rss` are ordinary functions taking their symbols as arguments,
so the same source serves every symbol type on this page.

**Contents** — [Requirements](#requirements) · [Using it](#using-it) ·
[Building](#building) · [Expressions](#expressions) · [Points](#points) ·
[Equation](#equation) · [Reverse mode](#reverse-mode) ·
[Forward mode](#forward-mode) · [Runtime expressions](#runtime-expressions) ·
[Printing](#printing) · [Map](#map) ·
[Compile-time results](#compile-time-results) ·
[Values and lifetime](#values-and-lifetime) · [Errors](#errors) ·
[What `ddx.hpp` declares](#what-ddxhpp-declares)

---

## Requirements

- A C++23 compiler and standard library: **GCC 14+**, **Clang 19+** over
  libstdc++ 14+, **Clang 17+** over libc++ 17+, or MSVC (VS 2022,
  `/std:c++latest`).
- **CMake 3.26+** to build through CMake.
- **Boost's headers**, on the machine that builds ddx and on any machine that
  compiles against it — `libboost-dev`, `boost-devel`, `brew install boost`.
  ddx names Mp11 for the symbol lists, and `ddx::rt` names Boost.Graph for the
  Hessian colouring and Boost.DynamicBitset for the coupling rows. All of it is
  header-only, so no compiled Boost library is ever linked. No version is
  demanded: configure checks that the headers ddx names compile, which is what
  a partial install fails rather than an old one. The version a consumer
  resolves is reported when it differs from the one ddx was built against.
- GoogleTest and Google Benchmark are fetched at configure time, so a first
  configure that builds the tests or the benchmarks wants a network.
- `-DDDX_BUILD_JIT=ON` additionally needs an LLVM 18–20 installation, pointed
  at with `LLVM_DIR`.

## Using it

In a project that vendors ddx:

```cmake
add_subdirectory(ddx)
target_link_libraries(my_app PRIVATE ddx::ddx)
```

Or against an installed or built copy:

```cmake
find_package(ddx CONFIG REQUIRED COMPONENTS dual jit)
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
which option turns it on. The package also reports the flags ddx itself was
compiled with, as `ddx_CODEGEN_FLAGS` and `ddx_FP_CONTRACT`; matching
`ddx_FP_CONTRACT` is worth doing if your results have to agree with ddx's to the
last bit.

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

```sh
cmake --preset release_jit
cmake --build --preset release_jit
ctest --preset release_jit
```

The JIT presets point `LLVM_DIR` at the Debian/Ubuntu `llvm-20` layout;
override it on the command line, which wins over the preset:

```sh
cmake --preset release_jit -DLLVM_DIR=/opt/llvm-19/lib/cmake/llvm
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `DDX_BUILD_JIT` | `OFF` | compile the LLVM backend into the library |
| `DDX_SHARED_LIBS` | `ON` | build the library shared rather than static |
| `DDX_BUILD_BENCHMARKS` | `ON` | build the benchmark targets |
| `DDX_INSTALL` | on if top-level | generate the install and `find_package` rules |
| `ENABLE_NATIVE_ARCH` | `ON` | `-march=native`, falling back to `x86-64-v3` |
| `DDX_FP_FLAGS` | `ON` | `-ffp-contract=fast -fno-math-errno` |
| `DDX_MDSPAN_MODE` | `auto` | `auto` / `std` / `vendored` — which `mdspan` to bind to |
| `DDX_DEDUCING_THIS` | `auto` | `auto` / `on` / `off` — accessor spelling (P0847) |

`-ffast-math` is not used and not recommended: it changes derivative values.

ddx's own targets compile `-fno-exceptions`. Nothing in the library throws —
failures come back as values, `result` from the runtime calls and `jit::error`
from the backend — and the headers throw nothing either, so what you compile
your own code with is your business.

---

## Expressions

```cpp
#include "ddx.hpp"
using namespace ddx;
```

Everything up to [runtime expressions](#runtime-expressions) is header-only and
`constexpr`.

### Symbols

A symbol is a name and a scalar type. Name it with a template argument, or with
a symbol you already hold:

| Spelling | Symbol |
|---|---|
| `var<"x">` | `"x"`, valued `double` |
| `var<"x", T>` | `"x"`, valued `T` |
| `var_of<"x">(v)` | `"x"`, valued `decltype(v)` — `v` supplies the type only |
| `variable("x"_s)` | `"x"`, valued `double` |
| `variable<T>("x"_s)` | `"x"`, valued `T` |
| `var_of("x"_s, v)` | `"x"`, valued `decltype(v)` |
| `2.0_vd` | `"v"`, valued `double` |
| `4_vi` | `"c"`, valued `int` |

`"x"_s` and `sym<"x">` are the symbol itself, without a scalar type — the label
`Map` keys by, `named` binds to, and `variable` turns into a symbol of a type.

```cpp
constexpr auto x = var<"x">;
constexpr auto n = var<"n", int>;
constexpr auto y = variable("y"_s);
```

The literals — `"x"_s`, `_vd`, `_vi`, `_cd`, `_ci` — live in `ddx::literals`:

```cpp
using namespace ddx::literals;
```

### Constants

| Spelling | Constant |
|---|---|
| `constant(3.0)` | a `double` stored in the expression |
| `1.5_cd` | the same, as a literal |
| `3_ci` | an `int` constant |

A bare scalar mixed with an expression is promoted for you: `x * 2.0`,
`2.0 * x + 1.0` and `y - a * exp(…)` need no wrapping. Reach for `constant`
when you want the constant spelled out.

### Operators and functions

| Kind | Available |
|---|---|
| Arithmetic | `+` `-` `*` `/` unary `-` |
| Unary | `sin` `cos` `tan` `exp` `log` `log10` `sqrt` `cbrt` `abs` `asin` `acos` `atan` `sinh` `cosh` `tanh` `asinh` `acosh` `atanh` `erf` |
| Binary | `pow` `atan2` `hypot` `max` `min` |

They are found by argument-dependent lookup, so they work on an expression
whether or not `ddx` is in scope.

```cpp
auto f = (x + y) * (x - y) + exp(x * y) + sin(y) * x * y;
auto g = pow(x, 2.0) + hypot(x, y) - log(max(x, y));
```

Expressions are algebraically folded as they are built, so `x + 0`, `x * 1`,
`x / 1`, `-(-x)`, `pow(x, 1)` and literal arithmetic never reach the tree.

---

## Points

Every call that needs numbers takes them in four interchangeable spellings:

```cpp
auto f = x * y + 3.0 * x;

f.eval(4.0, 2.0);                               // positional
f.eval(named<"y">(2.0), named<"x">(4.0));       // by name, any order
f.eval("y"_s = 2.0, "x"_s = 4.0);               // by name, assignment spelling
f.eval(std::array{4.0, 2.0});                   // any range
```

**Positional order is alphabetical by symbol name**, not the order the symbols
appear in the expression. For `rss` above the order is `a`, `b`.

`named` also takes a symbol in hand: `named("x"_s, 4.0)`, `named(x, 4.0)`.

Every entry point in the library accepts all four spellings. The return type
follows the spelling:

| Point | Returns |
|---|---|
| positional | the value |
| named | the value |
| a range whose size is in its type (`std::array`) | the value |
| a range whose size is not (`std::vector`, `std::span<T>`) | `result<value>` |

```cpp
const std::vector pt{2.0, 0.5};
const auto v = eq.evaluate(pt);        // ddx::result<double>
if (v) { std::println("{}", *v); }
```

`Equation::point(args...)` builds the point on its own, in canonical order:

```cpp
const auto at = decltype(eq)::point(named<"b">(0.5), named<"a">(2.0));
```

---

## Equation

`Equation` wraps one expression for f: ℝⁿ → ℝ, or several for f: ℝⁿ → ℝᵐ.

```cpp
auto eq  = Equation{x * y};                    // one output
auto sys = Equation(x + y, x * y, sin(x));     // three outputs
Equation eq2 = x * y;                          // implicit from one expression
```

| Member | Answers |
|---|---|
| `input_dim` | number of distinct symbols, `n` |
| `output_dim` | number of outputs, `m` |
| `number_of_derivatives` | `n` |
| `evaluate(point)` | f at the point |
| `jacobian(point)` | J, `J[i, j] = ∂fᵢ/∂xⱼ`; ∇f itself when `m == 1` — [reverse mode](#reverse-mode) |
| `hessian(point)` | ∇²f — [forward mode](#forward-mode) |
| `derivative_tensor<K>(point)` | all K-th order partials — [forward mode](#forward-mode) |
| `univariate_derivative<K>(x0)` | the K-th derivative of a one-symbol function |
| `eq[idx<N>()]`, `eq.get<N>()` | slot `N` as an expression |

```cpp
static_assert(decltype(eq)::input_dim == 2);
static_assert(decltype(eq)::output_dim == 1);
```

An expression evaluates on its own, without an `Equation`:

```cpp
(x * y).eval(4.0, 2.0);                        // 8
```

---

## Reverse mode

Values and Jacobians. One sweep per output, whatever `n` is.

```cpp
const auto eq = Equation{rss(var<"a">, var<"b">)};

eq.evaluate(2.0, 0.5);                          // 4.1343959887e-05
eq.jacobian(2.0, 0.5);                          // {-0.0010757424682, 0.0150678475595}
eq.evaluate(named<"b">(0.5), named<"a">(2.0));  // the same value
```

`jacobian` is the only first-derivative entry point, and `output_dim` picks the
shape it answers in: one output drops the leading axis and hands back a flat
`std::array<T, n>`, which is the gradient; a system answers a rank-2 `m × n`
tensor. `jacobian<DiffMode::Symbolic>(point)` differentiates the tree instead of
sweeping it, and answers the same shape.

A system answers with an array of values and a Jacobian. The residuals of the
example make one whose Jacobian is the design matrix of the fit:

```cpp
const auto a = var<"a">;
const auto b = var<"b">;

const auto res = Equation(residual(a, b, 0.0, 2.00), residual(a, b, 1.0, 1.21),
                          residual(a, b, 2.0, 0.74), residual(a, b, 3.0, 0.45));

res.evaluate(2.0, 0.5);                        // std::array<double, 4>
res.jacobian(2.0, 0.5);                        // 4 × 2, J[i, j] = ∂rᵢ/∂xⱼ
```

### Return types

| Call | Returns |
|---|---|
| `eval`, `evaluate` (`m == 1`) | the scalar type `T` |
| `evaluate` (`m > 1`) | `std::array<T, m>` |
| `jacobian` (`m == 1`) | `std::array<T, n>` |
| `jacobian` (`m > 1`) | rank-2 tensor, `m × n` |

`jacobian_t<Eq>`, `hessian_t<Eq>` and `derivative_tensor_t<Eq, K>` name those
shapes where you have to declare one. They name the shape itself: the members
return `auto`, because a point whose length is known only at run time comes back
wrapped in `result`.

Tensors index two ways, and report their shape:

```cpp
auto J = res.jacobian(2.0, 0.5);

J[0, 1];                    // ∂r₀/∂b
J[0][1];                    // the same element
J.extent(0);                // 4
decltype(J)::rank();        // 2
J.data();                   // contiguous storage
```

---

## Forward mode

Second and higher derivatives: Hessians, derivative tensors of any order,
univariate derivatives to order K, and one-pass directional derivatives. Same
`Equation`, same points, same header.

### `hessian`

`hessian` reads its second derivative out of dual-valued symbols, so the
symbols are declared with the scalar type `dual`:

| Spelling | Symbol |
|---|---|
| `var<"a", dual>` | `"a"`, valued `dual` |
| `variable<dual>("a"_s)` | the same |
| `dual_var_of<"a">(v)` | `"a"`, dual over `decltype(v)` |
| `dual_var_of("a"_s, v)` | the same |

`dual2nd` is the twice-nested scalar; declare it the same way when you want a
dual-valued expression one order deeper.

```cpp
const auto eq = Equation{rss(var<"a", dual>, var<"b", dual>)};
const auto H = eq.hessian(2.0, 0.5);

H[0, 0];    //  3.1060035856
H[0, 1];    // -3.1441109272
H[1, 1];    // 10.8108574600
```

**The point is always given in the base scalar type.** An expression over
`dual` symbols still takes `double` values:

```cpp
eq.hessian(named<"b">(0.5), named<"a">(2.0));
eq.hessian(std::array{2.0, 0.5});
```

One output gives an `n × n` tensor; a system gives `m × n × n`, one block per
output:

```cpp
const auto a = var<"a", dual>;
const auto b = var<"b", dual>;

Equation{a * b}.hessian(2.0, 3.0);              // 2 × 2
Equation(a * b, a * a).hessian(2.0, 3.0);       // 2 × 2 × 2, H[k, i, j]
```

With plain symbols the call does not exist — use `derivative_tensor<2>`.

### `derivative_tensor<K>`

All K-th order partials, on plain symbols, for any `K ≥ 1`:

```cpp
const auto eq = Equation{rss(var<"a">, var<"b">)};

eq.derivative_tensor<1>(2.0, 0.5);      // {-0.0010757424682, 0.0150678475595}
eq.derivative_tensor<2>(2.0, 0.5);      // the same Hessian as above
eq.derivative_tensor<3>(2.0, 0.5);      // rank 3

const auto T = eq.derivative_tensor<3>(2.0, 0.5);
T[0, 0, 1];      // ∂³f/∂a²∂b = -3.1516448510
```

For a system the leading axis is the output:

```cpp
Equation(x * y, sin(x)).derivative_tensor<2>(1.0, 2.0);   // 2 × 2 × 2
```

### `univariate_derivative<K>`

The K-th derivative of a one-symbol function, as a plain number:

```cpp
const auto in_b = Equation{rss(constant(2.0), var<"b">)};

in_b.univariate_derivative<2>(0.5);     // 10.8108574600
in_b.univariate_derivative<4>(0.5);     // 367.3330206117
```

The expression must have exactly one symbol and one output.

### `eval_with_tangent`

A value and one directional derivative from a single expression, without an
`Equation`:

```cpp
const auto t = (var<"a"> * var<"b">).eval_with_tangent<"a">(2.0, 0.5);

t.value();      // 1.0
t.deriv();      // ∂(ab)/∂a = 0.5
```

Arguments are positional, in canonical (alphabetical) order. The result is a
`dual`, seeded on the named symbol.

### Return types

`n` is the symbol count, `m` the output count, `T` the base scalar type.

| Call | `m == 1` | `m > 1` |
|---|---|---|
| `hessian` | rank-2, `n × n` | rank-3, `m × n × n` |
| `derivative_tensor<1>` | rank-1, `n` | rank-2, `m × n` |
| `derivative_tensor<2>` | rank-2, `n × n` | rank-3, `m × n × n` |
| `derivative_tensor<K>` | rank-K, `n^K` | rank-(K+1), `m × n^K` |
| `univariate_derivative<K>` | `T` | — |
| `eval_with_tangent<"x">` | `dual` | — |

Symmetric tensors are stored packed, so `size()` is smaller than `n^K` and
`data()` is not a dense row-major block — reach elements through `[i, j]` or
`[i][j]`.

Forward mode is part of every build. What decides whether these entry points
exist is the expression's scalar: `hessian` and the higher tensors need a symbol
valued `dual` or `dual2nd`, and constrain themselves away on one valued plainly.

---

## Runtime expressions

Build an expression while the program runs — terms looped over a data file, a
model read from configuration — and differentiate it through the same
`Equation` interface. One point at a time, or a whole batch in one call.

```cpp
#include "ddx.hpp"
#include "rt/equation.hpp"
using namespace ddx;
```

Link `ddx::rt`, or `ddx::jit` for the compiled batch calls.

The residual sum of squares of the example, over measurements that arrive at
run time:

```cpp
struct Sample { double t, y; };
const std::vector<Sample> data = load();          // however you read it

const auto eq = rt::equation([&] {
  const auto a = rt::var("a");
  const auto b = rt::var("b");

  rt::RTExpression<double> rss = 0.0;
  for (const auto &s : data) {
    const auto r = s.y - a * exp(-b * s.t);
    rss = rss + r * r;
  }
  return rss;
});

*eq.arity();                   // 2
*eq.symbols();                 // {"a", "b"}
*eq.evaluate(2.0, 0.5);        // 4.1343959887e-05
*eq.jacobian(2.0, 0.5);        // {-0.0010757424682, 0.0150678475595}
```

### Building an equation

`rt::equation` takes a callback, runs it, and hands back the equation. Symbols
are named inside it with `rt::var(name)` — a string, not a template argument.

```cpp
const auto eq = rt::equation([] {
  const auto x = rt::var("x");
  const auto y = rt::var("y");
  return exp(x) * sin(y);
});
```

`rt::RTExpression<T>` is the expression type. A bare number converts to one, so
an accumulator starts at `0.0`, and the operators and math functions are
spelled exactly as [above](#operators-and-functions).

Return several expressions — as a `std::array` — for a system:

```cpp
const auto sys = rt::equation([] {
  const auto x = rt::var("x");
  const auto y = rt::var("y");
  return std::array{x * y + sin(x), exp(x) - y * y};
});

*sys.jacobian(1.3, 0.7);       // row-major, m × n
```

Another scalar type is a template argument on both: `rt::var<T>(name)` inside
`rt::equation<T>(...)`.

Symbols are ordered canonically — alphabetically by name — the same as
everywhere else in the library. `symbols()` lists them in that order and
`arity()` counts them, both `nullopt` if the equation is poisoned.

### Failure, in two layers

`rt::equation` always hands back an `Equation`. When the callback could not
produce one — a symbol named with no arena, an expression naming no graph — that
equation is *poisoned*: it carries the error rather than a graph, and every call
on it answers with that error instead of a value.

```cpp
const auto eq = rt::equation([] { return rt::var("x") * 2.0; });

if (const auto bad = eq.status()) {
  std::println("{}", ddx::message(bad->code));      // errc::no_arena
  return;
}
```

`status()` is a `std::optional<ddx::error>`; `poisoned()` is the same question
without the reason.

Calls still answer `result<T>` — `std::expected<T, ddx::error>` — on top of
that, and always will: the symbol list exists only at run time, so an arity
mismatch or an unknown name is a genuine runtime failure with no compile-time
counterpart.

```cpp
const auto j = eq.jacobian(1.0, 2.0);      // two values, one symbol
if (!j) {
  std::println("{}", ddx::message(j.error().code));  // errc::wrong_arity
}
```

The `*` in the examples here is that per-call check skipped.

The six accessors that answer no `result` answer `std::optional` instead:
`arity()`, `symbols()`, `value_columns()`, `jacobian_columns()`,
`hessian_columns()` and `hessian_colors()` are `nullopt` on a poisoned equation.
A bare count could not say it — an equation over a literal-only graph
legitimately has no symbols and no output columns, so 0 would not be
distinguishable from there being no equation — and a loop written over
`*eq.arity()` stops compiling rather than silently running no times.

Points come in the same spellings as everywhere else, and `eq.point(args...)`
builds the point vector on its own:

```cpp
*eq.jacobian(2.0, 0.5);                                 // positional, canonical order
*eq.jacobian(std::array{2.0, 0.5});                     // any range
*eq.jacobian(named<"a">(2.0), named<"b">(0.5));         // by name, any order
```

### Per-point calls

| Member | Answers |
|---|---|
| `poisoned()`, `status()` | whether the build failed, and with what |
| `arity()` | `std::optional<std::size_t>` — symbol count, `n` |
| `symbols()` | `std::optional<std::span<const std::string>>` — canonical order |
| `point(args…)` | `result<std::vector<T>>` — a point in canonical order |
| `evaluate(point)` | `result<T>`, or `result<std::vector<T>>` for a system |
| `jacobian(point)` | `result<std::vector<T>>`, row-major `m × n`, so `n` long when `m == 1` |
| `hessian(point)` | `result<std::vector<T>>`, dense row-major `m × n × n`, `m == 1` |
| `hessian_colors()` | `std::optional<std::size_t>` |
| `univariate_derivative<K>(x0)` | `result<T>`, one symbol only |
| `value_columns()`, `jacobian_columns()`, `hessian_columns()` | `std::optional<std::size_t>` — column counts for the batch calls |

```cpp
const auto H = *eq.hessian(2.0, 0.5);      // 4 entries: H[i*n + j]
H[0];                                      //  3.1060035856
H[1];                                      // -3.1441109272
```

### Batch calls

The batch calls take columns: one input column per symbol, one output pointer
per output column, every column `n` points long. They fill the outputs in place
and answer `result<void>`.

```cpp
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
  std::println("{}", ddx::message(ok.error().code));
}

f[2];                    // 4.1343959887e-05  — RSS at (2.0, 0.50)
j[2];                    // ∂RSS/∂a there
j[n + 2];                // ∂RSS/∂b there
```

A column count that does not match answers `errc::wrong_column_count`, and
nothing is written.

`hessian` takes a fourth block, `hessian_columns()` wide:

```cpp
std::vector<double> h(n * *eq.hessian_columns());
std::vector<double *> hessians;
for (std::size_t k = 0; k < *eq.hessian_columns(); ++k) {
  hessians.push_back(h.data() + k * n);
}

const auto ok = eq.hessian(xs, values, partials, hessians, n);
```

### Ownership and threads

**The equation owns everything it needs.** It is move-only: return it from a
function, store it in a class, keep it as long as you like. Nothing has to be
kept alive beside it.

```cpp
auto make_model(const std::vector<Sample> &data) {
  return rt::equation([&] {
    const auto a = rt::var("a");
    const auto b = rt::var("b");
    rt::RTExpression<double> rss = 0.0;
    for (const auto &s : data) {
      const auto r = s.y - a * exp(-b * s.t);
      rss = rss + r * r;
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

If you hold an `rt::Builder<T>` of your own, `rt::var(builder, name)` names a
symbol in it and `rt::equation(expr…)` builds an equation over expressions
already in it. That equation borrows the builder, which must then outlive it.

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

### The JIT

Configure with `-DDDX_BUILD_JIT=ON` and link `ddx::jit`, and the batch calls run
compiled code; without it they run interpreted. The spellings above do not
change, the answers do not change, and there is no compiler or kernel object to
hold. Compilation happens on the first batch call that needs it.

---

## Printing

Expressions and `Equation`s are formattable and streamable:

```cpp
std::format("{}", x * y + sin(x));      // "x * y + sin(x)"
std::format("{}", x / (y / x));         // "x / (y / x)"
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
eq.get<1>();                            // same slot as eq[idx<1>()]
```

`functions()` hands back the outputs as a tuple of expressions, and
`jacobian_rows()` the derivative rows, one per output.

---

## Map

`map` collects the same keyword arguments into a value with the keys in its
type. The values need not share a type.

```cpp
constexpr auto m = map(named<"n">(3), named<"x">(1.5));

static_assert(m.get<"n">() == 3);          // int
static_assert(m["x"_s] == 1.5);            // double
static_assert(m.contains<"n">());
static_assert(m.size == 2);
```

| Call | Does |
|---|---|
| `m.get<"x">()`, `m["x"_s]` | read a slot |
| `m.set<"x">(v)`, `m["x"_s] = v` | write it in place, keeping its type |
| `m.insert(named<"y">(v))` | a new map with one more entry |
| `m.erase<"x">()` | a new map with one fewer, order preserved |
| `m.contains<"x">()` | `consteval bool` |
| `m.size` | entry count |
| `m.keys()` | the keys, in entry order |
| `m.for_each(f)` | call `f(key, value)` over the entries |
| `m == m2` | keys in order, then values |

Four equivalent spellings of the same map:

```cpp
map(named<"n">(3), named<"x">(1.5));
Map{named<"n">(3), named<"x">(1.5)};
Map{NamedValue{"n"_s, 3}, NamedValue{var<"x">, 1.5}};
Map<NamedValue<"n", int>, NamedValue<"x", double>>{{3}, {1.5}};
```

Adding or removing an entry gives a new map; writing one is in place:

```cpp
auto m2 = m;
m2.set<"x">(2.5);
m2["x"_s] = 2.5;

constexpr auto m3 = m.insert(named<"y">('c'));                  // 3 entries
constexpr auto m4 = m3.erase<"n">();                            // 2 entries
constexpr auto m5 = m.erase<"n">().insert(named<"n">(2.5f));    // "n" is now float
```

A map is not a point: pass `named<"x">(v)` arguments to `evaluate` directly.

---

## Compile-time results

Every entry point outside `ddx::rt` is `constexpr`, so a value or a derivative
computed at a known point is baked into the binary:

```cpp
constexpr auto g = Equation{rss(var<"a">, var<"b">)}.jacobian(std::array{2.0, 0.5});
static_assert(g[0] < 0.0);

constexpr auto H = Equation{var<"x", dual> * var<"y", dual>}.hessian(std::array{2.0, 3.0});
static_assert(H[0, 1] == 1.0);
```

The same calls serve at run time; nothing here is `consteval`.

---

## Values and lifetime

Expressions, `Equation`s and partials are values: copy them, store them in a
class, return them from a function, pass them across threads.

A point is read during the call and never kept. Everything a call returns —
scalar, `std::array`, tensor — is an independent owning value with no reference
back into the equation or the point, and stays valid after both are gone.

```cpp
auto curvature(double a0, double b0) {
  return Equation{rss(var<"a", dual>, var<"b", dual>)}.hessian(a0, b0);
}
```

---

## Errors

Most mistakes are compile errors:

| Message | Cause |
|---|---|
| `eval: supply exactly one value per symbol, in canonical order` | wrong number of positional values |
| `eval: no value supplied for this symbol` | a named point misses a symbol |
| `Map: key not present (see keys())` | no such key |
| `Map::insert: key already present (use set<Key>)` | duplicate key |
| `Map::erase: key not present` | no such key |

A range whose length is not in its type is checked at run time instead, and
answers `result<T>` — `std::expected<T, ddx::error>`:

```cpp
const std::vector short_pt{2.0};
const auto v = eq.evaluate(short_pt);
if (!v) {
  std::println("{}", ddx::message(v.error().code));   // errc::short_point
}
```

Runtime expressions answer `result` from every call, with these codes:

| `errc` | Means |
|---|---|
| `no_arena` | a symbol was named outside a `rt::equation` callback |
| `no_graph` | the expression names no graph — a bare literal, for instance |
| `wrong_arity` | the point does not supply one value per symbol |
| `unknown_symbol` | a named point uses a name the equation does not have |
| `short_point` | a range point is shorter than the symbol list |
| `wrong_column_count` | a batch block has the wrong number of columns |
| `not_univariate` | `univariate_derivative` on more than one symbol |
| `jit_target`, `jit_module`, `jit_verify`, `jit_lookup` | the JIT could not compile the graph |

`ddx::message(code)` turns one into text. Nothing here throws.

---

## What `ddx.hpp` declares

`#include "ddx.hpp"` is the whole public surface, and `using namespace ddx;`
brings in these names:

| Name | Is |
|---|---|
| `Equation` | the derivative API — every derivative entry point is a member |
| `var<"x">`, `sym<"x">`, `variable("x"_s)`, `idx<N>()` | name a symbol; index into an `Equation` |
| `var_of<"x">(v)`, `dual_var_of<"x">(v)` | name a symbol, taking its scalar type from `v` |
| `constant(3.0)` | a value stored in the expression |
| `literals` — `"x"_s`, `_cd`, `_ci`, `_vd`, `_vi` | the user-defined literals, as a namespace |
| `named<"x">(v)`, `NamedValue` | one keyword argument of a point, or one map entry |
| `map(…)`, `Map` | a compile-time map of those entries |
| `dual`, `dual2nd` | the symbol value types `hessian` needs |

The operators, the math functions, `operator<<` and the `std::formatter`
specialisations are found by argument-dependent lookup, so they work on an
expression without being named here.

Runtime expressions are opt-in and live in `ddx::rt`, reached through
`#include "rt/equation.hpp"`.

## License

See [LICENSE.txt](LICENSE.txt). Suggestions and pull requests are welcome.
