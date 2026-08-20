[![CMake](https://github.com/reach2sayan/ddx/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/reach2sayan/ddx/actions/workflows/cmake-multi-platform.yml)
[![C++](https://img.shields.io/badge/C++-%2300599C.svg?logo=c%2B%2B&logoColor=white)](#)

# ddx

A header-only C++23 library for symbolic expression trees and automatic
differentiation: symbolic derivatives, forward mode (dual numbers), reverse mode
(adjoint sweeps), Hessians, higher-order derivative tensors, and Taylor-mode
univariate derivatives — all usable at run time or inside `constexpr`.

```cpp
#include "ddx.hpp"
using namespace ddx;

constexpr auto x = var<"x">;
constexpr auto y = var<"y">;

auto f  = exp(x) * sin(y);
auto v  = Equation{f}.evaluate(1.0, 2.0);   // f(1, 2)
auto g  = Equation{f}.gradient(1.0, 2.0);   // {∂f/∂x, ∂f/∂y}
```

**Contents**

1. [Requirements and setup](#requirements-and-setup)
2. [Building expressions](#building-expressions)
3. [Supplying a point](#supplying-a-point)
4. [Derivatives: the `Equation` API](#derivatives-the-equation-api)
5. [Return types](#return-types)
6. [Getting the most speed out of it](#getting-the-most-speed-out-of-it)
7. [Compile-time use](#compile-time-use)
8. [Building the project](#building-the-project)
9. [Cheat sheet](#cheat-sheet)
10. [Diagnostics](#diagnostics)

---

## Requirements and setup

- A C++23 compiler with a C++23 standard library: **GCC 14+**, or **Clang 17+**
  over libstdc++ 14+ / libc++ 17+. MSVC (VS 2022, `/std:c++latest`) is supported.
- **CMake 3.20+** if you build through CMake.

C++23 is a hard requirement — the library uses `constexpr std::bitset` inside
`consteval` functions and the multidimensional subscript `t[i, j, k]`.

The library is header-only and has no third-party dependencies. A reference
implementation of `std::mdspan` is vendored under `include/md/third_party/` and
is used automatically when your toolchain has no complete `<mdspan>`.

### Using it

With CMake, link the interface target — it puts `include/` on your include path:

```cmake
add_subdirectory(ddx)
target_link_libraries(my_app PRIVATE ddx::ddx)   # or: ddx
```

Then include the public header:

```cpp
#include "ddx.hpp"    // the whole public surface

using namespace ddx;  // assumed by every example below
```

### What `namespace ddx` contains

`ddx.hpp` puts nine names in `namespace ddx`, and that is the whole surface:

| Name | Purpose |
|---|---|
| `Equation` | the derivative API — every derivative entry point is a member |
| `var<"x">`, `sym<"x">`, `idx<N>()` | name a symbol; index into an `Equation` |
| `literals` — `"x"_s` | the symbol literal, as a namespace |
| `named<"x">(v)` | one keyword argument of a point |
| `DiffMode` | `Symbolic` vs `Reverse` |
| `dual`, `dual2nd` | the symbol value type `Equation::hessian` needs |

The operators (`+`, `*`, …), the math functions (`sin`, `exp`, …), `operator<<`
and the `std::formatter` specialisations are deliberately *not* on that list.
They are found by argument-dependent lookup, so they work on an expression
without their names being visible. `using namespace ddx;` therefore brings in
nine names, not a hundred and a half.

The convenience macros (`PC`, `PV`, `PDV`) and the user-defined literals (`_cd`,
`_ci`, `_vd`, `_vi`) are global, in no namespace at all.

---

## Building expressions

### Symbols

A variable is a **name**, spelled `var<"x">`:

```cpp
constexpr auto x = var<"x">;              // a double-valued symbol named "x"
constexpr auto n = var<"n", int>;         // another scalar type
auto xd = var<"x", dual>;                 // dual-valued symbol (needed for hessian())
```

Equivalent spellings, when you would rather deduce the scalar type from an
exemplar value:

| Syntax | Means |
|---|---|
| `var<"x">` | a symbol named `"x"`, valued `double` |
| `var<"x", T>` | a symbol named `"x"`, valued `T` |
| `PV(1.0, "x")` | same as `var<"x">` — the `1.0` supplies only the *type* |
| `PDV(1.0, "x")` | same as `var<"x", dual>` |
| `2.0_vd` | a `double` symbol named `"v"` |
| `4_vi` | an `int` symbol named `"c"` |

### Constants

| Syntax | Means |
|---|---|
| `PC(3.0)` | a `double` constant stored in the tree |
| `1.5_cd` | the same, as a literal |
| `3_ci` | an `int` constant |

A bare scalar mixed with an expression is promoted automatically, so `x * 2.0`
and `2.0 * x + 1.0` need no wrapping. Use `PC` only when you want the constant
spelled explicitly.

### Operators and functions

| Kind | Available |
|---|---|
| Arithmetic | `+`  `-`  `*`  `/`  unary `-` |
| Unary functions | `sin` `cos` `tan` `exp` `log` `log10` `sqrt` `cbrt` `abs` `asin` `acos` `atan` `sinh` `cosh` `tanh` `asinh` `acosh` `atanh` `erf` |
| Binary functions | `pow` `atan2` `hypot` `max` `min` |

```cpp
auto f = (x + y) * (x - y) + exp(x * y) + sin(y) * x * y;
auto g = pow(x, 2.0) + hypot(x, y) - log(max(x, y));
```

Trees are simplified as they are built: `x + 0`, `x * 1`, `x * 0`, `x / 1`,
`-(-x)`, `pow(x, 0)`, `pow(x, 1)` and literal folding are applied by the
operators themselves, so a derivative comes out as `y` rather than
`1 * y + x * 0`.

### Symbols carry no value

A symbol holds nothing, and neither does any node built from it. An expression
over symbols is an *empty type*, whatever its depth and however often a symbol
repeats:

```cpp
auto f = (x + y) * (x - y) + exp(x * y) + sin(y) * x * y;
static_assert(std::is_empty_v<decltype(f)>);   // sizeof(f) == 1
```

The point is supplied where you ask for a value or a derivative, never where the
tree is built. One value slot exists per *symbol*, not per leaf occurrence, and
an expression is free to copy, store, and pass by value.

### Printing

Expressions and `Equation`s are formattable and streamable:

```cpp
std::format("{}", x * y + sin(x));      // "x * y + sin(x)"
std::format("{}", x / (y / x));         // "x / (y / x)"  — parentheses as needed
std::format("{::.3f}", 2.0 * x);        // "2.000 * x"    — spec applies to every number
std::cout << (x - y * x) << '\n';       // "x - y * x"
```

For an `Equation` the format prints each function followed by its gradient row.
A symbol held constant for the purpose of one partial derivative prints with a
`_c` suffix:

```
f0: x * y
  grad: y_c, x_c
```

---

## Supplying a point

Every entry point that needs numbers accepts the point in three interchangeable
spellings:

```cpp
auto f = x * y + PC(3.0) * x;                       // f(x, y) = xy + 3x

f.eval(4.0, 2.0);                                   // positional, canonical order
f.eval(named<"y">(2.0), named<"x">(4.0));           // by name, order-independent
f.eval(std::array{4.0, 2.0});                       // any input range
```

All of these give `20`. The same spellings work for `Equation::evaluate`,
`gradient`, `jacobian`, `hessian`, `derivative_tensor` — one normaliser serves
them all.

### Canonical order

**Canonical order is alphabetical by symbol name, not the order you wrote them
in.** For `f(w, x, y, z)` the positional form expects `w` first — and for
`f(x, y)` above, `x` then `y`.

The positional and range forms check the count at compile time, so a missing or
extra value is a compile error. The named form binds by name and is immune to
ordering entirely — prefer them when the symbol set is large. A
non-sized range that runs short throws `std::out_of_range` (a compile error
during constant evaluation).

## Derivatives: the `Equation` API

`Equation` is the derivative API. Wrap one expression for a scalar function
f: ℝⁿ → ℝ, or several for a vector function f: ℝⁿ → ℝᵐ:

```cpp
auto eq  = Equation{x * y};                    // scalar
auto sys = Equation(x + y, x * y, sin(x));     // three outputs
```

`n` is the number of distinct symbols across all outputs, in canonical order;
`m` is the number of outputs. Both are available as
`decltype(eq)::input_dim` / `::output_dim`.

### Scalar functions

```cpp
constexpr auto x = var<"x">;
constexpr auto y = var<"y">;
auto eq = Equation{exp(x) * sin(y)};
const std::array pt{1.0, 2.0};

eq.evaluate(pt);                          // f(x, y)
eq.gradient(pt);                          // reverse mode — one backward sweep
eq.gradient<DiffMode::Symbolic>(pt);      // evaluate the stored partial trees
eq.derivative_tensor<1>(pt);              // forward mode gradient
eq.derivative_tensor<2>(pt);              // forward-over-forward Hessian
```

Every one of them also takes positional or named arguments:

```cpp
eq.gradient(1.0, 2.0);
eq.gradient(named<"y">(2.0), named<"x">(1.0));
```

#### Hessians

There are two Hessians, and they differ in what the symbols must be:

```cpp
// forward-over-forward: plain scalar symbols
auto H1 = Equation{x * y}.derivative_tensor<2>(std::array{2.0, 3.0});

// forward-over-reverse: dual-valued symbols
auto a = var<"x", dual>;
auto b = var<"y", dual>;
auto H2 = Equation{a * b}.hessian(std::array{2.0, 3.0});
```

`hessian()` needs `dual`-valued symbols because it seeds tangents into the
tree; `derivative_tensor<K>` builds its nested duals internally and works on
plain scalar symbols. Either way the *point* is given in the base scalar type
(`double` above), never in the dual type.

#### Higher-order and univariate derivatives

`derivative_tensor<K>` generalises to any order: it returns a rank-K symmetric
tensor of all K-th order partials.

```cpp
auto T3 = Equation{x * y + sin(x)}.derivative_tensor<3>(std::array{1.0, 2.0});
T3[0, 0, 1];   // ∂³f/∂x²∂y
```

For a **single-variable** function, `univariate_derivative<K>` is far cheaper —
it runs one Taylor-mode sweep in O(K²) instead of building a rank-K tensor:

```cpp
double d4 = Equation{sin(x)}.univariate_derivative<4>(1.0);   // f⁗(1.0)
```

#### The symbolic partials

An `Equation` can hand back the derivative *expressions* themselves. Slot 0 is
the function; slot k > 0 is the partial with respect to the k-th symbol in
canonical order:

```cpp
auto eq = Equation(x * y);

eq[idx<0>()];              // x * y
eq[idx<1>()].eval(2.0);    // ∂f/∂x = y  → 2
eq[idx<2>()].eval(4.0);    // ∂f/∂y = x  → 4
eq.get<1>();               // same as eq[idx<1>()]
```

Because the partials are simplified, each names only the symbols it actually
depends on and takes just those values — `∂(xy)/∂x` is `y`, a one-symbol
expression. The symbols that were held constant to form a partial print with a
`_c` suffix (`y_c`); they still read their value from the point as usual, their
derivative is simply zero. Like any expression the partials are empty types, so
holding them is free.

### Vector functions

```cpp
auto sys = Equation(x * y, sin(x) + y * y);
const std::array pt{1.0, 2.0};

sys.evaluate(pt);                          // std::array<double, 2>
sys.jacobian(pt);                          // reverse mode, J[i][j] = ∂fᵢ/∂xⱼ
sys.jacobian<DiffMode::Symbolic>(pt);      // symbolic
sys.derivative_tensor<1>(pt);              // forward-mode Jacobian
sys.derivative_tensor<2>(pt);              // per-output Hessians, H[k][i][j]
```

The reverse-mode per-output Hessian is available too, again on `dual`-valued
symbols:

```cpp
auto a = var<"x", dual>;
auto b = var<"y", dual>;
auto H = Equation(a * b, a * a).hessian(std::array{2.0, 3.0});
// H[0][i][j] = ∂²(xy)/∂xᵢ∂xⱼ,  H[1][i][j] = ∂²(x²)/∂xᵢ∂xⱼ
```

### Choosing a mode

| Call | Mode | Symbols | Cost |
|---|---|---|---|
| `gradient(pt)` | reverse | plain or dual | one backward sweep |
| `gradient<DiffMode::Symbolic>(pt)` | symbolic | plain or dual | evaluates n partial trees |
| `derivative_tensor<1>(pt)` | forward | plain | n forward sweeps |
| `hessian(pt)` | forward-over-reverse | **`dual` required** | one sweep per *colour* — see below |
| `derivative_tensor<2>(pt)` | forward-over-forward | plain | one sweep per index pair |
| `derivative_tensor<K>(pt)` | forward | plain | one sweep per distinct K-index |
| `univariate_derivative<K>(x0)` | Taylor | plain, n = 1 | one sweep, O(K²) |

Which to reach for is in [Getting the most speed out of
it](#getting-the-most-speed-out-of-it); the short version is reverse for a
gradient of many variables, and symbolic when you want to *see* the derivative
rather than just evaluate it.

There is no `DiffMode::Forward` — forward mode is reached through
`derivative_tensor<K>` and `univariate_derivative<K>`.

### One-shot forward tangent

To get a value and one directional derivative in a single pass, without an
`Equation`:

```cpp
auto t = (x * y).eval_with_tangent<"x">(4.0, 2.0);
t.value();   // 8
t.deriv();   // ∂(xy)/∂x = 2
```

It returns a `dual`, seeded on the named symbol. Arguments are positional, in
canonical order.

---

## Return types

Everything below is an owning value; the library keeps no reference to it.

| Call | Returns |
|---|---|
| `eval` / `evaluate` (m = 1) | the scalar type `T` |
| `evaluate` (m > 1) | `std::array<T, m>` |
| `gradient` | `std::array<S, n>` |
| `jacobian` | rank-2 tensor, `m × n` |
| `hessian` / `derivative_tensor<2>` (m = 1) | rank-2 tensor, `n × n` |
| `hessian` / `derivative_tensor<2>` (m > 1) | rank-3 tensor, `m × n × n` |
| `derivative_tensor<K>` (m = 1) | rank-K tensor, `n^K` |
| `derivative_tensor<K>` (m > 1) | rank-(K+1) tensor, `m × n^K` |
| `univariate_derivative<K>` | the scalar type `S` |

`S` is the base scalar type: for a `dual`-valued expression it is
`double`. A single-output system carries no leading output axis — a scalar
function's Hessian is an `n × n` matrix, not a `1 × n × n` stack.

### Tensors

The tensor type accepts two equivalent index spellings:

```cpp
auto H = Equation{x * y}.derivative_tensor<2>(std::array{2.0, 3.0});

H[0, 1];        // the mdspan spelling
H[0][1];        // the nested spelling — same element
H.extent(0);    // n
decltype(H)::rank();
H.data();       // contiguous storage (packed: symmetric entries are stored once)
```

Symmetric derivative tensors use a packed layout, so `data()` holds fewer
elements than `n^K`; index it through `[i, j]` / `[i][j]` rather than assuming a
dense stride.

---

## Getting the most speed out of it

**Pick the call that matches the shape of the problem.** This is worth more than
everything else on this page put together.

| You want | Call | What it costs |
|---|---|---|
| a gradient, many variables | `gradient(pt)` | one backward sweep, independent of n |
| a gradient, two or three variables | `gradient<DiffMode::Symbolic>(pt)` | n folded partial trees — often quicker at that size |
| a Hessian | `hessian(pt)`, symbols declared `dual` | one sweep per colour of the sparsity pattern |
| a Hessian without dual-valued symbols | `derivative_tensor<2>(pt)` | one sweep per index pair — O(n²) |
| the K-th derivative of one variable | `univariate_derivative<K>(x0)` | one Taylor sweep, O(K²) |
| all K-th partials of n variables | `derivative_tensor<K>(pt)` | one sweep per distinct K-index |

Reverse is the default for a reason, but it is not a clean win at every size: on
a three-symbol system here, `jacobian<DiffMode::Symbolic>` runs in 7.7 ns against
12.5 ns for `jacobian<DiffMode::Reverse>`, because at that width the folded
partial trees are cheaper than a sweep. Reverse pulls away as n grows. If n is
small and the call is hot, measure both — it is a one-word change.

**`hessian()` reads your problem's sparsity off the type, for free.** Two
variables that never appear in the same second-derivative term can be seeded in
the *same* backward sweep, so the cost is one sweep per *colour* of the coupling
pattern rather than one per variable — and the pattern is computed from the
expression type at compile time, so there is nothing to switch on and nothing to
pay at run time. On an 8-variable chain (each variable coupled only to its
neighbours, plus one long-range term) that is 5 sweeps instead of 8, measured
here at **1.7–1.9x** faster end to end. A dense problem colours in n and costs
exactly what it always did. The wider and more structured the problem, the more
this wins — so prefer `hessian()` over `derivative_tensor<2>()` whenever you can
declare the symbols `dual`.

**Prefer `univariate_derivative<K>` whenever the function really has one
variable.** `derivative_tensor<K>` builds a rank-K tensor to hold a single
number; the Taylor sweep is O(K²) and allocates nothing.

**Do not worry about rebuilding the tree.** Expressions and `Equation`s are
empty types — `sizeof(Equation{f})` is 1 — and the whole structure lives in the
type, so `Equation{f}.gradient(pt)` inside a loop constructs nothing at run
time. Hoisting it into a variable is a readability choice, not a speed one.

**Let the tree be simplified for you.** Algebraic folding happens as the
expression is built, so `∂(xy)/∂x` really is the single node `y`, and the
partials each name only the symbols they still depend on. Nothing you can write
by hand beats it, and hand-expanding an expression usually makes it worse.

**Give the point in the base scalar type.** A `dual`-valued expression still
takes a point of `double`; the seeding happens inside. Positional and named
arguments cost exactly the same — the reordering is resolved at compile time.

**Compile with the flags the project already sets.** `-ffp-contract=fast` and
`-fno-math-errno` (`DDX_FP_FLAGS=ON`, the default) are both worth having.
`-ffast-math` is **not**: it was measured at 19% *slower* here, and it changes
derivative values. `-march=native` (`ENABLE_NATIVE_ARCH=ON`) is on by default.

**Know where the time actually goes.** For a gradient of anything with `exp`,
`log`, `sin` or `pow` in it, the libm call dominates — around three quarters of
the total. Reducing the number of transcendental calls in the expression is the
optimisation with the most left in it; shaving arithmetic nodes around them is
not.

**Move it to compile time if the point is known.** Every entry point is
`constexpr`; see the next section.

---

## Compile-time use

Evaluation *and* differentiation are `constexpr`. A gradient, a Hessian or a
Taylor-mode derivative can be computed during constant evaluation and baked into
the binary:

```cpp
constexpr auto x = var<"x">;
constexpr auto y = var<"y">;

constexpr auto g  = Equation{x * y}.gradient(std::array{3.0, 4.0});
constexpr auto gf = Equation{x * y}.derivative_tensor<1>(std::array{3.0, 4.0});
constexpr auto H  = Equation{x * y}.derivative_tensor<2>(std::array{3.0, 4.0});
constexpr auto d2 = Equation{x * x * x}.univariate_derivative<2>(2.0);

static_assert(g[0] == 4.0 && gf[0] == 4.0 && H[0][1] == 1.0 && d2 == 12.0);
```

These are `constexpr`, not `consteval`: the same call serves a `static_assert`
and a value read from a file at run time.

---

## Building the project

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Benchmarks:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target benchmarks
./build/benchmarks
```

See [BENCHMARKS.md](benchmarks/BENCHMARKS.md) for the suite description and
results, and `src/main.cpp` for a runnable tour of every entry point.

### CMake options

| Option | Default | Meaning |
|---|---|---|
| `DDX_BUILD_BENCHMARKS` | `ON` | build the Google Benchmark targets |
| `DDX_BUILD_COMPARE` | `OFF` | build comparison benchmarks against autodiff (needs Eigen) |
| `ENABLE_NATIVE_ARCH` | `ON` | `-march=native` (falls back to `x86-64-v3`) |
| `DDX_FP_FLAGS` | `ON` | `-ffp-contract=fast -fno-math-errno` |
| `DDX_MDSPAN_MODE` | `auto` | `auto` / `std` / `vendored` — which `mdspan` to bind to |
| `DDX_DEDUCING_THIS` | `auto` | `auto` / `on` / `off` — accessor spelling (P0847) |

`-ffast-math` is not used and is not recommended: it changes derivative values.

---

## Cheat sheet

```cpp
// symbols and expressions
constexpr auto x = var<"x">;  constexpr auto y = var<"y">;
auto f = exp(x) * sin(y) + pow(x, 2.0);

// values
f.eval(1.0, 2.0);                              // positional (alphabetical order!)
f.eval(named<"y">(2.0), named<"x">(1.0));      // by name

// scalar derivatives
auto eq = Equation{f};
eq.evaluate(1.0, 2.0);
eq.gradient(1.0, 2.0);                         // reverse mode
eq.gradient<DiffMode::Symbolic>(1.0, 2.0);
eq.derivative_tensor<1>(1.0, 2.0);             // forward gradient
eq.derivative_tensor<2>(1.0, 2.0);             // forward Hessian
eq[idx<1>()];                                  // ∂f/∂x as an expression

// dual-valued symbols → reverse-mode Hessian
auto a = var<"x", dual>;  auto b = var<"y", dual>;
Equation{exp(a) * sin(b)}.hessian(1.0, 2.0);

// univariate higher order
Equation{sin(x)}.univariate_derivative<4>(1.0);

// vector systems
auto sys = Equation(x * y, sin(x) + y * y);
sys.jacobian(1.0, 2.0);
sys.derivative_tensor<2>(1.0, 2.0);            // per-output Hessians

// single expression → Equation, implicitly
Equation eq2 = x * y;                          // no braces needed
```

---

## Diagnostics

Common compile-time messages and what they mean:

| Message | Cause |
|---|---|
| `eval: supply exactly one value per symbol, in canonical order` | wrong number of positional values — canonical order is alphabetical |
| `eval: no value supplied for this symbol` | the point does not cover every symbol |
| no matching call to `hessian` | the symbols are not `dual`-valued — use `var<"x", dual>`, or use `derivative_tensor<2>` |

At run time the library throws only where a wrong point would otherwise pass
silently, and always `std::out_of_range`: an input range that supplies fewer
values than the expression has symbols.
Nothing on the evaluation path is `noexcept` for that reason — silently
differentiating at the wrong point is worse than an exception.

---

## License

See [LICENSE.txt](LICENSE.txt). Suggestions and pull requests are welcome.
