[![CMake](https://github.com/reach2sayan/ExpressionSolver/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/reach2sayan/ExpressionSolver/actions/workflows/cmake-multi-platform.yml)
[![C++](https://img.shields.io/badge/C++-%2300599C.svg?logo=c%2B%2B&logoColor=white)](#)

# Expression Differentiator

A header-only C++23 library for symbolic expression trees and automatic
differentiation: symbolic derivatives, forward mode (dual numbers), reverse mode
(adjoint sweeps), Hessians, higher-order derivative tensors, and Taylor-mode
univariate derivatives — all usable at run time or inside `constexpr`.

```cpp
#include "expression_differentiator.hpp"
using namespace diff;

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
6. [Dual numbers](#dual-numbers)
7. [Differentiating a plain callable](#differentiating-a-plain-callable)
8. [Sparse Hessians](#sparse-hessians)
9. [Compile-time use](#compile-time-use)
10. [Building the project](#building-the-project)
11. [Cheat sheet](#cheat-sheet)
12. [Diagnostics](#diagnostics)

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
add_subdirectory(ExpressionDifferentiator)
target_link_libraries(my_app PRIVATE ExpressionDifferentiator::diff)   # or: diff
```

Then include the headers you need. These three cover the whole manual:

```cpp
#include "expr/bound.hpp"        // expressions, eval, named/values/bind
#include "expr/equation.hpp"     // Equation — every derivative entry point
#include "drivers/hessian.hpp"   // gradient/hessian over plain callables

using namespace diff;            // assumed by every example below
```

`expression_differentiator.hpp`, at the repository root, is an umbrella header
that pulls in everything. Add the root to your include path to use it:

```cmake
target_include_directories(my_app PRIVATE ExpressionDifferentiator)
```

```cpp
#include "expression_differentiator.hpp"   // everything
```

What lives where:

| Header | Contents |
|---|---|
| `expr/expressions.hpp`, `expr/values.hpp` | `var`, `Variable`, `Constant`/`Lit`, operators |
| `expr/bound.hpp` | `eval`, `named`, `values`, `bind`, `symbol_order` |
| `expr/equation.hpp` | `Equation` — every derivative entry point |
| `expr/format.hpp` | `std::format` / `operator<<` for expressions |
| `dual/dual.hpp` | `Dual`, `dual`, `dual2nd`, `nth_dual_t` |
| `dual/taylor_dual.hpp` | `TaylorDual` |
| `drivers/hessian.hpp` | `gradient` / `hessian` over plain callables and graphs |
| `drivers/symbolic.hpp` | `sparse_hessian`, `symbol_order`, `DiffMode` |

Everything lives in `namespace diff`, except the convenience macros (`PC`, `PV`,
`PDV`) and the user-defined literals (`_cd`, `_ci`, `_vd`, `_vi`), which are
global.

---

## Building expressions

### Symbols

A variable is a **name**, spelled `var<"x">`:

```cpp
constexpr auto x = var<"x">;              // Variable<double, "x">
constexpr auto n = var<"n", int>;         // another scalar type
auto xd = var<"x", Dual<double>>;         // dual-valued symbol (needed for hessian())
```

Equivalent spellings, when you would rather deduce the scalar type from an
exemplar value:

| Syntax | Means |
|---|---|
| `var<"x">` | `Variable<double, FixedString{"x"}>{}` |
| `var<"x", T>` | `Variable<T, FixedString{"x"}>{}` |
| `PV(1.0, "x")` | same as `var<"x">` — the `1.0` supplies only the *type* |
| `PDV(1.0, "x")` | same as `var<"x", Dual<double>>` |
| `2.0_vd` | a `double` symbol named `"v"` |
| `4_vi` | an `int` symbol named `"c"` |

### Constants

| Syntax | Means |
|---|---|
| `PC(3.0)` | `Constant<double>{3.0}` — a value stored in the tree |
| `1.5_cd` | `Constant<double>{1.5}` |
| `3_ci` | `Constant<int>{3}` |

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

Expressions, `Equation`s, `Dual`s and `TaylorDual`s are all formattable and
streamable:

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

Every entry point that needs numbers accepts the point in four interchangeable
spellings, both as a member and as a free function:

```cpp
auto f = x * y + PC(3.0) * x;                       // f(x, y) = xy + 3x

f.eval(4.0, 2.0);                                   // positional, canonical order
f.eval(named<"y">(2.0), named<"x">(4.0));           // by name, order-independent
f.eval(values(named<"x">(4.0), named<"y">(2.0)));   // a stored ValueMap
f.eval(std::array{4.0, 2.0});                       // any input range
eval(f, 4.0, 2.0);                                  // free-function spelling
```

All of these give `20`. The same spellings work for `Equation::evaluate`,
`gradient`, `jacobian`, `hessian`, `derivative_tensor` — one normaliser serves
them all.

### Canonical order

**Canonical order is alphabetical by symbol name, not the order you wrote them
in.** For `f(w, x, y, z)` the positional form expects `w` first. Ask when in
doubt:

```cpp
symbol_order<decltype(f)>();   // std::array<std::string_view, 2>{"x", "y"}
symbol_order(f);               // same thing, deduced from a value
```

The positional and range forms check the count at compile time, so a missing or
extra value is a compile error. The named and `ValueMap` forms bind by name and
are immune to ordering entirely — prefer them when the symbol set is large. A
non-sized range that runs short throws `std::out_of_range` (a compile error
during constant evaluation).

### Storing a point: `values` and `bind`

```cpp
auto point = values(named<"x">(4.0), named<"y">(2.0));   // ValueMap
point.get<"x">();                                        // 4.0
point.set<"y">(3.0);

using namespace diff::literals;
point["x"_s] = 5.0;                                      // subscript spelling
```

`bind` pairs an expression with a point so that `eval()` becomes nullary:

```cpp
auto bf = bind(f, named<"x">(4.0), named<"y">(2.0));
bf.eval();          // 20
bf.get<"x">();      // 4.0
bf.set<"y">(3.0);
bf.eval();          // 24
```

A `Bound` costs exactly its map — the expression contributes no bytes. The map
may be a superset of the expression's symbols, so one point can serve several
expressions; a map *missing* a symbol is a compile error.

An expression with no free symbols needs no point at all:

```cpp
constexpr auto k = PC(2.0) * PC(3.0);
static_assert(k.eval() == 6.0);
```

---

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

Every one of them also takes positional, named or `ValueMap` arguments:

```cpp
eq.gradient(1.0, 2.0);
eq.gradient(named<"y">(2.0), named<"x">(1.0));
```

#### Hessians

There are two Hessians, and they differ in what the symbols must be:

```cpp
// forward-over-forward: plain scalar symbols
auto H1 = Equation{x * y}.derivative_tensor<2>(std::array{2.0, 3.0});

// forward-over-reverse: Dual-valued symbols
auto a = var<"x", Dual<double>>;
auto b = var<"y", Dual<double>>;
auto H2 = Equation{a * b}.hessian(std::array{2.0, 3.0});
```

`hessian()` needs `Dual<T>`-valued symbols because it seeds tangents into the
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

The reverse-mode per-output Hessian is available too, again on `Dual`-valued
symbols:

```cpp
auto a = var<"x", Dual<double>>;
auto b = var<"y", Dual<double>>;
auto H = Equation(a * b, a * a).hessian(std::array{2.0, 3.0});
// H[0][i][j] = ∂²(xy)/∂xᵢ∂xⱼ,  H[1][i][j] = ∂²(x²)/∂xᵢ∂xⱼ
```

### Choosing a mode

| Call | Mode | Symbols | Cost |
|---|---|---|---|
| `gradient(pt)` | reverse | plain or dual | one backward sweep |
| `gradient<DiffMode::Symbolic>(pt)` | symbolic | plain or dual | evaluates n partial trees |
| `derivative_tensor<1>(pt)` | forward | plain | n forward sweeps |
| `hessian(pt)` | forward-over-reverse | **`Dual<T>` required** | n backward sweeps |
| `derivative_tensor<2>(pt)` | forward-over-forward | plain | one sweep per index pair |
| `derivative_tensor<K>(pt)` | forward | plain | one sweep per distinct K-index |
| `univariate_derivative<K>(x0)` | Taylor | plain, n = 1 | one sweep, O(K²) |

Rules of thumb: reverse mode for a gradient of many variables; `hessian()` for a
second derivative when you can afford `Dual`-valued symbols;
`univariate_derivative` whenever the function has one variable; the symbolic mode
when you want to *see* the derivative rather than just evaluate it.

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

It returns a `Dual<T>`, seeded on the named symbol. Arguments are positional, in
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

`S` is the base scalar type: for a `Dual<double>`-valued expression it is
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

### Driver tuples

The runtime drivers (next section) return plain tuples that destructure alike:

```cpp
auto [v, g, H, n] = hessian(f, x);      // callable: unique_ptr<double[]> buffers + extent
auto [v, g, H]    = hessian(expr, x);   // expression graph: std::array, extent is n
```

The Hessian buffer is **row-major**: element (i, j) is `H[i * n + j]`. It is
exactly symmetric.

---

## Dual numbers

`Dual<T>` is a value paired with its derivative, `{val, deriv}`, with every
operator and math function overloaded to carry the chain rule. You can use it
directly — no expression tree involved:

```cpp
Dual<double> u{2.0, 1.0};        // value 2, tangent 1  (seeded on this variable)
Dual<double> v{3.0};             // value 3, tangent 0  (a constant)

auto w = u * u + sin(v);
w.value();      // 4 + sin(3)
w.deriv();      // d/du of the above = 4
```

| Spelling | Meaning |
|---|---|
| `Dual<T>{v, d}` | value `v`, derivative `d` |
| `Dual<T>{v}` | derivative zero — an implicit conversion, so `2.0 + u` works |
| `u.value()`, `u.deriv()` | the two components (readable and writable) |
| `u.get<0>()`, `u.get<1>()` | the same, indexed; also `auto [a, b] = u;` |
| `val(u)` | peel *every* dual layer down to the base scalar |

Comparisons (`==`, `<=>`) compare the value components, so `max`, `min` and
ordinary branching work on duals.

### Higher orders

| Type | Use | Cost per operation |
|---|---|---|
| `Dual<T>` / `dual` | first-order forward mode (`dual` = `Dual<double>`) | O(1) |
| `Dual<Dual<T>>` / `dual2nd` | second order — Hessians via forward-over-forward | O(1), 4 scalars |
| `nth_dual_t<S, N>` | N-th order nested dual | O(2^N), 2^N scalars |
| `TaylorDual<S, N>` | N-th order **univariate** | O(N²), N+1 scalars |

```cpp
dual2nd a{1.0};
a.value().deriv() = 1.0;         // seed the inner tangent
a.deriv().value() = 1.0;         // seed the outer tangent
auto r = a * a * a;
r.deriv().deriv();               // d²(x³)/dx² at 1 = 6
```

`TaylorDual<S, N>` stores the *normalised* Taylor coefficients
`c[k] = f⁽ᵏ⁾(x)/k!`, which is why the N-th derivative is `c[N] * N!`:

```cpp
TaylorDual<double, 4> t{2.0};    // value 2
t.c[1] = 1.0;                    // seed: differentiate with respect to this variable
auto r = sin(t) * t;
double third = r.c[3] * 6.0;     // 3! = 6
```

For expressions this is all packaged up — `univariate_derivative<K>` does the
seeding and the factorial for you.

Helper traits, when you write generic code over these types:

| Trait | Meaning |
|---|---|
| `is_dual_v<T>`, `DualLike<T>` | is `T` a `Dual<...>`? |
| `dual_value_t<Dual<T>>` | the component type `T` |
| `dual_scalar_t<T>` | peel **one** `Dual` layer, or `T` unchanged |
| `scalar_base_t<T>` | peel **every** layer — the base scalar |
| `dual_depth_v<T>` | how many layers deep |
| `Numeric<T>` | the concept every scalar in the library satisfies |

Duals and Taylor duals print in ε notation:

```cpp
std::format("{}", Dual<double>{1.5, 2.0});       // "1.5+2ε"
std::format("{:.2f}", Dual<double>{1.5, 2.0});   // "1.50+2.00ε"

TaylorDual<double, 2> t{};
t.c = {1.0, 2.0, 3.0};
std::format("{}", t);                            // "1+2ε+3ε^2"
```

The coefficients printed are the stored, normalised ones — `f⁽ᵏ⁾/k!`, not the
derivatives themselves.

---

## Differentiating a plain callable

You do not need an expression tree. Any callable that accepts a pointer to
seeded dual degrees of freedom can be differentiated directly:

```cpp
std::vector<double> x{1.0, 2.0};
auto energy = [](const auto *d) { return d[0] * d[0] * d[1]; };

auto g = gradient(energy, x);                 // std::vector<double> {4, 1}
auto [v, grad, H, n] = hessian(energy, x);    // v == 2, H[0 * n + 1] == 2
```

The callable is invoked with `const dual*` by `gradient` and with `const dual2nd*`
by `hessian`, so write it as a template (`const auto *`) or overload it. The
point is a `std::span<const double>` parameter, which is non-deduced — a
`std::vector`, `std::array`, C array or contiguous view binds with nothing spelled
at the call site.

### Differentiating a subset

Pass an index range as a third argument to differentiate only some variables:

```cpp
std::array<std::size_t, 2> active{0, 2};
auto g = gradient(energy, x, active);          // 2 entries, for x[0] and x[2]
auto h = hessian(energy, x, active);           // 2 × 2
```

### Reusing scratch

The owning forms allocate. For a loop over many points, hand in a workspace and
your own output buffer instead:

```cpp
GradientWorkspace ws;
std::vector<double> out(x.size());
for (const auto &pt : points) {
    gradient(energy, pt, std::views::iota(0uz, pt.size()), ws, out);
}
```

The workspace allocates on first use and never again (small points are seeded
into an inline block and never touch the heap).

### Expression graphs work here too

An expression is accepted by the same entry points, and the extent is then a
compile-time constant, so nothing is allocated:

```cpp
auto expr = var<"x", Dual<double>> * var<"x", Dual<double>> * var<"y", Dual<double>>;
auto [v, grad, H] = hessian(expr, x);       // std::array results
```

`hessian()` picks the driver for you: an expression graph over all of its symbols
takes the forward-over-reverse path (n backward sweeps); anything else — a raw
callable, or a graph with a restricted `active` set — takes the probe driver
(n(n+1)/2 evaluations). Both return an exactly symmetric Hessian.

---

## Sparse Hessians

For an expression graph the *sparsity pattern is a property of the type*, read
off the tree at compile time. `sparse_hessian` computes and stores only the
structurally nonzero entries:

```cpp
auto H = sparse_hessian(expr, x);

H.rows;                  // extent — a compile-time constant
decltype(H)::nnz;        // stored entries — a compile-time constant
H.outer(), H.inner();    // std::span<const int>, the CSC structure, constexpr
H.values();              // std::span<const double>, the nnz values
H[i, j];                 // read it as though it were dense
decltype(H)::structural(i, j);   // is (i, j) in the pattern at all?
```

`outer` and `inner` are a standard compressed-column (CSC) triple: column `j`
occupies `[outer()[j], outer()[j + 1])`, and `inner()[k]` is the row of stored
value `k`. Every sparse linear-algebra library consumes that directly — for
Eigen it is one line:

```cpp
Eigen::Map<const Eigen::SparseMatrix<double>> m{
    H.rows, H.rows, H.nnz, H.outer().data(), H.inner().data(), H.values().data()};
```

The library itself has no linear-algebra dependency; results leave as pointers,
arrays and spans, and mapping your own matrix type onto them is the one line
above.

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
and a value read from a file at run time. The runtime drivers over callables
(`gradient(f, x)`, `hessian(f, x)`) allocate, so those are run time only.

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
| `DIFF_BUILD_BENCHMARKS` | `ON` | build the Google Benchmark targets |
| `DIFF_BUILD_COMPARE` | `OFF` | build comparison benchmarks against autodiff (needs Eigen) |
| `ENABLE_NATIVE_ARCH` | `ON` | `-march=native` (falls back to `x86-64-v3`) |
| `DIFF_FP_FLAGS` | `ON` | `-ffp-contract=fast -fno-math-errno` |
| `DIFF_MDSPAN_MODE` | `auto` | `auto` / `std` / `vendored` — which `mdspan` to bind to |
| `DIFF_DEDUCING_THIS` | `auto` | `auto` / `on` / `off` — accessor spelling (P0847) |

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
symbol_order(f);                               // the canonical order

// scalar derivatives
auto eq = Equation{f};
eq.evaluate(1.0, 2.0);
eq.gradient(1.0, 2.0);                         // reverse mode
eq.gradient<DiffMode::Symbolic>(1.0, 2.0);
eq.derivative_tensor<1>(1.0, 2.0);             // forward gradient
eq.derivative_tensor<2>(1.0, 2.0);             // forward Hessian
eq[idx<1>()];                                  // ∂f/∂x as an expression

// dual-valued symbols → reverse-mode Hessian
auto a = var<"x", Dual<double>>;  auto b = var<"y", Dual<double>>;
Equation{exp(a) * sin(b)}.hessian(1.0, 2.0);

// univariate higher order
Equation{sin(x)}.univariate_derivative<4>(1.0);

// vector systems
auto sys = Equation(x * y, sin(x) + y * y);
sys.jacobian(1.0, 2.0);
sys.derivative_tensor<2>(1.0, 2.0);            // per-output Hessians

// plain callables
auto g = gradient([](const auto *d) { return d[0] * d[1]; }, std::vector{1.0, 2.0});
auto [v, grad, H, n] = hessian([](const auto *d) { return d[0] * d[1]; },
                               std::vector{1.0, 2.0});

// sparse
auto S = sparse_hessian(a * b, std::vector{1.0, 2.0});
```

---

## Diagnostics

Common compile-time messages and what they mean:

| Message | Cause |
|---|---|
| `eval: supply exactly one value per symbol, in canonical order` | wrong number of positional values — check `symbol_order<Expr>()` |
| `eval: no value supplied for this symbol` | the point does not cover every symbol |
| `ValueMap: symbol not present in map` | `get<"...">` on a name the map does not hold |
| `bind: the map does not supply every symbol the expression uses` | the map is missing a symbol |
| `values: duplicate symbol` | the same name passed twice to `values(...)` |
| no matching call to `hessian` | the symbols are not `Dual`-valued — use `var<"x", Dual<double>>`, or use `derivative_tensor<2>` |

At run time the library throws only where a wrong point would otherwise pass
silently, and always `std::out_of_range`: an input range that supplies fewer
values than the expression has symbols, and `hessian(graph, x)` handed a point
that is not one value per symbol or an `active` index that names no symbol.
Nothing on the evaluation path is `noexcept` for that reason — silently
differentiating at the wrong point is worse than an exception.

---

## License

See [LICENSE.txt](LICENSE.txt). Suggestions and pull requests are welcome.
