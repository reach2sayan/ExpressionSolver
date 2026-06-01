[![CMake](https://github.com/reach2sayan/ExpressionSolver/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/reach2sayan/ExpressionSolver/actions/workflows/cmake-multi-platform.yml)
[![C++](https://img.shields.io/badge/C++-%2300599C.svg?logo=c%2B%2B&logoColor=white)](#)

# Expression Differentiator

A header-only C++20 library for symbolic expression trees, symbolic differentiation, and automatic differentiation — forward mode (dual numbers), reverse mode (adjoint backpropagation), and second-order (Hessian) computation.

## What it does

- Build typed expression trees from `Variable`, `Constant`, and operator nodes.
- Compute exact symbolic derivatives with product, quotient, and chain rules.
- Wrap a scalar expression in `Equation` to evaluate all partial derivatives symbolically.
- Wrap multiple outputs in `Equation<F0, F1, ...>` to evaluate Jacobians (f: ℝⁿ → ℝᵐ).
- Use `gradient<DiffMode::Reverse>(expr)` for scalar gradients.
- Use `hessian<DiffMode::Reverse>(expr)` for scalar Hessians (forward-over-reverse) — requires `Dual<T>`-valued variables.
- Use `derivative_tensor<K>(expr)` for forward-mode K-th order derivatives (gradient at `K=1`, Hessian at `K=2`); works on plain scalar variables.
- Use `univariate_derivative<K>(expr)` / `TaylorDual<S, N>` for efficient N-th order single-variable derivatives in O(N²) time.
- Use `eq.jacobian<DiffMode::Symbolic | Reverse>()` for vector Jacobians and `eq.derivative_tensor<K>()` for forward-mode Jacobians (`K=1`) / per-output Hessians (`K=2`); `eq.hessian<DiffMode::Reverse>()` gives reverse-mode per-output Hessians.
- Evaluate constant-only expressions at compile time.

## Requirements

- C++20 compiler (GCC 13+ or Clang 17+ recommended)
- CMake 3.20+
- Eigen 3.4+
- Boost headers with `boost::mp11`

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

On Windows, the GitHub workflow builds with MSVC and vcpkg-provided Boost headers. The local CMake setup also falls back to fetching `boost/mp11` when `Boost::headers` is not already available.

## Benchmarks

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target benchmarks
./build/benchmarks
```

To compare against [autodiff](https://autodiff.github.io/) v1.1.2:

```sh
cmake -S . -B build_compare -DCMAKE_BUILD_TYPE=Release
cmake --build build_compare --target benchmarks_compare
./build_compare/benchmarks_compare
```

See [BENCHMARKS.md](benchmarks/BENCHMARKS.md) for the full suite description, snapshots, and comparison results.

## Main types

### Primitives

| Type | Description |
|---|---|
| `Constant<T>` | Fixed numeric value; derivative is zero |
| `Variable<T, FixedString{"x"}>` | Named symbolic variable |

### Expression nodes

| Type | Description |
|---|---|
| `Expression<Op, LHS, RHS>` | Binary node |
| `MonoExpression<Op, Expr>` | Unary node |

### Supported operations

Arithmetic: `+`, `-`, `*`, `/`, unary `-`

Math functions: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`

### Dual numbers and higher-order types

| Type | Use |
|---|---|
| `Dual<T>` | First-order forward-mode AD; stores `{val, deriv}` |
| `Dual<Dual<T>>` | Second-order forward-mode AD (Hessian via forward-over-forward) |
| `nth_dual_t<S, N>` | N-th order nested dual; 2^N doubles, O(2^N) cost per operation |
| `TaylorDual<S, N>` | N-th order univariate AD; N+1 coefficients, O(N²) cost per operation |

### Variable storage hooks

A `Variable<T, "x", Storage>` can hold its value in two ways depending on `Storage`:

| Storage type | Owns value? | Description |
|---|---|---|
| `T` (default) | yes | Value stored directly in the variable |
| `FuncHook<T, GetF, AccumDF, ZeroDF>` | no | Reads/writes through user-supplied callables; all types deduced via CTAD |
| `VectorFuncHook<T, GetF, AccumDF, ZeroDF>` | no | Multi-slot callable hook; call `.element(i)` to get a per-index `FuncHook` |
| any `CHook<H,T>` type | no | Custom hook — implement `get_f()`, `accum_df(adj)`, `zero_df()` |

Hooked variables skip `update()` / `collect()` entirely — the expression tree reads and writes through the hook directly. `Variable::operator=` is a no-op for hook types without `set_f`.

### Higher-level wrappers

| Type | Description |
|---|---|
| `Equation<Expr>` | Scalar f: ℝⁿ → ℝ with symbolic partials |
| `Equation<F0, F1, ...>` | Vector f: ℝⁿ → ℝᵐ with Jacobian and Hessian methods |

## Convenience macros

| Syntax | Meaning |
|---|---|
| `PC(v)` | `Constant<decltype(v)>{v}` |
| `PV(v, "x")` | `Variable<decltype(v), FixedString{"x"}>{v}` |
| `PDV(v, "x")` | `Variable<Dual<decltype(v)>, FixedString{"x"}>{Dual<decltype(v)>{v, 0}}` |
| `3_ci` | `Constant<int>{3}` |
| `1.5_cd` | `Constant<double>{1.5}` |
| `4_vi` | `Variable<int, FixedString{"c"}>{4}` |
| `2.0_vd` | `Variable<double, FixedString{"v"}>{2.0}` |
| `idx<N>()` | Compile-time derivative index for scalar `Equation` |
| `IDX(N)` | Macro form of `idx<N>()` |

## Examples

> All examples assume `using namespace diff;`. The `PC` / `PV` / `PDV` / `IDX` macros and the `_cd` / `_vd` literals are global; everything else (`Equation`, `gradient`, `hessian`, `Dual`, `DiffMode`, `Variable`, `idx<>()`, …) lives in `namespace diff`. A bare scalar mixed with an expression or a `Dual` operand (e.g. `x + 2.0`) is automatically promoted to a zero-derivative constant, so literals never need explicit wrapping.

### Symbolic differentiation

`derivative()` is the single-variable symbolic derivative, so use a one-variable expression here; for partials of a multi-variable function use `Equation` (next section).

```cpp
auto x = PV(4.0, "x");
auto expr = x * x + PC(3.0) * x; // f(x) = x² + 3x

expr.eval();              // 28
expr.derivative().eval(); // f'(x) = 2x + 3 = 11
```

### Scalar partial derivatives

`Equation` wraps a scalar expression so all of its partial derivatives can be evaluated and indexed: build one with CTAD as `Equation(expr)`, then `eq[idx<0>()]` is the value and `eq[idx<k>()]` is the k-th partial.

```cpp
auto x = PV(4, "x");
auto y = PV(2, "y");
auto eq = Equation(x * y);

eq.evaluate();          // 8
eq[idx<1>()].eval();    // df/dx = y = 2
eq[idx<2>()].eval();    // df/dy = x = 4
auto [dx, dy] = eq.eval_derivatives();
```

### Reverse-mode gradient (scalar)

```cpp
auto x = PV(1.0, "x");
auto y = PV(2.0, "y");
auto g = gradient<DiffMode::Reverse>(exp(x) * sin(y));
// g[0] = ∂f/∂x,  g[1] = ∂f/∂y
```

### Forward-mode gradient (scalar)

Forward mode is exposed through `derivative_tensor<K>`; `K=1` returns the gradient. It seeds dual tangents internally, so the variables stay plain scalars.

```cpp
auto x = PV(3.0, "x");
auto y = PV(4.0, "y");
auto g = derivative_tensor<1>(x * y, std::array{3.0, 4.0});
// g[0] = ∂f/∂x = 4,  g[1] = ∂f/∂y = 3
```

### Scalar Hessian — reverse mode (forward-over-reverse)

Reverse-mode Hessians need `Dual`-valued variables (`PDV`) so tangents can be seeded.

```cpp
auto x = PDV(2.0, "x");
auto y = PDV(3.0, "y");
auto expr = x * y;

// Pass values explicitly:
auto H = hessian<DiffMode::Reverse>(expr, std::array{2.0, 3.0});
// Or read from variable state:
auto H2 = hessian<DiffMode::Reverse>(expr);
// H[i][j] = ∂²f/∂xᵢ∂xⱼ
```

### Scalar Hessian — forward mode (forward-over-forward, stateless)

`derivative_tensor<2>` returns the Hessian. It builds the nested duals internally, so the variables stay plain scalars.

```cpp
auto x = PV(2.0, "x");
auto y = PV(3.0, "y");
auto expr = x * y;

auto H = derivative_tensor<2>(expr, std::array{2.0, 3.0});
// Or read from variable state: auto H = derivative_tensor<2>(expr);
// H[i][j] = ∂²f/∂xᵢ∂xⱼ
```

### Vector Jacobian — symbolic

`Equation<F0, F1, ...>` wraps multiple outputs (f: ℝⁿ → ℝᵐ); `jacobian` returns `J[i][j] = ∂fᵢ/∂xⱼ`.

```cpp
auto x = PV(3.0, "x");
auto y = PV(4.0, "y");
auto ve = Equation(x + y, x * y);

auto J = ve.jacobian<DiffMode::Symbolic>();
// J[0][0]=1, J[0][1]=1, J[1][0]=4, J[1][1]=3
```

### Vector Jacobian — reverse-mode

```cpp
auto x = PV(2.0, "x");
auto y = PV(3.0, "y");
auto ve = Equation(x * y, sin(x) + y * y);

auto J = ve.jacobian<DiffMode::Reverse>();
// Evaluate at a new point:
auto J2 = ve.jacobian<DiffMode::Reverse>(std::array{1.0, 2.0});
```

### Vector Jacobian — forward-mode

Forward-mode Jacobian is `derivative_tensor<1>` on the `Equation`; variables stay plain scalars.

```cpp
auto x = PV(2.0, "x");
auto y = PV(3.0, "y");
auto ve = Equation(x * y, sin(x) + y * y);

auto J = ve.derivative_tensor<1>();
// Or with explicit point:
auto J2 = ve.derivative_tensor<1>(std::array{2.0, 3.0});
```

### Per-output Hessian — reverse mode (forward-over-reverse)

`H` is `std::array<std::array<std::array<S, n>, n>, m>` where `S` is the base scalar type.

Reverse-mode per-output Hessians need `Dual`-valued variables (`PDV`).

```cpp
auto x = PDV(2.0, "x");
auto y = PDV(3.0, "y");
auto ve = Equation(x * y, x * x);

auto H = ve.hessian<DiffMode::Reverse>();
// H[0][i][j] = ∂²(x*y)/∂xᵢ∂xⱼ  →  [[0,1],[1,0]]
// H[1][i][j] = ∂²(x²)/∂xᵢ∂xⱼ   →  [[2,0],[0,0]]

// Or with explicit point:
auto H2 = ve.hessian<DiffMode::Reverse>(std::array{2.0, 3.0});
```

### Per-output Hessian — forward mode (forward-over-forward)

Forward-mode per-output Hessian is `derivative_tensor<2>` on the `Equation`; variables stay plain scalars.

```cpp
auto x = PV(2.0, "x");
auto y = PV(3.0, "y");
auto ve = Equation(x * y, x * x);

auto H = ve.derivative_tensor<2>();
// H[k][i][j] = ∂²fₖ/∂xᵢ∂xⱼ
```

### FuncHook — callable value source

Lets a variable read from any callable — a sensor, a computed value, an external system. All callable types are deduced at construction (CTAD); no `std::function` overhead.

```cpp
double sensor_val = 0.0, sensor_grad = 0.0;

auto hook = FuncHook{
    [&]{ return sensor_val; },              // get_f
    [&](double adj){ sensor_grad += adj; }, // accum_df
    [&]{ sensor_grad = 0.0; }              // zero_df
};
Variable<double, FixedString{"x"}, decltype(hook)> x{hook};
auto expr = x * x;

using Syms = extract_symbols_from_expr_t<decltype(expr)>;
std::array<double, 1> grads{};

sensor_val = 4.0;
expr.eval();                                // returns 16.0
expr.backward(Syms{}, 1.0, grads);         // sensor_grad == 8.0
```

### VectorFuncHook — multi-slot callable hook

Like `FuncHook` but indexed; `element(i)` returns a `FuncHook` bound to slot `i`. Useful when a block of variables share the same get/accumulate logic over an array, matrix row, etc.

```cpp
std::array<double, 2> f{3.0, 4.0}, df{};

auto vh = VectorFuncHook{
    [&](std::size_t i){ return f[i]; },
    [&](std::size_t i, double adj){ df[i] += adj; },
    [&]{ df.fill(0.0); }
};

auto hx = vh.element(0);
auto hy = vh.element(1);
Variable<double, FixedString{"x"}, decltype(hx)> x{hx};
Variable<double, FixedString{"y"}, decltype(hy)> y{hy};

auto expr = x * y;
using Syms = extract_symbols_from_expr_t<decltype(expr)>;
std::array<double, 2> grads{};
expr.backward(Syms{}, 1.0, grads);  // df[0] == 4.0, df[1] == 3.0
```

### Higher-order univariate derivative (TaylorDual)

For N-th order derivatives of single-variable expressions, `TaylorDual<S,N>` stores N+1
Taylor coefficients and costs O(N²) per operation — far cheaper than the O(2^N) nested
`Dual<Dual<...>>` approach.

```cpp
auto x = PV(1.0, "x");
auto expr = sin(x);

// 4th derivative of sin at x=1.0: should be sin(1.0)
double d4 = univariate_derivative<4>(expr, 1.0);

// Or read the current variable value:
double d4_current = univariate_derivative<4>(expr);
```

For multi-variable expressions, use `derivative_tensor<K>` instead:

```cpp
auto x = PV(1.0, "x");
auto y = PV(2.0, "y");
auto expr = x * y + sin(x);

// Full rank-2 Hessian tensor: result[i][j] = ∂²f/∂xᵢ∂xⱼ
auto H = derivative_tensor<2>(expr, std::array{1.0, 2.0});
```

## API summary

### `DiffMode` enum

```cpp
enum class DiffMode { Symbolic, Forward, Reverse };
```

### `gradient.hpp` — scalar free functions

There are no `gradient`/`hessian` overloads for `DiffMode::Forward`; forward-mode derivatives are produced by `derivative_tensor<K>` (`K=1` gradient, `K=2` Hessian) and `univariate_derivative<K>`. `gradient` / `hessian` only exist for `DiffMode::Reverse`. The `expr`-only overloads read the current variable values via `collect()`.

| Call | Mode | Mutates | Cost |
|---|---|---|---|
| `gradient<DiffMode::Reverse>(expr)` | reverse | no | 1 backward pass |
| `hessian<DiffMode::Reverse>(expr, values)` | forward-over-reverse | yes (restored) | N backward passes |
| `hessian<DiffMode::Reverse>(expr)` | forward-over-reverse | yes (restored) | N backward passes |
| `derivative_tensor<K>(expr, values)` | forward | no | N^K `eval_seeded_as` passes |
| `derivative_tensor<K>(expr)` | forward | no | N^K `eval_seeded_as` passes |
| `univariate_derivative<K>(expr, x0)` | forward (TaylorDual) | no | 1 pass, O(K²) |
| `univariate_derivative<K>(expr)` | forward (TaylorDual) | no | 1 pass, O(K²) |

`hessian<DiffMode::Reverse>` requires `Dual<T>`-valued variables; `derivative_tensor` / `univariate_derivative` work on plain scalar variables.

### `Equation<F0, F1, ...>` — vector methods

| Method | Returns |
|---|---|
| `evaluate()` | `T` for `m=1`, else `std::array<T, m>` |
| `jacobian<DiffMode::Symbolic>()` | `std::array<std::array<T, n>, m>` (symbolic, compile-time only) |
| `jacobian<DiffMode::Reverse>([values])` | `std::array<std::array<T, n>, m>` — reverse-mode |
| `derivative_tensor<1>([values])` | `std::array<std::array<S, n>, m>` — forward-mode Jacobian |
| `hessian<DiffMode::Reverse>([values])` | `std::array<std::array<std::array<S, n>, n>, m>` — forward-over-reverse, requires `Dual<T>` |
| `derivative_tensor<2>([values])` | `std::array<std::array<std::array<S, n>, n>, m>` — forward-over-forward |

There are no `jacobian`/`hessian` overloads for `DiffMode::Forward`; use `derivative_tensor<K>` (which works on plain scalar variables). `S` is the base scalar type (extracted from any `Dual<…>` nesting).

## Notes

- Symbol order is derived from variable type labels and sorted at compile time.
- `Equation<F0, F1, ...>` deduces and unions all symbols across all component expressions.
- `hessian<DiffMode::Reverse>` (scalar and per-output) requires `Dual`-valued variables; the base scalar `S` is extracted automatically.
- `derivative_tensor<K>` and `univariate_derivative<K>` are fully stateless (`const` expr) and work on plain scalar variables — they build the dual tangents internally.
- `hessian<DiffMode::Reverse>` mutates the expression to seed dual tangents but restores on exit.

## Contributing

This is a personal learning project, but suggestions and pull requests are welcome.
