[![CMake](https://github.com/reach2sayan/ExpressionSolver/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/reach2sayan/ExpressionSolver/actions/workflows/cmake-multi-platform.yml)
[![C++](https://img.shields.io/badge/C++-%2300599C.svg?logo=c%2B%2B&logoColor=white)](#)

# Expression Differentiator

A header-only C++20 library for symbolic expression trees, symbolic differentiation, and automatic differentiation — forward mode (dual numbers), reverse mode (adjoint backpropagation), second-order (Hessian) computation, and parallel reverse-mode Jacobians.

## What it does

- Build typed expression trees from `Variable`, `Constant`, and operator nodes.
- Compute exact symbolic derivatives with product, quotient, and chain rules.
- Wrap a scalar expression in `Equation` to evaluate all partial derivatives symbolically.
- Wrap multiple outputs in `Equation<F0, F1, ...>` to evaluate Jacobians (f: ℝⁿ → ℝᵐ).
- Use `Dual<T>` for forward-mode AD; nest as `Dual<Dual<T>>` for second-order.
- Use `TaylorDual<S, N>` for efficient N-th order univariate derivatives in O(N²) time.
- Select differentiation mode with `DiffMode::Forward` / `DiffMode::Reverse` / `DiffMode::Symbolic` / `DiffMode::ParallelReverse`.
- Use `gradient<DiffMode>` for scalar gradients; `hessian<DiffMode>` for scalar Hessians.
- Use `eq.jacobian<DiffMode>()` for vector Jacobians; `eq.hessian<DiffMode>()` for per-output Hessians.
- Evaluate constant-only expressions at compile time.
- Compute multi-output Jacobians in parallel using pre-spawned workers and `std::barrier` (zero allocation in the hot path).

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

See [BENCHMARKS.md](benchmarks/BENCHMARKS.md) for the full suite description, snapshots, comparison results, and notes on the parallel Jacobian experiment.

## Main types

### Primitives

| Type | Description |
|---|---|
| `Constant<T>` | Fixed numeric value; derivative is zero |
| `Variable<T, 'x'>` | Named symbolic variable |
| `RuntimeVariable<T>` | Index-addressed variable; symbol list resolved at runtime |

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

### Higher-level wrappers

| Type | Description |
|---|---|
| `Equation<Expr>` | Scalar f: ℝⁿ → ℝ with symbolic partials |
| `Equation<F0, F1, ...>` | Vector f: ℝⁿ → ℝᵐ with Jacobian and Hessian methods |

## Convenience macros

| Syntax | Meaning |
|---|---|
| `PC(v)` | `Constant<decltype(v)>{v}` |
| `PV(v, 'x')` | `Variable<decltype(v), 'x'>{v}` |
| `PDV(v, 'x')` | `Variable<Dual<decltype(v)>, 'x'>{Dual<decltype(v)>{v, 0}}` |
| `RV(v, i)` | `RuntimeVariable<decltype(v)>{v, i}` |
| `3_ci` | `Constant<int>{3}` |
| `1.5_cd` | `Constant<double>{1.5}` |
| `4_vi` | `Variable<int, 'c'>{4}` |
| `2.0_vd` | `Variable<double, 'v'>{2.0}` |
| `idx<N>()` | Compile-time derivative index for scalar `Equation` |
| `IDX(N)` | Macro form of `idx<N>()` |

## Examples

### Symbolic differentiation

```cpp
auto x = PV(4.0, 'x');
auto y = PV(2.0, 'y');
auto expr = x * y + PC(3.0) * x * y * y;

expr.eval();            // 56
expr.derivative().eval(); // df/dx at current point
```

### Scalar partial derivatives

```cpp
auto x = PV(4, 'x');
auto y = PV(2, 'y');
auto eq = Equation(x * y);

eq.eval();              // 8
eq[idx<1>()].eval();    // df/dx = y = 2
eq[idx<2>()].eval();    // df/dy = x = 4
auto [dx, dy] = eq.eval_derivatives();
```

### Reverse-mode gradient (scalar)

```cpp
auto x = PV(1.0, 'x');
auto y = PV(2.0, 'y');
auto g = gradient<DiffMode::Reverse>(exp(x) * sin(y));
// g[0] = ∂f/∂x,  g[1] = ∂f/∂y

// Or with Dual<double> variables:
auto xd = PDV(1.0, 'x');
auto g2 = gradient<DiffMode::Reverse>(exp(xd) * sin(PDV(2.0, 'y')));
```

### Forward-mode gradient (scalar)

```cpp
using D = Dual<double>;
Variable<D, 'x'> x{D{3.0}};
Variable<D, 'y'> y{D{4.0}};
auto g = gradient<DiffMode::Forward>(x * y, std::array{3.0, 4.0});
// g[0] = ∂f/∂x,  g[1] = ∂f/∂y
```

### Scalar Hessian — reverse mode (forward-over-reverse)

```cpp
using D = Dual<double>;
Variable<D, 'x'> x{D{2.0}};
Variable<D, 'y'> y{D{3.0}};
auto expr = x * y;

// Pass values explicitly:
auto H = hessian<DiffMode::Reverse>(expr, std::array{2.0, 3.0});
// Or read from variable state:
auto H2 = hessian<DiffMode::Reverse>(expr);
// H[i][j] = ∂²f/∂xᵢ∂xⱼ
```

### Scalar Hessian — forward mode (forward-over-forward, stateless)

```cpp
using DD = Dual<Dual<double>>;
using D  = Dual<double>;
Variable<DD, 'x'> x{DD{D{2.0}, D{}}};
Variable<DD, 'y'> y{DD{D{3.0}, D{}}};
auto expr = x * y;

auto H = hessian<DiffMode::Forward>(expr, std::array{2.0, 3.0});
// Or: auto H = hessian<DiffMode::Forward>(expr);
```

### Vector Jacobian — symbolic

```cpp
auto x = PV(3.0, 'x');
auto y = PV(4.0, 'y');
auto ve = Equation(x + y, x * y);

auto J = ve.jacobian<DiffMode::Symbolic>();
// J(0,0)=1, J(0,1)=1, J(1,0)=4, J(1,1)=3
```

### Vector Jacobian — reverse-mode

```cpp
auto x = PV(2.0, 'x');
auto y = PV(3.0, 'y');
auto ve = Equation(x * y, sin(x) + y * y);

auto J = ve.jacobian<DiffMode::Reverse>();
// Evaluate at a new point:
auto J2 = ve.jacobian<DiffMode::Reverse>(Eigen::Vector2d{1.0, 2.0});
```

### Vector Jacobian — forward-mode

```cpp
using D = Dual<double>;
Variable<D, 'x'> x{D{2.0}};
Variable<D, 'y'> y{D{3.0}};
auto ve = Equation(x * y, sin(x) + y * y);

auto J = ve.jacobian<DiffMode::Forward>();
// Or with explicit point:
auto J2 = ve.jacobian<DiffMode::Forward>(Eigen::Vector2d{2.0, 3.0});
```

### Per-output Hessian — reverse mode (forward-over-reverse)

`H` is `std::array<Eigen::Matrix<S, n, n>, m>` where `S` is the base scalar type.

```cpp
using D = Dual<double>;
Variable<D, 'x'> x{D{2.0}};
Variable<D, 'y'> y{D{3.0}};
auto ve = Equation(x * y, x * x);

auto H = ve.hessian<DiffMode::Reverse>();
// H[0](i,j) = ∂²(x*y)/∂xᵢ∂xⱼ  →  [[0,1],[1,0]]
// H[1](i,j) = ∂²(x²)/∂xᵢ∂xⱼ   →  [[2,0],[0,0]]

// Or with explicit point:
auto H2 = ve.hessian<DiffMode::Reverse>(Eigen::Vector2d{2.0, 3.0});
```

### Per-output Hessian — forward mode (forward-over-forward)

```cpp
using DD = Dual<Dual<double>>;
using D  = Dual<double>;
Variable<DD, 'x'> x{DD{D{2.0}, D{}}};
Variable<DD, 'y'> y{DD{D{3.0}, D{}}};
auto ve = Equation(x * y, x * x);

auto H = ve.hessian<DiffMode::Forward>();
```

### Higher-order univariate derivative (TaylorDual)

For N-th order derivatives of single-variable expressions, `TaylorDual<S,N>` stores N+1
Taylor coefficients and costs O(N²) per operation — far cheaper than the O(2^N) nested
`Dual<Dual<...>>` approach.

```cpp
auto x = PV(1.0, 'x');
auto expr = sin(x);

// 4th derivative of sin at x=1.0: should be sin(1.0)
double d4 = univariate_derivative<4>(expr, 1.0);

// Or read the current variable value:
double d4_current = univariate_derivative<4>(expr);
```

For multi-variable expressions, use `derivative_tensor<K>` instead:

```cpp
auto x = PV(1.0, 'x');
auto y = PV(2.0, 'y');
auto expr = x * y + sin(x);

// Full rank-2 Hessian tensor: result[i][j] = ∂²f/∂xᵢ∂xⱼ
auto H = derivative_tensor<2>(expr, std::array{1.0, 2.0});
```

### Parallel reverse-mode Jacobian

`Equation` pre-spawns `output_dim - 1` worker threads at construction time. Each
`jacobian<DiffMode::ParallelReverse>()` call uses two `std::barrier` phase transitions —
no dynamic allocation in the hot path.

```cpp
auto x = PV(1.0, 'x'); auto y = PV(0.5, 'y');
auto z = PV(1.7, 'z'); auto w = PV(0.3, 'w');
auto ve = Equation(exp(x*z) + sin(y*w),
                   x*y - z*w,
                   sin(x) + cos(y) + z);

auto J = ve.jacobian<DiffMode::ParallelReverse>();
// J[i][j] = ∂fᵢ/∂xⱼ
```

Workers are joined automatically when the `Equation` is destroyed (`std::jthread`).
Do not move an `Equation` after construction when `output_dim > 1`: worker lambdas
capture `this`.

### Runtime variables (dynamic input dimension)

```cpp
auto x = RV(2.0, 0);
auto y = RV(3.0, 1);
auto eq = Equation(2, x * y, x + y);   // 2 inputs, 2 outputs

auto J = eq.jacobian<DiffMode::Reverse>();
eq.update(Eigen::Vector2d{4.0, 5.0});
```

## API summary

### `DiffMode` enum

```cpp
enum class DiffMode { Symbolic, Forward, Reverse, ParallelReverse };
```

### `gradient.hpp` — scalar free functions

| Call | Mode | Mutates | Cost |
|---|---|---|---|
| `gradient<DiffMode::Reverse>(expr)` | reverse | no | 1 backward pass |
| `gradient<DiffMode::Forward>(expr, values)` | forward | no | N `eval_seeded` passes |
| `hessian<DiffMode::Reverse>(expr, values)` | forward-over-reverse | yes (restored) | N backward passes |
| `hessian<DiffMode::Reverse>(expr)` | forward-over-reverse | yes (restored) | N backward passes |
| `hessian<DiffMode::Forward>(expr, values)` | forward-over-forward | no | N² `eval_seeded` passes |
| `hessian<DiffMode::Forward>(expr)` | forward-over-forward | no | N² `eval_seeded` passes |
| `derivative_tensor<K>(expr, values)` | forward | no | N^K `eval_seeded_as` passes |
| `univariate_derivative<K>(expr, x0)` | forward (TaylorDual) | no | 1 pass, O(K²) |
| `univariate_derivative<K>(expr)` | forward (TaylorDual) | no | 1 pass, O(K²) |

### `Equation<F0, F1, ...>` — vector methods

| Method | Returns |
|---|---|
| `evaluate()` | `std::array<T, m>` |
| `jacobian<DiffMode::Symbolic>()` | `Eigen::Matrix<T, m, n>` (symbolic, compile-time only) |
| `jacobian<DiffMode::Reverse>([values])` | `Eigen::Matrix<T, m, n>` — reverse-mode, serial |
| `jacobian<DiffMode::ParallelReverse>()` | `std::array<std::array<T, n>, m>` — parallel reverse, barrier-based |
| `jacobian<DiffMode::Forward>([values])` | `Eigen::Matrix<S, m, n>` — forward-mode, requires `Dual<T>` |
| `hessian<DiffMode::Reverse>([values])` | `std::array<Eigen::Matrix<S, n, n>, m>` — forward-over-reverse, requires `Dual<T>` |
| `hessian<DiffMode::Forward>([values])` | `std::array<Eigen::Matrix<S, n, n>, m>` — forward-over-forward, requires `Dual<Dual<T>>` |

`S` is the base scalar type (extracted from `Dual<S>` or `Dual<Dual<S>>`).

## Notes

- Symbol order is derived from variable type labels and sorted at compile time.
- `Equation<F0, F1, ...>` deduces and unions all symbols across all component expressions.
- Hessian methods require `Dual`-valued variables; the base scalar `S` is extracted automatically.
- `hessian<DiffMode::Forward>` and `gradient<DiffMode::Forward>` are fully stateless (`const` expr).
- `hessian<DiffMode::Reverse>` (scalar) mutates the expression to seed dual tangents but restores on exit.
- `RuntimeVariable` nodes have `input_dim == 0`; the number of inputs is passed to the `Equation` constructor.

## Contributing

This is a personal learning project, but suggestions and pull requests are welcome.
