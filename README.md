[![CMake](https://github.com/reach2sayan/ddx/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/reach2sayan/ddx/actions/workflows/cmake-multi-platform.yml)
[![C++](https://img.shields.io/badge/C++-%2300599C.svg?logo=c%2B%2B&logoColor=white)](#)

# ddx

A C++23 library for differentiating expressions. Write a function over named
symbols and ask it for values, gradients, Jacobians, Hessians, higher-order
derivative tensors, or the derivative expressions themselves — at run time, in a
`static_assert`, or over a batch of thousands of points.

```cpp
#include "ddx.hpp"
using namespace ddx;

constexpr std::array ts{0.0, 1.0, 2.0, 3.0};
constexpr std::array ys{2.00, 1.21, 0.74, 0.45};

// residual sum of squares for the fit m(t) = a·exp(−b·t)
template <std::size_t I>
constexpr auto residual(auto a, auto b) { return ys[I] - a * exp(-b * ts[I]); }

template <std::size_t... I>
constexpr auto rss(auto a, auto b, std::index_sequence<I...>) {
  return ((residual<I>(a, b) * residual<I>(a, b)) + ...);
}

const auto eq = Equation{rss(var<"a">, var<"b">, std::make_index_sequence<4>{})};

eq.evaluate(2.0, 0.5);      // 4.1343959887091816e-05
eq.gradient(2.0, 0.5);      // {-0.0010757424682184863, 0.015067847559529016}
```

## The manuals

| Manual | Covers |
|---|---|
| [Compile-time expressions](include/symbolic/README.md) | symbols, points, `Equation`, values, gradients, Jacobians, the partial expressions, `Map`, printing |
| [Forward mode](include/dual/README.md) | `hessian`, `derivative_tensor<K>`, `univariate_derivative<K>`, `eval_with_tangent`, tensors |
| [Runtime expressions](include/rt/README.md) | expressions assembled while the program runs, the batch column API, the JIT |

---

## Requirements

- A C++23 compiler and standard library: **GCC 14+**, **Clang 19+** over
  libstdc++ 14+, **Clang 17+** over libc++ 17+, or MSVC (VS 2022,
  `/std:c++latest`).
- **CMake 3.26+** to build through CMake.
- Header-only dependencies are fetched at configure time, so the first configure
  wants a network. `-DDDX_BUILD_JIT=ON` additionally needs an LLVM 18–20
  installation, pointed at with `LLVM_DIR`.

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
| `DDX_BUILD_DUAL` | `ON` | forward mode — `hessian`, `derivative_tensor`, `univariate_derivative` |
| `DDX_BUILD_JIT` | `OFF` | compile the LLVM backend into the library |
| `DDX_SHARED_LIBS` | `ON` | build the library shared rather than static |
| `DDX_BUILD_BENCHMARKS` | `ON` | build the benchmark targets |
| `DDX_INSTALL` | on if top-level | generate the install and `find_package` rules |
| `ENABLE_NATIVE_ARCH` | `ON` | `-march=native`, falling back to `x86-64-v3` |
| `DDX_FP_FLAGS` | `ON` | `-ffp-contract=fast -fno-math-errno` |
| `DDX_NO_EXCEPTIONS` | `OFF` | build our own targets `-fno-exceptions` |
| `DDX_MDSPAN_MODE` | `auto` | `auto` / `std` / `vendored` — which `mdspan` to bind to |
| `DDX_DEDUCING_THIS` | `auto` | `auto` / `on` / `off` — accessor spelling (P0847) |

`-ffast-math` is not used and not recommended: it changes derivative values.

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
| `DiffMode` | `Symbolic` or `Reverse` |
| `dual`, `dual2nd` | the symbol value types `hessian` needs |

The operators, the math functions, `operator<<` and the `std::formatter`
specialisations are found by argument-dependent lookup, so they work on an
expression without being named here. `dual`, `dual2nd` and `dual_var_of` are
absent from a `-DDDX_BUILD_DUAL=OFF` build.

Runtime expressions are opt-in and live in `ddx::rt`, reached through
`#include "rt/equation.hpp"`.

## License

See [LICENSE.txt](LICENSE.txt). Suggestions and pull requests are welcome.
