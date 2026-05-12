# Benchmarks

## Running the suite

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target benchmarks
./build/benchmarks
```

Export machine-readable JSON:

```sh
cmake --build build --target benchmark_json
# writes build/benchmark-results/benchmarks.json
```

Filter to a subset:

```sh
./build/benchmarks --benchmark_filter='.*F4.*'
```

PowerShell:

```powershell
.\build-win\Release\benchmarks.exe "--benchmark_filter=F1|F2|F3|F4" --benchmark_min_time=0.05s
```

A manual GitHub Actions workflow at
[.github/workflows/benchmark-manual.yml](../.github/workflows/benchmark-manual.yml)
builds on Ubuntu and Windows, exports JSON, and uploads results as artifacts.

---

## Scalar gradient suite

Compares three ways to compute a full gradient for the same scalar function:

- **Symbolic** — `Equation(...).eval_derivatives()`
- **Forward** — `Dual<T>` with one seeded pass per input variable
- **Reverse** — `reverse_mode_gradient(expr)`

Functions benchmarked:

| Name | Expression |
|------|-----------|
| `F1` | `exp(x)*sin(x) + x³ + 2x` |
| `F2` | `xy + sin(x) + y² + exp(x+y)` |
| `F3` | `exp(xy) + x·sin(z) + yz + x²z` |
| `F4` | `(x+y)(z-w) + exp(xz) + sin(yw) + xyzw` |

### Windows / MSVC snapshot

| Function | Symbolic | Forward | Reverse |
|----------|--------:|--------:|--------:|
| `F1`     | 12.6 ns | 10.3 ns |  9.63 ns |
| `F2`     | 12.6 ns | 14.6 ns |  6.00 ns |
| `F3`     | 22.0 ns | 24.0 ns |  7.50 ns |
| `F4`     | 23.0 ns | 39.2 ns |  5.44 ns |

### Linux / GCC snapshot

| Function | Symbolic | Forward | Reverse |
|----------|--------:|--------:|--------:|
| `F1`     | 0.458 ns | 0.346 ns | 0.224 ns |
| `F2`     | 0.223 ns | 0.224 ns | 0.225 ns |
| `F3`     | 0.227 ns | 0.227 ns | 0.224 ns |
| `F4`     | 0.447 ns | 0.451 ns | 0.450 ns |

On Linux, all three modes are fully inlined to near-zero measured time — the benchmark loop overhead dominates. The Windows numbers reflect a less aggressive inliner and show the real relative cost.

---

## Vector Jacobian suite

Compares three Jacobian paths for a 2-output, 4-input function:

- **Symbolic** — `VectorEquation::eval_jacobian()`
- **Forward** — `VectorEquation::eval_jacobian_forward(...)`
- **Reverse** — `VectorEquation::eval_jacobian_reverse()`

```sh
./build/benchmarks --benchmark_filter='.*Vector.*'
```

### Windows snapshot

| Method | Time |
|--------|-----:|
| Symbolic | 27.9 ns |
| Forward  | 140 ns  |
| Reverse  | 6.25 ns |

### Linux snapshot

| Method | Time |
|--------|-----:|
| Symbolic | 0.894 ns |
| Forward  | 0.898 ns |
| Reverse  | ~0.9 ns  |

Same inlining caveat as above.

---

## Parallel Jacobian — `std::async` experiment (reverted)

`eval_jacobian_reverse` was briefly changed to launch one `std::async` task per output row, since the per-row `backward()` passes write to independent `J[i]` slices with no shared state. The sweep below measured the breakeven (same expression replicated across rows):

```sh
./build/benchmarks --benchmark_filter='.*(Parallel).*'
```

| output_dim | Symbolic (serial) | Async parallel |
|:----------:|------------------:|---------------:|
| 2          | 0.900 ns          | 44,372 ns      |
| 4          | 1.79 ns           | 81,571 ns      |
| 6          | 325 ns            | 153,560 ns     |
| 5 (trig/exp mix) | 225 ns      | 125,471 ns     |

`std::async` thread spawn costs ~40–50 µs. Each `backward()` pass costs nanoseconds. The parallel version is 50,000× slower at 2 rows and still ~470× slower at 6 rows. The change was reverted. `eval_jacobian_reverse` remains `constexpr` and serial.

The benchmarks for the parallel sweep (`BM_Reverse_Parallel_*`, `BM_Symbolic_Parallel_*`) are kept in the suite as a regression guard and to document this boundary.

---

## Parallel Jacobian — barrier-based (current, `DiffMode::ParallelReverse`)

`Equation` now optionally pre-spawns `output_dim - 1` worker threads at construction time. Each call to `jacobian<DiffMode::ParallelReverse>()` synchronises with workers using two `std::barrier` phase transitions; no dynamic allocation occurs in the hot path. Workers are stopped via `std::jthread` stop tokens on destruction.

```sh
./build/benchmarks --benchmark_filter='.*(Parallel).*'
```

### Linux / GCC snapshot (16-core, `-O3`)

| Benchmark | Reverse serial | Parallel (barrier) | `std::async` (old) |
|---|---:|---:|---:|
| `BM_Parallel_*_2Rows` | 21.9 ns | 1,299 ns | 44,372 ns |
| `BM_Parallel_*_4Rows` | 50.9 ns | 1,896 ns | 81,571 ns |
| `BM_Parallel_*_6Rows` | 99.1 ns | 1,891 ns | 153,560 ns |
| `BM_Parallel_Future_Large` | 48.5 ns | 1,687 ns | 125,471 ns |

The barrier approach eliminates per-call thread spawn cost (~40–50 µs → ~1.5 µs), an improvement of ~30–80× over `std::async`. However, the per-row `backward()` pass on these expressions costs only 10–50 ns — far below the barrier rendezvous overhead (~1.5 µs). Serial reverse mode remains the right choice for expressions of this size.

The crossover point where parallel wins requires each row's `backward()` to cost more than the barrier overhead (~750 ns/thread). That corresponds to expressions with hundreds of nodes and deep transcendental chains.

`BM_Reverse_Parallel_*` and `BM_Symbolic_Parallel_*` benchmarks are retained as regression guards for the serial path.

---

---

## Comparison vs autodiff v1.1.2

`benchmarks/benchmark_compare.cpp` compares this library against [autodiff](https://autodiff.github.io/) v1.1.2 across a set of tutorial-style expressions.

```sh
cmake -S . -B build_compare -DCMAKE_BUILD_TYPE=Release
cmake --build build_compare --target benchmarks_compare
./build_compare/benchmarks_compare
```

Functions:

| Name | Expression | Variables |
|---|---|---|
| `T1` | `sin(x)` | 1 |
| `TMulti3` | `sin(x) + cos(y) + exp(z)` | 3 |
| `TGrad2` | `log(x·y) + sin(x/y)` | 2 |
| `T4th` | `sin(x)` (4th derivative) | 1 |
| `THess` | `x·y + y²` (Hessian) | 2 |
| `TDir` | `sin(x) + cos(y) + x·y` (directional) | 2 |

Plus `F1`/`F2`/`F4` from the scalar gradient suite above.

### Linux / GCC snapshot (16-core, `-O3`)

**Forward mode:**

| Benchmark | Ours | autodiff | Notes |
|---|---:|---:|---|
| F1, F2, F4 | 14–33 ns | 14–29 ns | ~Tie |
| T1 | 2.68 ns | 1.05 ns | autodiff 2.5× |
| TMulti3 | 25.7 ns | 19.6 ns | autodiff 1.3× |
| TGrad2 | 44.7 ns | 28.5 ns | autodiff 1.6× |
| T4th (nested dual) | 31.9 ns | 7.70 ns | autodiff 4× |
| **T4th (TaylorDual)** | **14.9 ns** | 7.70 ns | autodiff 2× |
| THess | **3.24 ns** | 14.5 ns | Ours 4.5× |
| TDir | **4.65 ns** | 28.0 ns | Ours 6× |

**Reverse mode:**

| Benchmark | Ours | autodiff | Speedup |
|---|---:|---:|---:|
| F1 | 8.93 ns | 202 ns | 23× |
| F2 | 4.45 ns | 191 ns | 43× |
| F4 | 7.85 ns | 327 ns | 42× |
| T1 | 5.23 ns | 194 ns | 37× |
| TMulti3 | 5.43 ns | 418 ns | 77× |
| TDir | 9.12 ns | 89.6 ns | 10× |

Reverse mode dominates in every case. autodiff's `var` type uses a heap-allocated dynamic computation graph; this library's reverse pass is a single stack-based tree traversal with no allocation.

Forward mode is competitive. The gap on simple expressions (T1, TGrad2) comes from autodiff's leaner per-operation dual arithmetic for low-depth cases. The library wins on higher-order expressions (THess, TDir) where the static expression tree allows the compiler to specialise and inline more aggressively.

### `TaylorDual` vs nested `nth_dual_t` for higher-order derivatives

For the 4th derivative of `sin(x)`:

| Method | Time | Notes |
|---|---:|---|
| `nth_dual_t<double, 4>` (nested) | 31.9 ns | 2^4 = 16 doubles per value |
| `TaylorDual<double, 4>` (flat) | 14.9 ns | 5 coefficients, O(N²) multiply |
| autodiff `dual4th` | 7.70 ns | autodiff's internal flat representation |

`TaylorDual` halves the cost over nested duals. For higher orders (N ≥ 6) the 2^N
blowup of nested duals makes `TaylorDual` the only practical choice.

---

## Object footprint and batched throughput

```sh
./build/benchmarks --benchmark_filter='.*(Footprint|Batched).*'
```

### Object sizes (Linux)

| Object | Size |
|--------|-----:|
| Scalar expression (F4) | 136 B |
| `Equation` (F4, with cached partials) | 1,544 B |
| Forward-mode expression (`Dual<double>`) | 232 B |
| `Dual<double>` value | 16 B |

### Batched throughput — F4 (Linux)

| Mode | 256 items | 1024 items | 4096 items |
|------|----------:|-----------:|-----------:|
| Symbolic (`Equation`) | 14.4 M/s | 12.3 M/s | 11.8 M/s |
| Reverse  (expression) | 130.6 M/s | 125.3 M/s | 105.3 M/s |
| Forward  (dual expr)  | 13.7 M/s  | 12.9 M/s  | 12.5 M/s  |

Reverse mode has the best per-item throughput — smaller objects (136 B vs 1,544 B for `Equation`) and a single tree traversal. Forward mode pays both a larger object footprint and a multi-pass cost (one pass per input variable).

### Hardware counters (Linux only)

```sh
perf stat -e cache-references,cache-misses,cycles,instructions \
  ./build/benchmarks --benchmark_filter='.*(Footprint|Batched).*' --benchmark_min_time=0.1s
```

For L1/LLC breakdown:

```sh
perf stat -e LLC-loads,LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses \
  ./build/benchmarks --benchmark_filter='.*Batched.*' --benchmark_min_time=0.1s
```
