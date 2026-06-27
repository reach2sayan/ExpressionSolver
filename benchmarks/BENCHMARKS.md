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

### Linux / GCC snapshot (`-O3`, `-march=x86-64-v3`, 16-core)

| Function | Symbolic | Forward | Reverse |
|----------|--------:|--------:|--------:|
| `F1`     | 36.0 ns | 14.7 ns | 19.5 ns |
| `F2`     | 71.4 ns | 71.6 ns |  7.62 ns |
| `F3`     | 17.3 ns | 19.8 ns |  8.67 ns |
| `F4`     | 47.0 ns | 28.8 ns |  9.00 ns |

Reverse mode wins on multi-variable functions (single backward pass). Symbolic is slower because it evaluates N pre-stored derivative expression trees. Forward mode is competitive for low-arity functions.

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

### Linux snapshot (`-O3`, `-march=x86-64-v3`, 16-core)

| Method | Time |
|--------|-----:|
| Symbolic | 71.9 ns |
| Forward  | 49.8 ns |
| Reverse  | 12.7 ns |

Reverse mode is fastest: one backward pass over 2 output rows. Forward mode evaluates a derivative tensor in N seeded passes. Symbolic evaluates all pre-stored partial derivative trees.

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

### Vector-valued (dense) Hessian vs autodiff

`benchmark_compare.cpp` also benchmarks a full `n × n` Hessian of a multivariate
energy `f: Rⁿ → R`, swept over `n`, three ways:

- **Ours VForward** — `hessian_vforward()`, vector-forward-over-forward: one sweep
  per row via `Dual<VectorDual<N>>`, O(n) sweeps. Lane capacity is bucketed to the
  smallest power of two ≥ n (`DIFF_VFORWARD_CAPACITY`, default 32). **This is also
  what the public `hessian()` now routes to** (it delegates to `hessian_vforward`
  for `m ≤ kVForwardN`, scalar fallback otherwise).
- **Ours Forward** — the scalar O(n²) forward-over-forward `dual2nd` driver, now
  exposed as `detail::hessian_scalar` (the fallback + bit-exact cross-check path).
- **autodiff** — `autodiff::hessian()` with `VectorXdual2nd` + Eigen (also O(n²)).

```sh
cmake -S . -B build_compare -DCMAKE_BUILD_TYPE=Release -DDIFF_BUILD_COMPARE=ON
cmake --build build_compare --target benchmarks_compare
./build_compare/benchmarks_compare --benchmark_filter='Hess/|HessSparse/|HessExpr'
```

Three energy regimes are covered:

| Regime | Energy | Eval cost |
|---|---|---|
| Dense | `Σ yᵢlog yᵢ + Σ_{i<j} c_ij·yᵢyⱼ/(1+yᵢ) + exp(y₀y_{n-1})` | O(n²) |
| Sparse | `Σ yᵢlog yᵢ + Σ c_i·(yᵢ−yᵢ₊₁)² + exp(y₀y_{n-1})` | O(n) |
| Expr-template | same sparse math, built as a compile-time `Variable`/`+`/`*`/`log`/`exp` graph and bridged into the driver via `eval_seeded_as<T,Syms>` | O(n) |

#### Linux / GCC snapshot (`-O3`, `-march=x86-64-v3`, `taskset`-pinned, median of 7)

The box is shared and noisy: *unpinned* run-to-run variance is large (±30%+) and can
flip mid-`n` orderings, so these are pinned to one core. Even so, treat ~10% gaps as ties.

**Dense** (ns) — division-heavy (`yᵢyⱼ/(1+yᵢ)`):

| n | Ours VForward (= `hessian()`) | Ours scalar | autodiff |
|---|---:|---:|---:|
| 4  | 926 | 980 | **668** |
| 8  | **5,262** | 7,308 | 6,232 |
| 16 | 85,548 | 76,240 | **72,899** |
| 32 | 3,065,057 | **979,178** | 956,926 |

**Sparse** (ns) — no divisions:

| n | Ours VForward (= `hessian()`) | Ours scalar | autodiff |
|---|---:|---:|---:|
| 4  | 583 | 556 | **429** |
| 8  | 3,166 | 3,166 | **2,507** |
| 16 | 20,743 | 20,743 | **18,050** |
| 32 | 243,917 | **154,316** | 134,242 |

The scalar `dual2nd` driver runs the sparse Hessian ~1.17× off autodiff at
n=16/32. With seeding reduced to two in-place scalar writes per probe
(optimization 7), the residual is per-operation `dual2nd` arithmetic and the
value-level recompute the scalar driver repeats per probe — vector-forward
shares that value-level work but loses to pack width at n ≥ 16.

**Expression-template graph** (ns; autodiff baseline = sparse at same n):

| n | Ours `hessian()` | VForward (forced) | autodiff |
|---|---:|---:|---:|
| 4 | **723** | 4,095 | 439 |
| 8 | **3,309** | 7,578 | 2,769 |

`hessian(graph, x)` routes a compile-time expression graph (`CExpression`) to the
scalar driver — the caller passes the graph straight in, with no wrapping or
driver choice. Vector-forward is ~2× slower here because each graph node carries a
wide `Dual<VectorDual<N>>` intermediate; it remains reachable via `hessian_vforward`
if forced, but is never the right pick for a graph.

#### Per-evaluation cost (`dual2nd`, one seeded eval, compile-time arity)

Isolating a single forward-over-forward evaluation of the sparse energy with `N=16` as a
compile-time constant (so the energy loops unroll), no Hessian driver loop:

| variant | ns |
|---|---:|
| Ours — struct-node lazy ET | **158** |
| Ours — eager (reference) | 142 |
| autodiff | 161 |
| Ours — earlier closure-based lazy | 288 |

With a compile-time arity the struct-based expression-template `Dual` matches autodiff
(158 vs 161); the earlier closure-based attempt was 2× slower (288). What actually governs
the realistic runtime-`n` driver is the per-*operation* cost of the two dominant ops:

#### Per-operation cost (`dual2nd`, runtime n, varied inputs)

| op | ours | autodiff |
|---|---:|---:|
| multiply | **130** | 187 |
| divide | **183** | 217 |

Both now beat autodiff. Divide is the one that mattered: the textbook quotient form
`(a'b − ab')/b²` did **two** divisions per level (304 ns, 1.4× slower than autodiff); the
reciprocal form `inv = 1/b; {a·inv, (a' − (a·inv)b')·inv}` does **one** (183 ns). The dense
energy is division-heavy, so this is what was losing it.

Takeaways:

- **`hessian()` (vector-forward) wins at small `n` (≈4–8)** — dense n=8 5,262 vs 6,232,
  sparse n=8 1,940 vs 2,640. At `n ≥ 16` the per-lane × per-sweep cost overtakes the
  fewer-sweeps saving and autodiff wins. The vector-forward sweet spot is *small* Hessians.
- **The scalar `dual2nd` path matches autodiff on the division-heavy dense Hessian**
  (n=16: 76µs vs 73µs; n=32: 979µs vs 957µs) — the reciprocal-form `Dual` divide is what
  brings it to parity. On the **sparse** energy (no divisions) the scalar path is ~1.17×
  behind at n=16/32; with seeding reduced to two in-place scalar writes per probe, that
  residual is per-operation `dual2nd` arithmetic, not driver overhead.
- **Expression graphs run on the scalar driver, not vector-forward** (n=4: 723 vs
  4,095 ns). Graph nodes carry wide `Dual<VectorDual<N>>` intermediates whose per-node
  cost scales with pack width and dwarfs the fewer-sweeps saving, so `hessian()` detects
  `CExpression` and bridges the graph to the scalar driver — the caller makes no driver
  choice (optimization 6 below).

#### Optimizations behind these numbers

1. **Capacity bucketing** (`vforward_driver.hpp`): `hessian_vforward` dispatches the
   smallest power-of-two lane bucket ≥ n instead of always using the full 32-wide pack —
   a large win for small dense Hessians (which is where vector-forward beats autodiff).

2. **Scalar↔dual fusion** (`dual.hpp`): mixed `scalar OP dual` operations distribute
   the constant straight through `val`/`deriv` instead of promoting it to a zero-seeded
   `Dual` and running dual×dual arithmetic against the zeros (autodiff gets the same via
   its `NumberDualMulExpr`). Bit-identical for finite operands, so no value test moves.
   The **vector-forward (`VectorDual`) path benefited most** — its scalar-times-pack
   terms were multiplying through an N-wide array of zeros.

3. **`Dual` as a lazy expression-template number** (`dual.hpp`): the arithmetic operators
   and math functions return struct-based `Bin`/`Mono` expression nodes (stateless op-tag
   functors carrying the exact eager formula) that materialize to `Dual<T>` only at the
   consumption boundary — the same shape as autodiff's forward dual, so `Dual` is now
   consistent with the rest of the (expression-template) library. Operands are held by
   reference (lvalues) or moved (rvalues), so nothing is copied and a node may safely be
   returned from an `auto` energy lambda. Results are bit-identical, and per-evaluation it
   matches autodiff with a compile-time arity (table above). A first attempt using closures
   (`reference_wrapper` capture) was 2× slower and was replaced by the struct nodes.
   Laziness is suppressed for wide scalars like `VectorDual<N>`, which evaluate eagerly so
   the SIMD lane loops stay vectorizable.

4. **O(m²) scalar seeding** (`forward_driver.hpp`): `detail::hessian_scalar` seeds the dof
   array once and, per `(i,j)` pair, touches only the (at most) two active dofs and resets
   them — instead of rewriting all `n` dofs every pair. Correct (cross-checked bit-for-bit
   against vector-forward) but a minor win on its own.

5. **Reciprocal-form division** (`dual.hpp`): `dual_div` computes `inv = 1/denominator`
   once and uses multiplies, instead of the textbook `(a'b − ab')/b²` (two divisions per
   nesting level — at `dual2nd` each inner `Dual<double>` divide recurses into more).
   Cut the `dual2nd` divide from 304 → 183 ns (now faster than autodiff's 217), which
   brought the division-heavy dense scalar Hessian to parity with autodiff. `VectorDual`
   already used this form; plain `Dual` now does too. Not bit-identical (it reassociates),
   but within rounding — the `EXPECT_NEAR` and vector-forward cross-check tests cover it.

6. **Automatic driver routing** (`vforward_driver.hpp`, `seeded_energy.hpp`): the public
   `hessian(f, x)` inspects `f` at compile time. A compile-time expression graph
   (`CExpression`) is bridged via `seeded_energy()` and dispatched to the scalar driver —
   its wide `Dual<VectorDual<N>>` graph nodes make vector-forward ~2× slower (table above)
   — while a plain runtime energy lambda routes to vector-forward, the small-`n` winner.
   The `seeded_energy()` bridge packs the driver's seeded dofs in symbol order and
   traverses the graph once via `eval_seeded_as`, carrying a `kSeededExprEnergy` tag, so
   callers write `hessian(graph, x)` and get the right driver with no wrapping.

7. **In-place seed toggle on the scalar Hessian driver** (`forward_driver.hpp`): a
   `dual2nd` is four scalars `[val.val, val.deriv, deriv.val, deriv.deriv]`; per probe
   only the two first-order seeds (`val.deriv` = inner `e_j`, `deriv.val` = outer `e_i`)
   move — `val.val == x[k]` and the second-order seed are loop-invariant.
   `detail::hessian_scalar` toggles just those two scalars in place and hoists the inner
   seed out of the `i`-loop (constant per column), rather than reconstructing the whole
   `dual2nd` on every seed and reset. Bit-identical to the full reconstruction (same
   derivative values seeded), with the per-probe seeding writes kept off the energy
   eval's critical path.

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
