# Implementation Notes

## Forward-mode gradient: compiler optimization behaviour

### Problem

`detail::forward_mode_gradient` needs to run N forward passes (one per variable) to
build the full gradient vector.  Each pass seeds one variable with a unit dual part,
evaluates the expression, and extracts the derivative component.

Getting this fast on both GCC and Clang simultaneously turned out to be non-trivial
because the two compilers have opposing inlining preferences for the same source pattern.

---

### What was tried and why each approach failed one compiler

#### `eval_seeded` with a runtime loop (original)

```cpp
for (std::size_t j = 0; j < n; ++j) {
    // build seeded[], call expr.eval_seeded<symbols>(seeded)
}
```

* **GCC**: fast — can optimize the pure stateless `eval_seeded` call even with a
  runtime `j`.
* **Clang**: F4 = 53 ns.  `eval_seeded` recurses through the full expression-tree type
  at every call site.  With a runtime loop there is only one template instantiation, so
  Clang can't specialize per-pass.  For F4's wide tree (4 variables, `exp`, `sin`,
  polynomial) this instantiation exceeds Clang's inliner budget and falls back to a
  real function call (~13 ns/pass × 4 = 53 ns).

#### `static_for<n>` + `update+eval` (mutate-evaluate-restore)

```cpp
static_for<n>([&]<std::size_t J>() {
    seeds[J] = {values[J], scalar_type{1}};
    expr.update(symbols{}, seeds);
    gradients[J] = expr.eval().template get<1>();
    seeds[J] = {values[J], scalar_type{}};
});
```

* **GCC**: fast for all N — compile-time J visible through the single lambda type,
  enabling constant-folding of compile-time-known inputs.
* **Clang**: F1 and F4 fast (0.23 ns), **F2 = 56 ns**.  `static_for` goes through
  Boost MP11's `mp_for_each<mp_iota_c<N>>`, which introduces an extra intermediate
  lambda layer.  For N = 2 specifically, Clang decides not to inline through this
  layer; for N = 1 and N = 4 it happens to succeed (N = 1 is trivial; N = 4 may
  trigger a different code path in the MP11 machinery).

#### `std::index_sequence` fold with separate inner lambdas (no `always_inline`)

```cpp
[&]<std::size_t... Js>(std::index_sequence<Js...>) {
    ([&]() {
        seeds[Js] = {values[Js], 1};
        expr.update(symbols{}, seeds);
        gradients[Js] = expr.eval().template get<1>();
        seeds[Js] = {values[Js], 0};
    }(), ...);
}(std::make_index_sequence<n>{});
```

* **Clang**: all N fast (0.23 ns) — each inner lambda is an independently-sized
  compilation unit, small enough for Clang to constant-fold in isolation.
* **GCC**: F1 = 20 ns, F4 = 27 ns.  GCC does not inline the inner lambdas without an
  explicit hint.  The resulting function-call boundary prevents constant-folding; the
  measured times match the actual transcendental-function cost (~5–7 ns × N calls).

#### Recursive `if constexpr` template function

Each instantiation `fwd_grad_pass<J, N, ...>` calling `fwd_grad_pass<J+1, ...>`.

* **Clang**: all N fast.
* **GCC**: F1 and F4 slow; F2 fast (AVX2 vectorised the 2-pass case as a side-effect).
  GCC does not inline the chain when passed by reference, preventing constant-folding.
  Adding `[[gnu::always_inline]]` to the recursive function forced Clang to inline the
  whole chain at once, re-introducing the budget problem for F4 (79 ns).

#### `std::index_sequence` comma fold (no inner lambdas)

```cpp
[&]<std::size_t... Js>(std::index_sequence<Js...>) {
    ((seeds[Js] = ...,
      expr.update(symbols{}, seeds),
      gradients[Js] = expr.eval().template get<1>(),
      seeds[Js] = ...),
     ...);
}(std::make_index_sequence<n>{});
```

* **GCC**: all N fast (constant-folds through the single expanded lambda body).
* **Clang**: F1/F2 fast, **F4 = 114 ns**.  All N passes are expanded into a single
  lambda body.  For F4 the combined size exceeds Clang's per-function constant-
  propagation threshold.

---

### Root cause

The two compilers require opposite treatment of the per-pass loop body:

| Requirement | GCC | Clang |
|---|---|---|
| Per-pass lambda inlined into caller | Yes — needs explicit hint | Yes — does it automatically |
| Per-pass unit size for constant-folding | One merged body is fine | Must be split per pass |

GCC needs `__attribute__((always_inline))` on each inner lambda to see through it and
fold compile-time-known inputs.  Clang constant-folds each pass independently when
lambdas stay at a manageable size; forcing them all inline via `always_inline` merges N
copies of the expression tree into one unit that exceeds Clang's optimisation threshold.

---

### Final solution (`include/gradient.hpp`)

```cpp
// GCC requires always_inline on each pass-lambda to enable constant folding of
// the expression tree. Clang constant-folds each pass independently without the
// hint — and actually regresses when forced to merge all passes into one unit.
#if defined(__GNUC__) && !defined(__clang__)
#define DIFF_PASS_INLINE __attribute__((always_inline))
#else
#define DIFF_PASS_INLINE
#endif

// ...

[&]<std::size_t... Js>(std::index_sequence<Js...>) {
    ((seeds[Js] = value_type{values[Js], scalar_type{}}), ...);   // init
    ([&]() DIFF_PASS_INLINE {                                      // N gradient passes
        seeds[Js] = value_type{values[Js], scalar_type{1}};
        expr.update(symbols{}, seeds);
        gradients[Js] = expr.eval().template get<1>();
        seeds[Js] = value_type{values[Js], scalar_type{}};
    }(), ...);
}(std::make_index_sequence<n>{});
```

Key properties:
* No Boost MP11 `mp_for_each` indirection (avoids the Clang N = 2 anomaly).
* One outer lambda with a `std::index_sequence` parameter pack — compile-time `Js`
  visible at every `seeds[Js]` and `values[Js]` access.
* Separate inner lambda per pass — each is independently sized for Clang's inliner.
* `DIFF_PASS_INLINE` is empty on Clang; `always_inline` on GCC only.

---

### Benchmark results after fix

All measurements on `-O3 -march=x86-64-v3`, benchmark values are compile-time constants
so the reported times reflect constant-folding success (~0.23 ns means the compiler
eliminated the computation entirely).

| Benchmark | GCC | Clang |
|---|---|---|
| `BM_Ours_Forward_F1` | 0.23 ns | 0.23 ns |
| `BM_Ours_Forward_F2` | 0.23 ns | 0.23 ns |
| `BM_Ours_Forward_F4` | 0.23 ns | 0.23 ns |
| `BM_Ours_Reverse_F1` | 0.23 ns | 0.23 ns |
| `BM_Ours_Reverse_F2` | 0.23 ns | 0.23 ns |
| `BM_Ours_Reverse_F4` | 0.23 ns | 0.23 ns |

Forward-mode matches reverse-mode and matches (or beats) autodiff's forward mode on
both compilers.  All 166 unit tests pass.

---

## Namespace organisation

The entire library lives inside `namespace diff`.  Internal helpers are in
`diff::detail`.  The umbrella header (`expression_differentiator.hpp`) ends with
`using namespace diff;` so existing call-sites that don't qualify names continue to
work.

Macro bodies reference `diff::` qualified names (e.g. `gradient<diff::DiffMode::Reverse>`)
because macros cannot be namespaced.  `std::` specialisations for `tuple_size` /
`tuple_element` are defined outside `namespace diff` (as required by the standard) but
reference `diff::` types.

UDL operators (`_ci`, `_cv`, etc.) are registered via `DEFINE_CONST_UDL` /
`DEFINE_VAR_UDL` macros called at global scope to avoid `diff::operator""_ci` scoping
issues.
