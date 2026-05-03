# Forward and Reverse Mode AD — Jacobian Computation

_Method-by-method walkthrough with worked examples for `Expression_Differentiator`._

---

## Table of Contents

1. [Introduction & Running Example](#1-introduction--running-example)
2. [Expression Tree Representation](#2-expression-tree-representation)
   - 2.1 [Node types](#21-node-types)
   - 2.2 [Tree for the running example](#22-tree-for-the-running-example)
   - 2.3 [Universal methods on every node](#23-universal-methods-on-every-node)
   - 2.4 [`eval()`](#24-eval)
   - 2.5 [`update(symbols, updates)`](#25-updatesymbols-updates)
   - 2.6 [`collect(symbols, out)`](#26-collectsymbols-out)
   - 2.7 [`eval_seeded<Syms>(vals)`](#27-eval_seededsymsvals)
   - 2.8 [`backward(syms, adj, grads)`](#28-backwardsyms-adj-grads)
   - 2.9 [Operation-level backward rules](#29-operation-level-backward-rules)
3. [Dual Numbers — Engine of Forward Mode](#3-dual-numbers--engine-of-forward-mode)
4. [Forward-Mode Jacobian](#4-forward-mode-jacobian)
   - 4.1 [Mathematical basis](#41-mathematical-basis)
   - 4.2 [Code walkthrough](#42-code-walkthrough-detailforward_mode_gradient)
   - 4.3 [Flowchart](#43-flowchart)
   - 4.4 [Worked example](#44-worked-example)
5. [Reverse-Mode Jacobian](#5-reverse-mode-jacobian)
   - 5.1 [Mathematical basis](#51-mathematical-basis)
   - 5.2 [Code walkthrough](#52-code-walkthrough-detailreverse_mode_gradient)
   - 5.3 [Flowchart](#53-flowchart)
   - 5.4 [Worked example](#54-worked-example)
6. [Comparison](#6-comparison-of-forward-and-reverse-mode)
7. [Symbol Resolution](#7-symbol-resolution-compile-time-indexing)
8. [Quick Reference](#8-quick-reference)

---

## 1. Introduction & Running Example

Throughout this document we use

```
f(x, y) = x·y + sin(x)     at  (x, y) = (π/2, 3)
```

The Jacobian (gradient row-vector) is

```
∇f = [ ∂f/∂x,  ∂f/∂y ]
   = [ y + cos(x),  x ]
   |_{x=π/2, y=3}
   = [ 3 + 0,  π/2 ]
   = [ 3,  π/2 ]
```

Every code section below refers to real source lines in `include/`.

---

## 2. Expression Tree Representation

### 2.1 Node types

| Node type | Template signature | Role |
|---|---|---|
| `Constant<T>` | `Numeric T` | Fixed numeric literal |
| `Variable<T, symbol>` | `Numeric T`, `char symbol` | Named mutable scalar |
| `RuntimeVariable<T>` | `Numeric T` | Index-addressed variable |
| `Expression<Op, LHS, RHS>` | binary op, two children | Binary operation node |
| `MonoExpression<Op, Expr>` | unary op, one child | Unary operation node |
| `EvalResult<T>` | `Numeric T` | Thin value wrapper (see §2.7) |

Sources: `expressions.hpp`, `values.hpp`.

The concept `ExpressionConcept` is satisfied by all of the above through the
`is_expression_type<T>` tag trait (`expressions.hpp:38–67`).

### 2.2 Tree for the running example

```
                    Expression<SumOp>
                   /                  \
     Expression<MultiplyOp>      MonoExpression<SineOp>
       /            \                     |
Variable<'x'>   Variable<'y'>       Variable<'x'>

                f(x,y)  =  x·y  +  sin(x)
```

The symbol `'x'` appears in two distinct leaf nodes.  Both have the same type
`Variable<T,'x'>` and are updated in sync whenever `update()` is called on the
root.

### 2.3 Universal methods on every node

Every node (regardless of type) provides exactly these five methods:

```cpp
// 1. Evaluate: returns scalar at current variable state.
auto eval() const -> value_type;

// 2. Walk tree and write new values into every matching Variable.
void update(const auto& symbols, const auto& updates);

// 3. Walk tree and read current Variable values into an array.
void collect(const auto& symbols, auto& out) const;

// 4. Stateless evaluation from an explicit seed array (no mutation).
template<typename Syms, size_t N>
auto eval_seeded(const array<value_type, N>& vals) const -> value_type;

// 5. Reverse-mode adjoint propagation.
void backward(const auto& syms, value_type adj, auto& grads) const;
```

---

### 2.4 `eval()`

**Purpose.** Recursively evaluate the expression tree, returning the scalar
primal value.

**Per-node behaviour:**

| Node | What it does | Source |
|---|---|---|
| `Constant` | Returns stored literal | `values.hpp:176` |
| `Variable<T,C>` | Returns `value` (internal field) | `values.hpp:218` |
| `RuntimeVariable` | Returns `value_` | `values.hpp:311` |
| `Expression<Op,L,R>` | Calls `Op::eval(lhs, rhs)` via `std::apply` | `expressions.hpp:162` |
| `MonoExpression<Op,E>` | Calls `Op::eval(expression)` | `expressions.hpp:107` |

`Op::eval` for binary ops simply invokes the wrapped functor
(e.g. `std::plus<T>{}(lhs, rhs)`).  For unary ops it invokes the
function-object (e.g. `detail::sine_impl<T>{}(expr)`).

**Running example:**
```
root.eval()
  = mul.eval()  +  sine.eval()
  = (π/2 · 3)  +  sin(π/2)
  = 3π/2       +  1
  ≈ 5.712
```

---

### 2.5 `update(symbols, updates)`

**Purpose.** Walk the tree and write new values into every `Variable` node.

`symbols` is a Boost.MP11 compile-time type-list
`mp_list<integral_constant<char,'x'>, integral_constant<char,'y'>, ...>` —
one entry per unique variable.  `updates` is an array of `value_type` aligned
with that list.

**Per-node behaviour:**

| Node | What it does | Source |
|---|---|---|
| `Constant` | No-op | `values.hpp:184` |
| `Variable<T,symbol>` | Resolves `constexpr idx = find_index_of_char<symbol,Syms>()`, then `*this = updates[idx]` | `values.hpp:265–270` |
| `RuntimeVariable` | `value_ = T(updates[index_])` using runtime index | `values.hpp:317–319` |
| `Expression` | Recurses into `lhs` then `rhs` | `expressions.hpp:182–185` |
| `MonoExpression` | Recurses into `expression` | `expressions.hpp:116–118` |

> **Key invariant:** After `expr.update(symbols{}, seeds)`, every `Variable`
> in the tree has been refreshed.  A subsequent `expr.eval()` reflects the new
> values.  This is how forward mode seeds `Dual` numbers into the tree before
> each directional-derivative pass.

---

### 2.6 `collect(symbols, out)`

**Purpose.** The inverse of `update` — read current variable values _out of_
the tree into an array.

Used by the no-argument `hessian()` overloads to snapshot the evaluation point
before reseeding.

| Node | What it does | Source |
|---|---|---|
| `Constant` | No-op | `values.hpp:185` |
| `Variable<T,symbol>` | `out[idx] = value` | `values.hpp:272–278` |
| `RuntimeVariable` | No-op (index-based, not symbol-based) | `values.hpp:320` |
| `Expression` / `MonoExpression` | Recurses into children | `expressions.hpp` |

---

### 2.7 `eval_seeded<Syms>(vals)`

**Purpose.** A *stateless* evaluation that bypasses the mutable `Variable`
state entirely.  Instead of reading internal `value` fields, it indexes
directly into the caller-supplied array `vals`.

The template parameter `Syms` is the compile-time symbol list; it tells each
`Variable` node which slot in `vals` belongs to it.

**Per-node behaviour:**

| Node | What it returns | Source |
|---|---|---|
| `Constant` | The stored literal (ignores `vals`) | `values.hpp:189` |
| `Variable<T,symbol>` | `vals[find_index_of_char<symbol,Syms>()]` | `values.hpp:232–235` |
| `RuntimeVariable` | `value_` (ignores `vals`) | `values.hpp:327` |
| `Expression<Op,L,R>` | `Op::eval(EvalResult{lhs.eval_seeded(vals)}, EvalResult{rhs.eval_seeded(vals)})` | `expressions.hpp:172–180` |
| `MonoExpression<Op,E>` | `Op::eval(EvalResult{expression.eval_seeded(vals)})` | `expressions.hpp:109–114` |

#### Why `EvalResult`?

`Op::eval` expects arguments satisfying `ExpressionConcept`, but the result of
a recursive `eval_seeded` call is a plain scalar `T`.  `EvalResult<T>` is a
zero-overhead wrapper that satisfies `ExpressionConcept`:

```cpp
// expressions.hpp:59–64
template <Numeric T> struct EvalResult {
  T value;
  constexpr T eval() const { return value; }
  constexpr operator T() const { return value; }
};
```

By wrapping each child's result in `EvalResult` before passing it to
`Op::eval`, `eval_seeded` reuses exactly the same operation code paths as
`eval()`, with no duplication.  The wrapper is eliminated by the compiler.

`eval_seeded` is essential for the **forward-mode Hessian**, where the
expression is `const` (no mutation allowed) and must be evaluated at many
different `Dual<Dual<T>>` seed arrays.

---

### 2.8 `backward(syms, adj, grads)`

**Purpose.** Reverse-mode adjoint propagation.  Given that the output of this
node contributes to `f` scaled by `adj` (the upstream adjoint), accumulate
the partial derivatives into `grads`.

**Parameters:**

- `syms` — compile-time symbol list (same type as in `update`/`collect`).
- `adj` — the adjoint flowing **into** this node from its parent.
  Mathematically: `adj = ∂f/∂(this node's output)`.
- `grads` — output array, one slot per variable.  Contributions are
  **additively accumulated** (`grads[i] += ...`).

**Per-node behaviour:**

#### `Constant`
No variable → no contribution.
```cpp
// values.hpp:186
constexpr void backward(const auto&, T, auto&) const {}
```

#### `Variable<T,symbol>`
This _is_ a leaf variable.  `∂(self)/∂x_self = 1`, so:
```cpp
// values.hpp:286–292
constexpr void backward(const auto& syms, T adj, auto& grads) const {
    constexpr auto idx = find_index_of_char<symbol, Syms>();
    grads[idx] += adj;
}
```

#### `RuntimeVariable`
Same as above with a runtime index:
```cpp
// values.hpp:322–324
grads[index_] += adj;
```

#### `Expression<Op, L, R>`
Delegates to `Op::backward(lhs, rhs, adj, syms, grads)`:
```cpp
// expressions.hpp:190–193
std::apply([&](const auto&... e) { Op::backward(e..., adj, syms, grads); },
           inner_expressions);
```
Each `Op` encodes the chain rule for its specific operation (see §2.9).

#### `MonoExpression<Op, E>`
Delegates to `Op::backward(expression, adj, syms, grads)`:
```cpp
// expressions.hpp:122–124
Op::backward(expression, adj, syms, grads);
```

---

### 2.9 Operation-level backward rules

Each `Op` struct's `backward` static method encodes the chain rule for one
specific mathematical operation.  Below is the complete list.

#### Binary operations

**`SumOp` — `f = u + v`**
```
∂f/∂u = 1,  ∂f/∂v = 1
```
```cpp
// operations.hpp:62–67
lhs.backward(syms, adj, grads);      // adj unchanged
rhs.backward(syms, adj, grads);
```
Both children receive the full adjoint.

---

**`MultiplyOp` — `f = u · v`**
```
∂f/∂u = v,  ∂f/∂v = u
```
```cpp
// operations.hpp:78–83
lhs.backward(syms, adj * static_cast<T>(rhs), grads);   // adj · v
rhs.backward(syms, adj * static_cast<T>(lhs), grads);   // adj · u
```
Each child is scaled by the _other_ operand's primal value.

---

**`DivideOp` — `f = u / v`**
```
∂f/∂u = 1/v,  ∂f/∂v = -u/v²
```
```cpp
// operations.hpp:108–114
const T b = static_cast<T>(rhs);
lhs.backward(syms, adj / b, grads);
rhs.backward(syms, -adj * static_cast<T>(lhs) / (b * b), grads);
```

---

#### Unary operations

**`NegateOp` — `f = -u`**
```cpp
expr.backward(syms, -adj, grads);     // ∂f/∂u = -1
```

---

**`SineOp` — `f = sin(u)`**
```cpp
expr.backward(syms, adj * cos(static_cast<T>(expr)), grads);  // ∂f/∂u = cos(u)
```

---

**`CosineOp` — `f = cos(u)`**
```cpp
expr.backward(syms, -adj * sin(static_cast<T>(expr)), grads); // ∂f/∂u = -sin(u)
```

---

**`ExpOp` — `f = exp(u)`**
```cpp
expr.backward(syms, adj * exp(static_cast<T>(expr)), grads);  // ∂f/∂u = exp(u)
```

---

Complete table of all unary `backward` multipliers:

| Op | `f` | local `∂f/∂u` | multiplier on `adj` |
|---|---|---|---|
| `NegateOp` | `-u` | `-1` | `-adj` |
| `SineOp` | `sin(u)` | `cos(u)` | `adj * cos(u)` |
| `CosineOp` | `cos(u)` | `-sin(u)` | `-adj * sin(u)` |
| `ExpOp` | `exp(u)` | `exp(u)` | `adj * exp(u)` |
| `TanOp` | `tan(u)` | `1/cos²(u)` | `adj / (cos(u)²)` |
| `LogOp` | `log(u)` | `1/u` | `adj / u` |
| `SqrtOp` | `√u` | `1/(2√u)` | `adj / (2·sqrt(u))` |
| `AbsOp` | `|u|` | `sign(u)` | `adj * sign(u)` |
| `SinhOp` | `sinh(u)` | `cosh(u)` | `adj * cosh(u)` |
| `CoshOp` | `cosh(u)` | `sinh(u)` | `adj * sinh(u)` |
| `TanhOp` | `tanh(u)` | `1/cosh²(u)` | `adj / (cosh(u)²)` |
| `AsinOp` | `asin(u)` | `1/√(1-u²)` | `adj / sqrt(1-u²)` |
| `AcosOp` | `acos(u)` | `-1/√(1-u²)` | `-adj / sqrt(1-u²)` |
| `AtanOp` | `atan(u)` | `1/(1+u²)` | `adj / (1+u²)` |

Sources: `operations.hpp:86–343`.

---

## 3. Dual Numbers — Engine of Forward Mode

### Definition

A _dual number_ is a pair `(v, v̇)` where `v` is the primal value and `v̇` is
the tangent (directional derivative).  Arithmetic extends scalar arithmetic by
the nilpotency rule `ε² = 0`:

```
(v, v̇) + (u, u̇) = (v+u,  v̇+u̇)
(v, v̇) · (u, u̇) = (v·u,  v̇·u + v·u̇)       ← product rule
(v, v̇) / (u, u̇) = (v/u,  (v̇·u − v·u̇) / u²) ← quotient rule
```

### Implementation: `Dual<T>` (`dual.hpp`)

```cpp
template <typename T> class Dual {
    T val{};    // primal
    T deriv{};  // tangent

    constexpr Dual operator*(const Dual& o) const {
        // product rule baked in:
        return Dual{val * o.val,  deriv * o.val + val * o.deriv};  // dual.hpp:26–28
    }
};
```

Transcendental functions apply the chain rule automatically:

```cpp
// dual.hpp:68–71
friend constexpr Dual sin(const Dual& d) {
    return Dual{sin(d.val),  cos(d.val) * d.deriv};
}

// dual.hpp:77–79
friend constexpr Dual exp(const Dual& d) {
    const T e = exp(d.val);
    return Dual{e,  e * d.deriv};
}
```

`Dual<T>` satisfies the `Numeric` concept (`dual.hpp:127`), so
`Variable<Dual<double>,'x'>` is a perfectly ordinary expression node whose
scalar value happens to carry a tangent component.

### Component access

`Dual<T>::get<I>()` returns `val` (`I=0`) or `deriv` (`I=1`).

For nested duals `Dual<Dual<double>>` (Hessian computation):
- `get<0>()` → inner `Dual` (primal direction)
- `get<1>()` → outer tangent `Dual`
- `get<0>().get<0>()` → scalar primal
- `get<1>().get<1>()` → second derivative (used to extract Hessian entries)

---

## 4. Forward-Mode Jacobian

### 4.1 Mathematical basis

For `f : ℝⁿ → ℝ`, the `j`-th partial derivative `∂f/∂xⱼ` is the directional
derivative in direction `eⱼ` (the `j`-th standard basis vector).

Seeding the `j`-th variable with tangent 1 and all others with tangent 0:

```
f( (x₁,0), …, (xⱼ,1), …, (xₙ,0) )  =  ( f(x),  ∂f/∂xⱼ )
```

The dual arithmetic propagates this tangent through every operation, delivering
the exact partial derivative at the output — no finite differences needed.

The full Jacobian requires **n separate passes**, one per input dimension.

### 4.2 Code walkthrough: `detail::forward_mode_gradient`

**Location:** `gradient.hpp:58–91`

```cpp
template <ExpressionConcept Expr, ...>
auto forward_mode_gradient(Expr& expr, array<TArr, N> values)
    requires is_dual_v<typename Expr::value_type>
{
    using symbols    = extract_symbols_from_expr<Expr>::type;
    constexpr size_t n = mp::mp_size<symbols>::value;

    array<value_type, n>  seeds{};      // Dual seeds pushed into the tree
    array<scalar_type, n> gradients{};  // output

    [&]<size_t... Js>(index_sequence<Js...>) {
        // Step 1: initialise all seeds to (value, 0)
        ((seeds[Js] = value_type{values[Js], scalar_type{}}), ...);

        // Step 2: one pass per variable j
        ([&]() {
            seeds[Js] = value_type{values[Js], scalar_type{1}};  // tangent = 1
            expr.update(symbols{}, seeds);                         // write into tree
            gradients[Js] = expr.eval().template get<1>();         // extract ∂f/∂xⱼ
            seeds[Js] = value_type{values[Js], scalar_type{}};    // reset tangent
        }(), ...);
    }(make_index_sequence<n>{});

    expr.update(symbols{}, seeds);   // Step 3: restore zero-tangent state
    return gradients;
}
```

**Step by step:**

1. **Initialise seeds.** Every seed is `Dual{values[i], 0}`: primal = point,
   tangent = 0.

2. **One pass per `j` (loop unrolled at compile time):**
   - Set `seeds[j] = Dual{values[j], 1}` — direction `eⱼ`.
   - Call `expr.update(symbols{}, seeds)` — propagates seeds into every
     `Variable` node (§2.5).
   - Call `expr.eval()`.  Because every `Variable` now holds a `Dual`, all
     operations in the tree execute their dual-arithmetic overloads.  The
     returned `Dual` has `.get<1>()` = `∂f/∂xⱼ`.
   - Store `.get<1>()` into `gradients[j]`.
   - Reset `seeds[j]` tangent back to 0.

3. **Restore state.** One final `update` ensures the expression tree is left
   with zero tangents.

**Complexity:** `O(n · |E|)` — one full tree evaluation per variable.

### 4.3 Flowchart

```
┌──────────────────────────────────────┐
│  Input: expr, values[0..n-1]         │
└───────────────────┬──────────────────┘
                    │
                    ▼
┌──────────────────────────────────────┐
│  Init: seeds[i] = Dual{values[i], 0} │
└───────────────────┬──────────────────┘
                    │
                    ▼
              j ← 0
                    │
                    ▼
             ┌─────────┐
             │  j < n? │
             └────┬────┘
         yes │         │ no
             ▼         └──────────────────────────────────┐
┌────────────────────────────┐                            │
│ seeds[j].deriv ← 1         │                            │
└────────────┬───────────────┘                            │
             ▼                                            │
┌────────────────────────────┐                            │
│ expr.update(symbols, seeds) │                           │
└────────────┬───────────────┘                            │
             ▼                                            │
┌────────────────────────────┐                            │
│ result = expr.eval()        │                           │
└────────────┬───────────────┘                            │
             ▼                                            │
┌────────────────────────────┐                            │
│ grads[j] = result.get<1>() │                            │
└────────────┬───────────────┘                            │
             ▼                                            │
┌────────────────────────────┐                            │
│ seeds[j].deriv ← 0         │                            │
└────────────┬───────────────┘                            │
             ▼                                            │
          j ← j+1 ─────────────────────────────► (loop)  │
                                                          │
          ┌───────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ expr.update(symbols, seeds)     │  (final zero-tangent restore)
└──────────────┬──────────────────┘
               ▼
┌──────────────────────────────────┐
│  Return grads = ∇f               │
└──────────────────────────────────┘
```

### 4.4 Worked example

We compute `∇f(π/2, 3)` for `f(x,y) = xy + sin(x)`.

#### Pass `j=0`: compute `∂f/∂x`

Seeds: `x = (π/2, 1)`, `y = (3, 0)`

```
                  SumOp
                 /       \
         MultiplyOp        SineOp
        /          \           \
  x=(π/2, 1)   y=(3, 0)    x=(π/2, 1)
```

**Evaluating bottom-up with dual arithmetic:**

```
MultiplyOp:  (π/2, 1) · (3, 0)
           = ( π/2·3,  1·3 + π/2·0 )   ← product rule
           = ( 3π/2,   3 )

SineOp:     sin( (π/2, 1) )
           = ( sin(π/2),  cos(π/2)·1 )  ← chain rule
           = ( 1,          0 )

SumOp:      (3π/2, 3) + (1, 0)
           = ( 3π/2 + 1,  3 )
```

Result: `(5.712, 3)` → **`∂f/∂x = 3`** ✓  (expected: `y + cos(π/2) = 3 + 0`)

---

#### Pass `j=1`: compute `∂f/∂y`

Seeds: `x = (π/2, 0)`, `y = (3, 1)`

```
MultiplyOp:  (π/2, 0) · (3, 1)
           = ( 3π/2,   0·3 + π/2·1 )
           = ( 3π/2,   π/2 )

SineOp:     sin( (π/2, 0) )
           = ( 1,   cos(π/2)·0 )
           = ( 1,   0 )

SumOp:      (3π/2, π/2) + (1, 0)
           = ( 5.712,  π/2 )
```

Result: `(5.712, π/2)` → **`∂f/∂y = π/2`** ✓  (expected: `x = π/2`)

---

## 5. Reverse-Mode Jacobian

### 5.1 Mathematical basis

Define the **adjoint** of node `v` as `v̄ = ∂f/∂v` — how sensitive the scalar
output `f` is to that node's value.

Starting from the output (`f̄ = 1`), the backward pass applies the chain rule
at every node to distribute adjoints to its inputs.  For a binary operation
`f = ℓ(u, v)`:

```
ū  +=  f̄ · ∂ℓ/∂u
v̄  +=  f̄ · ∂ℓ/∂v
```

The **additive accumulation** (`+=`) is what makes this correct when variables
appear more than once in the expression (e.g. `x` appears in both `x·y` and
`sin(x)`).

### 5.2 Code walkthrough: `detail::reverse_mode_gradient`

**Location:** `gradient.hpp:31–40`

```cpp
template <ExpressionConcept Expr, typename T = typename Expr::value_type>
    requires (!is_dual_v<T>)
constexpr auto reverse_mode_gradient(const Expr& expr) {
    using Syms = extract_symbols_from_expr<Expr>::type;
    constexpr auto N = mp::mp_size<Syms>::value;

    std::array<T, N> grads{};            // zero-initialised output
    expr.backward(Syms{}, T{1}, grads);  // seed root adjoint = 1
    return grads;
}
```

**Step by step:**

1. Allocate `grads[0..n-1] = 0`.
2. Call `expr.backward(Syms{}, 1.0, grads)`.
   - Root adjoint is `1` because `∂f/∂f = 1`.
3. `backward` recursively applies the chain rule through the tree, additively
   accumulating into `grads` whenever a `Variable` leaf is reached.
4. Return `grads`: element `i` holds `∂f/∂xᵢ`.

**Complexity:** `O(|E|)` — a single traversal regardless of `n`.  This is the
core advantage of reverse mode for large `n`.

Note: the expression is taken as `const Expr&` — reverse mode does **not**
mutate the tree.  Variable nodes must already hold the evaluation-point values
before calling this function.

### 5.3 Flowchart

```
┌──────────────────────────────────────────────┐
│  Input: expr  (variables hold current values) │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  grads[0..n-1] = 0                            │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  expr.backward(Syms{}, 1.0, grads)            │
│                                               │
│   At each binary node Expression<Op,L,R>:     │
│     Op::backward(lhs, rhs, adj, syms, grads)  │
│     ├─ scale adj by local ∂/∂lhs, recurse lhs │
│     └─ scale adj by local ∂/∂rhs, recurse rhs │
│                                               │
│   At each unary node MonoExpression<Op,E>:    │
│     Op::backward(expr, adj, syms, grads)      │
│     └─ scale adj by local ∂/∂u, recurse expr  │
│                                               │
│   At each Variable leaf:                      │
│     grads[idx] += adj                         │
│                                               │
│   At each Constant leaf:                      │
│     (no-op)                                   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Return grads = ∇f                            │
└──────────────────────────────────────────────┘
```

### 5.4 Worked example

We compute `∇f(π/2, 3)` for `f(x,y) = xy + sin(x)` in a single backward pass.

#### Forward values (already in the tree)

```
x = π/2,  y = 3
x·y      = 3π/2
sin(x)   = 1
f        = 3π/2 + 1  ≈ 5.712
```

#### Backward pass (adjoint flow, top-down)

```
                  SumOp  ← adj = 1 (root seed)
                 /       \
         MultiplyOp        SineOp
         ← adj=1            ← adj=1
        /          \           \
  Variable 'x'  Variable 'y'  Variable 'x'
  ← adj=3        ← adj=π/2     ← adj=0
```

**SumOp** (`f = u + v`, `∂f/∂u = 1`, `∂f/∂v = 1`):
- Passes `adj=1` unchanged to both children.

**MultiplyOp** (`f = u·v`, `∂f/∂u = v`, `∂f/∂v = u`) with `adj=1`:
- Sends to `lhs` (`x`): `adj · v = 1 · 3 = 3`
- Sends to `rhs` (`y`): `adj · u = 1 · π/2 = π/2`

**SineOp** (`f = sin(u)`, `∂f/∂u = cos(u)`) with `adj=1`:
- Sends to `expr` (`x`): `adj · cos(π/2) = 1 · 0 = 0`

**Variable `'x'`** accumulates from both subtrees:
```cpp
grads[idx_x] += 3;    // from MultiplyOp path
grads[idx_x] += 0;    // from SineOp path
```

**Variable `'y'`** accumulates:
```cpp
grads[idx_y] += π/2;  // from MultiplyOp path
```

#### Result

```
grads[x] = 3 + 0  = 3    = y + cos(x)|_{π/2,3}  ✓
grads[y] = π/2         = x                       ✓
```

The two occurrences of `x` correctly contribute to the same `grads[idx_x]`
slot via `+=` in `Variable::backward`.  This is why reverse mode handles DAG
re-use correctly — not just trees.

---

## 6. Comparison of Forward and Reverse Mode

| Property | Forward mode | Reverse mode |
|---|---|---|
| Algorithm | Dual-number tangent propagation | Adjoint backpropagation |
| Passes required | `n` (one per input) | `1` |
| Tree mutation | Yes (`update` each pass) | No (`const Expr&`) |
| Output per pass | one partial derivative `∂f/∂xⱼ` | full gradient `∇f` |
| Time complexity | `O(n · |E|)` | `O(|E|)` |
| Preferred when | few inputs (`n ≪ m` outputs) | many inputs, scalar output |
| Variable value type | `Dual<T>` | plain `T` |
| How to extract result | `expr.eval().get<1>()` | `grads` array directly |

### Public API

```cpp
// gradient.hpp:196–211

// Reverse mode — expr must hold plain T values
auto g1 = gradient<diff::DiffMode::Reverse>(expr);

// Forward mode — expr must have Dual<T> value_type
auto g2 = gradient<diff::DiffMode::Forward>(expr, {x_val, y_val});

// Convenience macros (gradient.hpp:256–257)
auto g3 = reverse_mode_grad(expr);
auto g4 = forward_mode_grad(expr, {x_val, y_val});
```

---

## 7. Symbol Resolution: Compile-Time Indexing

Every `Variable<T, symbol>` carries its name as a compile-time `char` template
parameter.  The library builds an ordered, deduplicated list of all symbols in
an expression using Boost.MP11 at compile time:

```cpp
// From traits.hpp
using Syms = typename extract_symbols_from_expr<Expr>::type;
// For f(x,y):
//   mp_list< integral_constant<char,'x'>,
//            integral_constant<char,'y'> >

// Resolving a symbol to its array slot — zero runtime cost:
constexpr auto idx_x = find_index_of_char<'x', Syms>(); // == 0
constexpr auto idx_y = find_index_of_char<'y', Syms>(); // == 1
```

This is how `Variable::update`, `Variable::collect`, `Variable::backward`, and
`Variable::eval_seeded` all know which slot in an `array<T, N>` belongs to
them — entirely at compile time, with no runtime hash map, `if`-chain, or
branch.

---

## 8. Quick Reference

### Forward mode — setup checklist

```cpp
// 1. Declare variables with Dual value type
auto x = Variable<Dual<double>, 'x'>{Dual{x_val, 0.0}};
auto y = Variable<Dual<double>, 'y'>{Dual{y_val, 0.0}};
// (or use the PDV macro: PDV(x_val, 'x'))

// 2. Build expression tree
auto f = x * y + sin(x);

// 3. Compute gradient
auto grad = forward_mode_grad(f, {x_val, y_val});
// grad[0] = ∂f/∂x,  grad[1] = ∂f/∂y
```

### Reverse mode — setup checklist

```cpp
// 1. Declare variables with plain type
auto x = Variable<double, 'x'>{x_val};
auto y = Variable<double, 'y'>{y_val};
// (or use the PV macro: PV(x_val, 'x'))

// 2. Build expression tree
auto f = x * y + sin(x);

// 3. Variables already hold x_val, y_val from construction
//    (or call f.update() manually if you change the point)

// 4. Compute gradient
auto grad = reverse_mode_grad(f);
// grad[0] = ∂f/∂x,  grad[1] = ∂f/∂y
```
