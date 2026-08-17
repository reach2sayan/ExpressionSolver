#pragma once

#include "dual.hpp"
#include "expressions.hpp"
#include "md/tensor.hpp"
#include "named_value.hpp"
#include "scope_guard.hpp"
#include "taylor_dual.hpp"
#include "traits.hpp"
#include "vector_dual.hpp"
#include "mpl.hpp"
#include <algorithm>
#include <array>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>

namespace diff {

namespace mp = diff::mpl;

template <CExpression Expr>
using node_cache_t = std::array<typename std::remove_cvref_t<Expr>::value_type,
                                node_count_v<std::remove_cvref_t<Expr>>>;

// Leaves hold no value, so the point is threaded through the sweep: a leaf
// reads its slot out of `vals` (in canonical symbol order) instead of out of
// itself.
template <std::size_t Base = 0, CSymbolList Syms, CExpression E,
          CNumericBuffer Vals, CNumericBuffer Cache>
constexpr auto fill_cache(const E &node, const Vals &vals,
                          Cache &cache) noexcept {
  using U = std::remove_cvref_t<E>;
  using VT = typename U::value_type;
  if constexpr (CExpressionNode<U>) {
    using Kids = typename U::children_t;
    // Fill each child subtree into its slot, then combine the child values with
    // this node's operator into this node's slot.
    VT v = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return typename U::op_type::func_type{}(
          fill_cache<child_base_at<Base, Kids, I>(), Syms>(
              std::get<I>(node.expressions()), vals, cache)...);
    }(std::make_index_sequence<std::tuple_size_v<Kids>>{});
    cache[Base] = v;
    return v;
  } else { // leaf (Variable / Constant / Lit)
    VT v = node.template eval_seeded<Syms>(vals);
    cache[Base] = v;
    return v;
  }
}

// ===========================================================================
// Order-safe seeding for the symbolic value-array APIs.
//
// derivative_tensor(expr, values) and the reverse-mode hessian(expr, values)
// take a positional std::array whose slot i must hold the value of the i-th
// symbol in CANONICAL order — which is alphabetical by symbol name (see
// extract_symbols_from_expr_t / symbol_less in traits.hpp), NOT source order.
// Passing them in the wrong order silently computes a derivative at the wrong
// point.  make_values() removes that footgun: values bind by symbol *name*, so
// a missing, extra, duplicated, or misspelled symbol is a compile error and
// position is irrelevant.  symbol_order() exposes the canonical order for
// introspection / building arrays by hand.
// ===========================================================================

// The canonical (alphabetical-by-name) order of an expression's free symbols.
template <CExpression Expr>
[[nodiscard]] consteval auto symbol_order() noexcept {
  using SymList = extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  constexpr std::size_t N = mp::mp_size(SymList{});
  std::array<std::string_view, N> out{};
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    ((out[I] = mp::mp_at_c<SymList, I>::name), ...);
  }(std::make_index_sequence<N>{});
  return out;
}

// Build the canonical-order value array for `Expr` from named values.  Scalar
// defaults to the type derivative_tensor expects (scalar_base_t); override it
// for the reverse-mode hessian path (e.g. make_values<Expr, dual_scalar_t<T>>).
template <CExpression Expr,
          Numeric Scalar =
              scalar_base_t<typename std::remove_cvref_t<Expr>::value_type>,
          FixedString... Syms, Numeric... Vs>
[[nodiscard]] constexpr std::array<
    Scalar,
    mp::mp_size(extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{})>
make_values(NamedValue<Syms, Vs>... nv) noexcept {
  using SymList = extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  constexpr std::size_t N = mp::mp_size(SymList{});
  static_assert(sizeof...(Syms) == N,
                "make_values: supply exactly one value per symbol");
  static_assert(
      mp::mp_size(mp::mp_unique<mp::mp_list<symbol_type<Syms>...>>{}) ==
          sizeof...(Syms),
      "make_values: duplicate symbol");
  std::array<Scalar, N> out{};
  (
      [&]<FixedString Sy, Numeric Vv>(const NamedValue<Sy, Vv> &v) {
        constexpr std::size_t idx = find_index_of_symbol<Sy, SymList>();
        static_assert(idx < N, "make_values: symbol not present in expression");
        out[idx] = static_cast<Scalar>(v.value);
      }(nv),
      ...);
  return out;
}

// nd_array_t and nd_index used to live here: a recursively nested std::array
// plus a recursive operator[] walk over a span of indices.  Both are now
// md_tensor (include/md/tensor.hpp), which puts the rank in an md::extents
// instead of in the type nesting and gets the index walk from a layout
// mapping.  nd_tensor_t<S, N, Order> is the direct replacement, and the
// t[i][j][k] spelling those callers used still works.

enum class DiffMode { Symbolic, Forward, Reverse };

// Arity at which derivative_tensor<2> switches from forward-over-forward to
// forward-over-reverse.
//
// The two compute the same tensor but scale differently: forward evaluates
// once per distinct index pair, O(N^2) of them; reverse runs N sweeps, one per
// column.  Measured on a chain energy here, forward wins at N=3 (1.36x) and
// N=4 (1.15x), reverse wins from N=6 (1.22x) and widens at N=8 (1.27x) — so
// the crossover sits at 5.
//
// It is a knob because that number is a property of the *expression*, not of
// the library: a cheap-to-evaluate graph favours forward further out, a dense
// one favours reverse sooner.  Raise it past your arity to pin forward, set it
// to 1 to always take reverse where it is available.  Same spelling as
// DIFF_VFORWARD_CAPACITY in vector_dual.hpp.
#ifndef DIFF_HESSIAN_REVERSE_MIN_N
#define DIFF_HESSIAN_REVERSE_MIN_N 5
#endif
inline constexpr std::size_t kHessianReverseMinN = DIFF_HESSIAN_REVERSE_MIN_N;
static_assert(kHessianReverseMinN > 0,
              "DIFF_HESSIAN_REVERSE_MIN_N must be positive; use a value larger "
              "than your arity to pin the forward algorithm");

namespace detail {

template <CExpression Expr, Numeric T = typename Expr::value_type,
          std::size_t N = mp::mp_size(
              extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{})>
  requires(!DualLike<T>)
[[nodiscard]] constexpr auto
reverse_mode_gradient(const Expr &expr,
                      const std::array<T, N> &vals) noexcept {
  using Syms = extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  std::array<T, N> grads{};
  node_cache_t<Expr> cache{};
  fill_cache<0, Syms>(expr, vals, cache);
  expr.backward(Syms{}, T{1}, grads, cache);
  return grads;
}

template <CExpression Expr, Numeric T = typename Expr::value_type,
          std::size_t N = mp::mp_size(
              extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{})>
  requires DualLike<T>
[[nodiscard]] constexpr auto
reverse_mode_gradient(const Expr &expr,
                      const std::array<T, N> &vals) noexcept {
  using scalar_t = dual_scalar_t<T>;
  using Syms = extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  std::array<T, N> grads{};
  node_cache_t<Expr> cache{};
  fill_cache<0, Syms>(expr, vals, cache);
  expr.backward(Syms{}, T{1}, grads, cache);
  std::array<scalar_t, N> result{};
  std::ranges::transform(grads, result.begin(),
                         [](const T &g) { return g.template get<0>(); });
  return result;
}

template <CExpression Expr,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type,
          Numeric S = dual_scalar_t<T>,
          std::size_t N = mp::mp_size(
              extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{})>
  requires DualLike<T>
[[nodiscard]] constexpr auto
reverse_mode_hessian(const Expr &expr, std::array<S, N> values) noexcept {
  using symbols = extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  nd_tensor_t<S, N, 2> H{};

  // The value level is the point and is the same for every column; only the
  // tangent moves, so build the seeds once and toggle one derivative per sweep.
  std::array<T, N> seeds{};
  std::ranges::transform(values, seeds.begin(),
                         [](const S &v) { return T{v, S{}}; });

  for (std::size_t j = 0; j < N; j++) {
    // Seed column j, then one reverse sweep gives that column of the Hessian.
    // The guard scopes the tangent to this iteration; it is constexpr, so the
    // whole sweep still runs during constant evaluation.
    const auto seed = scoped_seed<1>(seeds[j].deriv());
    std::array<T, N> grads{};
    node_cache_t<Expr> cache{};
    fill_cache<0, symbols>(expr, seeds, cache);
    expr.backward(symbols{}, T{1}, grads, cache);

    const auto column = grads | std::views::transform([](const T &g) {
                          return g.template get<1>();
                        });
    // The tensor is not a range of rows the way the nested array was, so the
    // row index is zipped in rather than walked implicitly.
    for (auto &&[i, entry] : std::views::zip(std::views::iota(0uz, N), column)) {
      H[i, j] = entry;
    }
  }
  return H;
}

template <Numeric S, std::size_t Depth>
constexpr nth_dual_t<S, Depth> make_mixed_seed(S value,
                                               std::span<const std::size_t> idx,
                                               std::size_t k) noexcept {
  if constexpr (Depth == 0) {
    return value;
  } else if constexpr (Depth == 1) {
    return nth_dual_t<S, 1>{value, k == idx[0] ? S{1} : S{}};
  } else {
    auto inner =
        make_mixed_seed<S, Depth - 1>(std::move(value), idx.subspan(1), k);
    auto outer_tangent = embed_constant<S, Depth - 1>(k == idx[0] ? S{1} : S{});
    return nth_dual_t<S, Depth>{std::move(inner), std::move(outer_tangent)};
  }
}

// Every multi-index of a rank-Order tensor with uniform extent N, in
// layout_right order: the cartesian product of Order copies of [0, N).  The
// index_sequence element is discarded — it exists only to give the pack
// something to expand over.
template <std::size_t N, std::size_t Order>
[[nodiscard]] constexpr auto index_grid() noexcept {
  return []<std::size_t... D>(std::index_sequence<D...>) {
    return std::views::cartesian_product(
        ((void)D, std::views::iota(std::size_t{0}, N))...);
  }(std::make_index_sequence<Order>{});
}

// The multi-indices a symmetric rank-Order tensor actually has to evaluate:
// the non-decreasing ones, C(N + Order - 1, Order) of them rather than
// N^Order.  Mixed partials commute (Clairaut/Schwarz), so each of those is the
// canonical representative of a permutation class whose members all name the
// same packed cell.
//
// Enumerated directly rather than as `index_grid | views::filter(is_sorted)`.
// That spelling was shorter and wrong on cost: filtering still *walks* all
// N^Order tuples and throws most away, so it saved the evaluations but not the
// traversal — measured 4x SLOWER than the dense loop on a cheap expression at
// N=2/Order=2, where one evaluation is saved and four tuples are still visited.
// Advancing to the next non-decreasing tuple directly makes the traversal
// O(C(N + Order - 1, Order)) too.
// It is a consteval *table* rather than a lazy iterator, and that is the whole
// point.  An iterator that stops on a data-dependent flag hides the trip count
// from the optimiser, which costs far more than the traversal it saves: the
// generic loop below then stops unrolling, and a derivative tensor that used to
// fold to constants (the Hessian of x^2 + xy + y^2 does not depend on the
// point) stops folding.  Measured 4x slower that way.  A table has a
// compile-time size, so the loop over it unrolls exactly as the old flat
// counter loop did, and the traversal is C(N + Order - 1, Order) rather than
// N^Order.
template <std::size_t N, std::size_t Order>
  requires(N > 0 && Order > 0)
consteval auto simplex_index_table() noexcept {
  // binomial() lives in md/layouts.hpp, same namespace — the layout's own
  // cell count, so the table and the storage cannot disagree on the size.
  std::array<std::array<std::size_t, Order>, binomial(N + Order - 1, Order)>
      out{};
  std::array<std::size_t, Order> idx{};
  for (auto &slot : out) {
    slot = idx;
    // Advance to the next non-decreasing tuple: raise the rightmost position
    // that can still be raised, and collapse everything right of it onto the
    // new value, which is what keeps the tuple sorted.
    for (std::size_t p = Order; p-- > 0;) {
      if (idx[p] + 1 < N) {
        const std::size_t v = ++idx[p];
        std::ranges::fill(idx | std::views::drop(p + 1), v);
        break;
      }
    }
  }
  return out;
}

// Instantiated once per (N, Order), so the walk above runs once however many
// callers ask for the table.
template <std::size_t N, std::size_t Order>
inline constexpr auto simplex_index_table_v = simplex_index_table<N, Order>();

template <std::size_t N, std::size_t Order>
[[nodiscard]] constexpr const auto &symmetric_index_grid() noexcept {
  return simplex_index_table_v<N, Order>;
}

template <std::size_t N, Numeric T>
constexpr auto extract_nth(const T &x) noexcept {
  if constexpr (N == 0) {
    return x;
  } else {
    return extract_nth<N - 1>(x.template get<1>());
  }
}

template <std::size_t Order, CExpression Expr,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type,
          Numeric S = scalar_base_t<T>,
          std::size_t N = mp::mp_size(
              extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{})>
  requires(Order > 0 && N > 0)
[[nodiscard]] constexpr auto
derivative_tensor_impl(const Expr &expr, std::array<S, N> values) noexcept {
  using symbols = extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  using U = nth_dual_t<S, Order>;

  // Second-order, wide enough to pay for it: hand off to forward-over-reverse.
  //
  // The two algorithms compute the same tensor and return the same type, but
  // they scale differently — this path evaluates once per distinct index pair,
  // O(N^2) of them, while reverse_mode_hessian runs N reverse sweeps, one per
  // column.  Measured on a chain energy, forward wins at N=3 (1.36x) and N=4
  // (1.15x), reverse wins from N=6 (1.22x) and pulls further ahead at N=8
  // (1.27x); the crossover sits at N=5.  So the caller says *what* they want
  // and this picks *how*, rather than making them know the crossover.
  //
  // The guard is narrow on purpose: reverse_mode_hessian needs a Dual-valued
  // expression, and it takes dual_scalar_t<T> values, which only coincides
  // with this function's scalar_base_t<T> at depth exactly 1.  A plain-double
  // graph has no reverse path available at all and stays forward.
  // hessian<DiffMode::Reverse> remains the explicit override.
  if constexpr (Order == 2 && DualLike<T> && dual_depth_v<T> == 1 &&
                N >= kHessianReverseMinN) {
    return reverse_mode_hessian(expr, values);
  } else {

  nd_tensor_t<S, N, Order> result{};

  // First-order gradient fast path (N >= 3): one vector-forward pass instead of
  // N scalar Dual passes.  Seeding identity tangents into a VectorDual<N>
  // carries every partial as a lane, so the value-level work — transcendentals
  // and shared subexpressions — is computed once rather than recomputed per
  // variable. The scalar passes evaluate e.g. sin(x)/exp(x*y) once *per
  // variable*; this shares them, a win that grows with N (TMulti3 N=3 reaches
  // parity with autodiff, F4 N=4 overtakes it ~2x).  N <= 2 keeps the leaner
  // scalar path: at N=2 the single shared evaluation barely offsets the
  // VectorDual lane overhead (and N==1 is one plain Dual pass).  VectorDual
  // lanes are double, so this is gated to double-scalar expressions.
  if constexpr (Order == 1 && N >= 3 && std::same_as<S, double>) {
    using V = VectorDual<N>;
    std::array<V, N> seeds{};
    std::ranges::transform(values, seeds.begin(),
                           [](double v) { return V{v}; });
    // Identity tangents: variable k owns lane k.
    for (std::size_t k = 0; k < N; ++k) {
      seeds[k].grad[k] = double{1};
    }
    const V r = expr.template eval_seeded_as<V, symbols>(seeds);
    // Rank 1 under layout_right: storage order is index order, so the lane
    // pack copies straight into the tensor's cells.
    std::ranges::copy_n(r.grad.begin(), N, result.data());
    return result;
  }

  // First-order, low arity: one plain Dual pass per variable.  The generic
  // tensor loop below can do this, but only through machinery that is pure
  // overhead at Order == 1 — a runtime index walk over the cartesian grid and
  // make_mixed_seed's recursive span walk, none of it constant enough for the
  // compiler to unroll the N == 2 case.  Here the seeded variable J is a
  // template parameter, so `k == J` folds away and the passes become
  // straight-line code instead of a rolled loop over stack-resident seeds.
  if constexpr (Order == 1) {
    [&]<std::size_t... J>(std::index_sequence<J...>) {
      auto sweep = [&]<std::size_t Seeded>() {
        std::array<U, N> seeds{};
        for (std::size_t k = 0; k < N; ++k) {
          seeds[k] = U{values[k], k == Seeded ? S{1} : S{}};
        }
        result[Seeded] =
            expr.template eval_seeded_as<U, symbols>(seeds).template get<1>();
      };
      (sweep.template operator()<J>(), ...);
    }(std::make_index_sequence<N>{});
    return result;
  }
  
  // The multi-indices worth evaluating.  This used to be a flat counter plus a
  // hand-written `tmp % N; tmp /= N` unranking — layout_right's inverse
  // mapping, open-coded — running over all N^Order of them.
  //
  // Two things changed.  cartesian_product says the unranking directly, and
  // because it is a view the symmetry becomes one filter in front of it: the
  // tensor is symmetric in every pair of axes, so the permutations of a
  // multi-index all name the same packed cell and evaluating more than one of
  // them is recomputation.  That takes the sweep count from N^Order to
  // C(N + Order - 1, Order) — at Order 3 over 6 variables, 56 sweeps instead
  // of 216, each one a full seeded evaluation of the expression.
  // Left as a plain loop over the table.  Unrolling it with the multi-index as
  // a template parameter looks like the obvious next step — the Order == 1
  // path above does exactly that — but it needs the index to be nameable
  // inside the seeding lambda without capture, i.e. `static constexpr`, and
  // that puts it in static storage and stops it folding at all: measured 33 ns
  // against 4.6 ns for this loop on the quadratic THess case.
  for (const auto &idx : detail::symmetric_index_grid<N, Order>()) {
    std::array<U, N> seeds{};
    std::ranges::transform(
        values, std::views::iota(0uz, N), seeds.begin(),
        [&idx](const S &v, std::size_t k) {
          return make_mixed_seed<S, Order>(v, idx, k);
        });
    U val = expr.template eval_seeded_as<U, symbols>(seeds);
    result.at_index(idx) = extract_nth<Order>(val);
  }
  return result;
  } // end of the forward branch
}

} // namespace detail

template <DiffMode Mode, CExpression Expr,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type,
          std::size_t N = mp::mp_size(
              extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{})>
  requires(Mode == DiffMode::Reverse)
[[nodiscard]] constexpr auto gradient(const Expr &expr,
                                      const std::array<T, N> &values) noexcept {
  return detail::reverse_mode_gradient(expr, values);
}

// Order-safe named form: values bind by symbol name (see make_values).
template <DiffMode Mode, CExpression Expr, FixedString... Syms, Numeric... Vs,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type>
  requires(Mode == DiffMode::Reverse && sizeof...(Syms) > 0)
[[nodiscard]] constexpr auto gradient(const Expr &expr,
                                      NamedValue<Syms, Vs>... nv) noexcept {
  return detail::reverse_mode_gradient(expr, make_values<Expr, T>(nv...));
}

// Positional scalars, in canonical symbol order.
template <DiffMode Mode, CExpression Expr, Numeric... Args,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type,
          std::size_t N = mp::mp_size(
              extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{})>
  requires(Mode == DiffMode::Reverse && sizeof...(Args) > 0 &&
           (std::convertible_to<Args, T> && ...))
[[nodiscard]] constexpr auto gradient(const Expr &expr,
                                      const Args &...xs) noexcept {
  static_assert(sizeof...(Args) == N,
                "gradient: supply exactly one value per symbol, in canonical "
                "order (see symbol_order<Expr>())");
  return detail::reverse_mode_gradient(expr, std::array<T, N>{
                                                 static_cast<T>(xs)...});
}

template <DiffMode Mode, CExpression Expr,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type,
          Numeric S = dual_scalar_t<T>,
          std::size_t N = mp::mp_size(
              extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{})>
  requires(Mode == DiffMode::Reverse && DualLike<T>)
[[nodiscard]] constexpr auto hessian(const Expr &expr,
                                     std::array<S, N> values) noexcept {
  return detail::reverse_mode_hessian(expr, values);
}

// Order-safe named form: values bind by symbol name (see make_values).
template <DiffMode Mode, CExpression Expr, FixedString... Syms, Numeric... Vs,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type>
  requires(Mode == DiffMode::Reverse && DualLike<T>)
[[nodiscard]] constexpr auto hessian(const Expr &expr,
                                     NamedValue<Syms, Vs>... nv) noexcept {
  return detail::reverse_mode_hessian(
      expr, make_values<Expr, dual_scalar_t<T>>(nv...));
}


template <std::size_t Order, CExpression Expr,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type,
          Numeric S = scalar_base_t<T>,
          std::size_t N = mp::mp_size(
              extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{})>
  requires(Order > 0 && N > 0)
[[nodiscard]] constexpr auto derivative_tensor(const Expr &expr,
                                               std::array<S, N> values) noexcept {
  return detail::derivative_tensor_impl<Order>(expr, values);
}

// Order-safe named form: values bind by symbol name (see make_values).
template <std::size_t Order, CExpression Expr, FixedString... Syms,
          Numeric... Vs>
  requires(Order > 0 && sizeof...(Syms) > 0)
[[nodiscard]] constexpr auto derivative_tensor(const Expr &expr,
                                               NamedValue<Syms, Vs>... nv) noexcept {
  return detail::derivative_tensor_impl<Order>(expr, make_values<Expr>(nv...));
}


namespace detail {

template <CArithmetic T> consteval T compile_time_factorial(T Order) {
  T result = 1;
  for (T i = 1; i <= Order; ++i) {
    result *= i;
  }
  return result;
}
#if !defined(NDEBUG)
// A macro, not a `const char[]`: static_assert takes a *string literal* until
// C++26's user-generated messages, so naming the message any other way does not
// compile.
#define DIFF_FACTORIAL_BROKEN "compile_time_factorial is broken"
// clang-format off
static_assert(compile_time_factorial(5) == 120, DIFF_FACTORIAL_BROKEN);
static_assert(compile_time_factorial(7) == 5040, DIFF_FACTORIAL_BROKEN);
static_assert(compile_time_factorial(4) == 24, DIFF_FACTORIAL_BROKEN);
static_assert(compile_time_factorial(3) == 6, DIFF_FACTORIAL_BROKEN);
// clang-format on
#undef DIFF_FACTORIAL_BROKEN
#endif

template <std::size_t Order, CExpression Expr,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type,
          Numeric S = scalar_base_t<T>,
          std::size_t NVars = mp::mp_size(
              extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{})>
  requires(Order > 0 && NVars == 1)
[[nodiscard]] constexpr S univariate_derivative_impl(const Expr &expr,
                                                     S x0) noexcept {
  using symbols = extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  using TD = TaylorDual<S, Order>;

  TD seed;
  seed.c[0] = x0;
  seed.c[1] = S{1};

  TD result =
      expr.template eval_seeded_as<TD, symbols>(std::array<TD, 1>{seed});

  constexpr S factorial = compile_time_factorial(Order);
  return result.c[Order] * factorial;
}

} // namespace detail

template <std::size_t Order, CExpression Expr,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type,
          Numeric S = scalar_base_t<T>,
          std::size_t NVars = mp::mp_size(
              extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>{})>
  requires(Order > 0 && NVars == 1)
[[nodiscard]] constexpr S univariate_derivative(const Expr &expr,
                                                S x0) noexcept {
  return detail::univariate_derivative_impl<Order>(expr, x0);
}

} // namespace diff

#define reverse_mode_grad gradient<diff::DiffMode::Reverse>
#define reverse_mode_hess hessian<diff::DiffMode::Reverse>
