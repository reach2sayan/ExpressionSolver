#pragma once

#include "drivers/coupling.hpp" // compile-time Hessian sparsity + colouring
#include "drivers/common.hpp"   // HessianStatic, symmetrize

#include "dual/dual.hpp"
#include "dual/taylor_dual.hpp"
#include "expr/expressions.hpp"
#include "expr/named_value.hpp"
#include "expr/traits.hpp"
#include "md/tensor.hpp"
#include "util/config.hpp"
#include "util/mpl.hpp"
#include "util/scope_guard.hpp"
#include <algorithm>
#include <array>
#include <functional>
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
//
// `Store` says whether this node's value has to be written to its slot at all.
// A slot is only ever read back by a *parent's* adjoint rule -- every cache[]
// in operations.hpp is cache[cb[...]], a child index, so no rule reads its own
// slot -- which makes the answer entirely the parent's `reads_primals`.  A sum
// or a negation reads neither operand, so everything directly beneath one is
// computed, handed upward, and never written down.  The root is likewise never
// stored: its value is the return value, and nothing looks it up.
//
// The array keeps one slot per node either way; what goes away is the store.
// A slot that is written by nobody and read by nobody is dead, which is the
// form the optimiser can actually drop.
template <std::size_t Base = 0, CSymbolList Syms, bool Store = true,
          CExpression E, CNumericBuffer Vals, CNumericBuffer Cache>
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
          fill_cache<child_base_at<Base, Kids, I>(), Syms,
                     U::op_type::reads_primals>(std::get<I>(node.expressions()),
                                                vals, cache)...);
    }(std::make_index_sequence<std::tuple_size_v<Kids>>{});
    if constexpr (Store) {
      cache[Base] = v;
    }
    return v;
  } else { // leaf (Variable / Constant / Lit)
    VT v = node.template eval_seeded<Syms>(vals);
    if constexpr (Store) {
      cache[Base] = v;
    }
    return v;
  }
}

// One reverse sweep: fill the primal cache from `seeds`, then push adjoints
// back through it from a root adjoint of 1.  Returns {root value, gradients}.
//
// This is the whole reverse mode, and every caller of it wants exactly these
// four lines -- the cache, the fill, the backward, the root adjoint of one.
// They differ only in what they do with the answer: peel a dual level, harvest
// one Hessian column, scatter into a sparse buffer, stack a Jacobian row.
// Writing it once is also what fixes the convention (Syms{}, T{1}) in one
// place rather than six.
template <CSymbolList Syms, CExpression Expr, CNumericBuffer Seeds,
          CNumericBuffer Grads>
DIFF_ALWAYS_INLINE constexpr auto
reverse_sweep(const Expr &expr, const Seeds &seeds, Grads &grads) noexcept {
  using T = typename std::remove_cvref_t<Expr>::value_type;
  node_cache_t<Expr> cache{};
  // Store=false: the root's value is what this function returns, and no
  // adjoint rule looks up its slot, so writing it would be a dead store.
  const T root = fill_cache<0, Syms, false>(expr, seeds, cache);
  expr.backward(Syms{}, T{1}, grads, cache);
  return root;
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
  using SymList = detail::expr_symbols_t<std::remove_cvref_t<Expr>>;
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
[[nodiscard]] constexpr std::array<Scalar, detail::expr_arity_v<Expr>>
make_values(NamedValue<Syms, Vs>... nv) noexcept {
  using SymList = detail::expr_symbols_t<std::remove_cvref_t<Expr>>;
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

// The two modes an entry point can be asked for.  Forward mode is reached
// through the drivers (numeric.hpp / hessian.hpp), which take a
// callable rather than a Mode, so it is not a value here.
enum class DiffMode { Symbolic, Reverse };

namespace detail {

// A gradient of a dual-valued expression comes back carrying the seed in its
// dual part; the gradient proper is the real part.  Both gradient modes go
// through here so they cannot disagree about it.
template <Numeric T, std::size_t N>
[[nodiscard]] constexpr auto
strip_seed(const std::array<T, N> &grads) noexcept {
  if constexpr (DualLike<T>) {
    std::array<dual_scalar_t<T>, N> result{};
    std::ranges::transform(grads, result.begin(),
                           [](const T &g) { return g.template get<0>(); });
    return result;
  } else {
    return grads;
  }
}

// Reverse-mode gradient.  A dual-valued expression differentiates the same way
// -- the sweep is identical -- and then hands back only the value level of each
// gradient entry, which is the scalar derivative.
template <CExpression Expr, Numeric T = typename Expr::value_type,
          std::size_t N = detail::expr_arity_v<Expr>>
[[nodiscard]] constexpr auto
reverse_mode_gradient(const Expr &expr, const std::array<T, N> &vals) noexcept {
  using Syms = detail::expr_symbols_t<Expr>;
  std::array<T, N> grads{};
  reverse_sweep<Syms>(expr, vals, grads);
  return detail::strip_seed(grads);
}

template <CExpression Expr,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type,
          Numeric S = dual_scalar_t<T>,
          std::size_t N = detail::expr_arity_v<Expr>>
  requires DualLike<T>
[[nodiscard]] constexpr auto
reverse_mode_hessian(const Expr &expr, std::array<S, N> values) noexcept {
  using symbols = detail::expr_symbols_t<std::remove_cvref_t<Expr>>;
  nd_tensor_t<S, N, 2> H{};

  // The value level is the point and is the same for every column; only the
  // tangent moves, so build the seeds once and toggle one derivative per sweep.
  std::array<T, N> seeds{};
  std::ranges::transform(values, seeds.begin(),
                         [](const S &v) { return T{v, S{}}; });

  for (std::size_t j = 0; j < N; j++) {
    // Seed column j, then one reverse sweep gives that column of the Hessian.
    const auto seed = scoped_seed<1>(seeds[j].deriv());
    std::array<T, N> grads{};
    reverse_sweep<symbols>(expr, seeds, grads);

    const auto column = grads | std::views::transform([](const T &g) {
                          return g.template get<1>();
                        });

    for (auto &&[i, entry] :
         std::views::zip(std::views::iota(0uz, N), column)) {
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

// The non-decreasing multi-indices: C(N + Order - 1, Order) of them rather than
// N^Order, one canonical representative per permutation class (mixed partials
// commute).  A consteval table, not a filtered view and not a lazy iterator --
// both alternatives measured 4x slower, for two different reasons.
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

// The full seed array for one entry of the derivative grid: every variable
// lifted to nth_dual_t, with the tangents that `idx` names switched on.  Used
// by both tensor drivers -- the scalar one and Equation's stacked one -- which
// build it identically and differ only in where the answer lands.
template <Numeric S, std::size_t Order, std::size_t N>
[[nodiscard]] constexpr std::array<nth_dual_t<S, Order>, N>
mixed_seeds(const std::array<S, N> &values,
            std::span<const std::size_t> idx) noexcept {
  std::array<nth_dual_t<S, Order>, N> seeds{};
  std::ranges::transform(values, std::views::iota(0uz, N), seeds.begin(),
                         [&idx](const S &v, std::size_t k) {
                           return make_mixed_seed<S, Order>(v, idx, k);
                         });
  return seeds;
}

template <std::size_t Order, CExpression Expr,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type,
          Numeric S = scalar_base_t<T>,
          std::size_t N = detail::expr_arity_v<Expr>>
  requires(Order > 0 && N > 0)
[[nodiscard]] constexpr auto
derivative_tensor_impl(const Expr &expr, std::array<S, N> values) noexcept {
  using symbols = detail::expr_symbols_t<std::remove_cvref_t<Expr>>;
  using U = nth_dual_t<S, Order>;

  nd_tensor_t<S, N, Order> result{};

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
            expr.template eval_seeded<symbols>(seeds).template get<1>();
      };
      (sweep.template operator()<J>(), ...);
    }(std::make_index_sequence<N>{});
    return result;
  }

  for (const auto &idx : detail::symmetric_index_grid<N, Order>()) {
    const auto seeds = mixed_seeds<S, Order>(values, idx);
    U val = expr.template eval_seeded<symbols>(seeds);
    result.at_index(idx) = extract_nth<Order>(val);
  }
  return result;
}

} // namespace detail

namespace detail {

// Same fold as binomial() in md/layouts.hpp, which is the idiom this file
// should match.  The self-test is a plain static_assert rather than the macro
// dance the hand-rolled loop needed: consteval means it is checked here, once,
// in every build rather than only in debug ones.
template <CArithmetic T> consteval T compile_time_factorial(T Order) {
  return std::ranges::fold_left(std::views::iota(T{1}, T(Order + 1)), T{1},
                                std::multiplies{});
}
static_assert(compile_time_factorial(3) == 6);
static_assert(compile_time_factorial(5) == 120);
static_assert(compile_time_factorial(7) == 5040);

template <std::size_t Order, CExpression Expr,
          Numeric T = typename std::remove_cvref_t<Expr>::value_type,
          Numeric S = scalar_base_t<T>,
          std::size_t NVars = detail::expr_arity_v<Expr>>
  requires(Order > 0 && NVars == 1)
[[nodiscard]] constexpr S univariate_derivative_impl(const Expr &expr,
                                                     S x0) noexcept {
  using symbols = detail::expr_symbols_t<std::remove_cvref_t<Expr>>;
  using TD = TaylorDual<S, Order>;

  TD seed;
  seed.c[0] = x0;
  seed.c[1] = S{1};

  TD result = expr.template eval_seeded<symbols>(std::array<TD, 1>{seed});

  // Computed in size_t (exact) and converted once, rather than folded in S:
  // the cast is explicit so it does not read as an accidental narrowing.
  constexpr S factorial = static_cast<S>(compile_time_factorial(Order));
  return result.c[Order] * factorial;
}

} // namespace detail

// The expression-taking entry points that used to live here -- gradient<Mode>,
// hessian<Mode>, derivative_tensor<Order>, univariate_derivative<Order>, and
// the reverse_mode_grad / reverse_mode_hess macros -- are now members of
// Equation (expr/equation.hpp).  A bare expression converts to a one-output
// Equation implicitly, so `Equation{expr}.gradient(pt)` is the one spelling.
// Everything above stays here as the engine those members call.


// ---------------------------------------------------------------------------
// Forward-over-reverse Hessian of an expression graph.
//
// The only O(N)-sweep Hessian driver in the library: seeds a tangent, sweeps
// the graph backward once per colour, and harvests a whole set of rows per
// sweep.  It lives here, with the other graph sweeps, because what it needs is
// the tree -- `reverse_sweep` above and the compile-time sparsity pattern from
// coupling.hpp.  The router in hessian.hpp decides when it applies.
// ---------------------------------------------------------------------------

namespace detail {

template <CExpression Expr, typename Colors, typename Harvest>
DIFF_ALWAYS_INLINE void color_sweeps(const Expr &expr, std::span<const double> x,
                  const Colors &colors, Harvest &&harvest) {
  using E = std::remove_cvref_t<Expr>;
  using T = typename E::value_type;
  using S = dual_scalar_t<T>;
  using Syms = detail::expr_symbols_t<E>;
  constexpr std::size_t N = mpl::mp_size(Syms{});

  // The value level is the point and is identical on every sweep; only the
  // seeded tangents move between colours.
  std::array<T, N> seeds{};
  std::ranges::transform(x, seeds.begin(),
                         [](double v) { return T{static_cast<S>(v), S{}}; });

  for (std::size_t c = 0; c < colors.count; ++c) {
    for (auto &&[seed, color] : std::views::zip(seeds, colors.color)) {
      seed.deriv() = (color == c) ? S{1} : S{};
    }
    std::array<T, N> grads{};
    const T root = reverse_sweep<Syms>(expr, seeds, grads);
    harvest(c, root, grads);
  }
}

// Forward-over-reverse Hessian for a compile-time expression graph.  Seeding
// column j with a first-order tangent and running ONE backward sweep yields
// that whole Hessian column, so the full N x N matrix costs N sweeps against
// the scalar driver's N(N+1)/2 forward-over-forward probes — an O(N) vs O(N^2)
// difference, and the j == 0 sweep hands back the value and gradient for free.
// Only a graph can take this path: a runtime lambda has no tree to sweep
// backward.  The seed array is sized by the symbol set, so N is compile-time.
template <CExpression Expr>
HessianStatic<mpl::mp_size(detail::expr_symbols_t<std::remove_cvref_t<Expr>>{})>
hessian_expr_reverse(const Expr &expr, std::span<const double> x) {
  using E = std::remove_cvref_t<Expr>;
  using T = typename E::value_type;
  using Syms = detail::expr_symbols_t<E>;
  constexpr std::size_t N = mpl::mp_size(Syms{});

  // Sparsity read off the expression type, then a colouring of it: columns
  // that share no row can be seeded in the SAME sweep, because no row receives
  // a contribution from more than one of them.  A banded Hessian needs a fixed
  // number of colours regardless of n, so this takes the sweep count from O(N)
  // to O(colours) — the chain energy's tridiagonal-plus-corner pattern colours
  // in 5 whether n is 8 or 32.  A dense pattern colours in N and the loop
  // degenerates to one sweep per column, i.e. never worse.
  static constexpr auto kPattern = hessian_pattern<E>();
  static constexpr auto kColors = color_columns<N>(kPattern);
  // Value-initialised, and that is load-bearing: the scatter below writes only
  // structurally non-zero entries and relies on everything outside the pattern
  // already being zero.
  HessianStatic<N> res{};
  auto &res_gradient = std::get<1>(res);
  auto &res_hessian = std::get<2>(res);

  color_sweeps(expr, x, kColors, [&](std::size_t c, const T &root,
                                     const auto &grads) {
    static constexpr auto kScatter = scatter_targets<N>(kPattern, kColors);
    if (c == 0) {
      // The value and the first-order adjoints do not depend on the tangent
      // seeding, so any one sweep yields both.
      std::get<0>(res) = static_cast<double>(root.template get<0>());
      std::ranges::transform(grads, res_gradient.begin(), [](const T &g) {
        return static_cast<double>(g.template get<0>());
      });
    }

    // Row i received the sum over this colour's columns; the colouring
    // guarantees at most one of them is structurally nonzero in row i, so the
    // sum IS that entry, and which column it belongs to was resolved at compile
    // time.  Entries outside the pattern are never written and stay zero.
    for (auto &&[row, grad, target] :
         std::views::zip(res_hessian | std::views::chunk(N), grads,
                         kScatter[c])) {
      if (target != no_column) {
        // A chunk view indexes by its signed difference_type, not by size_t.
        row[static_cast<std::ranges::range_difference_t<decltype(row)>>(
            target)] = static_cast<double>(grad.template get<1>());
      }
    }
  });

  // Columns come from independent sweeps, so mirrored entries can differ in
  // the last ULP.
  detail::symmetrize(res_hessian.data(), N);
  return res;
}

// The nonzero VALUES of the Hessian, in the compressed-column order fixed by
// sparse_layout<Expr>() — the same colour-compressed sweeps as above, but
// scattered straight into compressed storage, so the dense N x N matrix is
// never materialised and the buffer is O(nnz) rather than O(N^2).
// No symmetrization pass: the layout already names each entry exactly once, and
// H(i,j) and H(j,i) are separate slots filled from the sweep of their own
// colour, so there is no pair to average.
//
// NNZ rides along as a defaulted template parameter — the same shape as
// sparse_layout() in coupling.hpp — because hessian_nnz is consteval: the size
// of this buffer is a property of Expr, so it is an array and not a vector, and
// the sparse path does not allocate.  std::vector is for the runtime-extent
// drivers above, where the active-variable count is a span and not a type.
template <CExpression Expr,
          std::size_t NNZ = hessian_nnz<std::remove_cvref_t<Expr>>()>
std::array<double, NNZ + 1> hessian_values_sparse(const Expr &expr,
                                                  std::span<const double> x) {
  using E = std::remove_cvref_t<Expr>;
  using T = typename E::value_type;
  using Syms = detail::expr_symbols_t<E>;
  constexpr std::size_t N = mpl::mp_size(Syms{});

  static constexpr auto kPattern = hessian_pattern<E>();
  static constexpr auto kColors = color_columns<N>(kPattern);

  // One cell past the nonzeros: layout_sparse_pattern maps every structural
  // zero onto that last slot, so H[i, j] answers for indices the pattern
  // excludes without the caller checking first.  A CSC consumer reads exactly nnz
  // entries from values.data(), so the extra cell is invisible to it.
  //
  // Value-initialised, which zeroes it: the sweeps below only write the slots
  // the pattern names, and every other cell — the sink included — has to read
  // back as exactly 0.0.
  std::array<double, NNZ + 1> values{};

  color_sweeps(expr, x, kColors,
               [&](std::size_t c, const T &, const auto &grads) {
                 static constexpr auto kSlots = sparse_slots<E>();
                 for (auto &&[grad, slot] : std::views::zip(grads, kSlots[c])) {
                   if (slot != no_column) {
                     values[slot] = static_cast<double>(grad.template get<1>());
                   }
                 }
               });
  return values;
}

// A graph can take the forward-over-reverse path only if its numbers carry a
// first-order tangent to seed — that is what makes one backward sweep yield a

} // namespace detail


// ---------------------------------------------------------------------------
// A sparse Hessian whose sparsity is a property of the expression TYPE.
//
// The structure comes from the compile-time coupling pass, so `outer` and
// `inner` are `static constexpr` and cost nothing at run time; only the nnz
// values are computed and stored.  Together they are a standard
// compressed-column (CSC) triple, which is what every sparse linear-algebra
// library consumes -- so handing the three of them over is all this needs to
// do, and it deliberately does not name any of those libraries.  Building an
// Eigen map from it is one line at the call site:
//
//   auto H = diff::sparse_hessian(expr, x);
//   Eigen::Map<const Eigen::SparseMatrix<double>> m{
//       H.rows, H.rows, H.nnz, H.outer().data(), H.inner().data(),
//       H.values().data()};
// ---------------------------------------------------------------------------
template <CExpression Expr> class SparseHessian {
  using E = std::remove_cvref_t<Expr>;
  static constexpr std::size_t kN = detail::expr_arity_v<E>;
  static constexpr auto kLayout = sparse_layout<E>();
  static constexpr std::size_t kNnz = decltype(kLayout)::nnz;

  // One past nnz: the dense view maps every structurally-absent (i, j) to a
  // sink cell so that reading one is a load rather than a branch.
  std::array<double, kNnz + 1> values_;

public:
  static constexpr std::size_t rows = kN;
  static constexpr std::size_t nnz = kNnz;

  explicit SparseHessian(std::array<double, kNnz + 1> values) noexcept
      : values_(values) {}

  // The CSC triple.  Column j occupies [outer()[j], outer()[j + 1]); inner()[k]
  // is the row of stored value k.  Both are compile-time constants.
  [[nodiscard]] static constexpr std::span<const int> outer() noexcept {
    return kLayout.outer;
  }
  [[nodiscard]] static constexpr std::span<const int> inner() noexcept {
    return kLayout.inner;
  }
  // The nnz values, without the sink cell.
  [[nodiscard]] std::span<const double> values() const & noexcept {
    return std::span<const double>{values_}.first(nnz);
  }
  auto values() const && = delete;

  // The compressed buffer read as the dense matrix it stands for.
  [[nodiscard]] auto view() const & noexcept {
    return sparse_matrix_view<E>(values_);
  }
  auto view() const && = delete;
  [[nodiscard]] double operator[](std::size_t i, std::size_t j) const noexcept {
    return view()[i, j];
  }
  // Whether (i, j) is in the pattern at all, as opposed to reading 0.0 because
  // the structure says it cannot be anything else.
  [[nodiscard]] static constexpr bool structural(std::size_t i,
                                                 std::size_t j) noexcept {
    return typename layout_sparse_pattern<E>::template mapping<
               md::extents<std::size_t, kN, kN>>{}
        .contains(i, j);
  }
};

// The sparse counterpart of hessian(graph, x).
template <CExpression Expr>
[[nodiscard]] SparseHessian<Expr> sparse_hessian(const Expr &expr,
                                                 std::span<const double> x) {
  return SparseHessian<Expr>{detail::hessian_values_sparse(expr, x)};
}

} // namespace diff
