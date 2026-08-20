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
#include <concepts>
#include <functional>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>

namespace diff::impl {

namespace mp = diff::impl::mpl;

template <CExpression Expr>
using node_cache_t = std::array<typename std::remove_cvref_t<Expr>::value_type,
                                node_count_v<std::remove_cvref_t<Expr>>>;

// Leaves hold no value, so a leaf reads its slot out of `vals` (canonical
// symbol order).  `Store` is entirely the parent's `reads_primals`: no adjoint
// rule reads its own slot.
template <std::size_t Base = 0, CSymbolList Syms, bool Store = true,
          CExpression E, CNumericBuffer Vals, CNumericBuffer Cache>
constexpr auto fill_cache(const E &node, const Vals &vals,
                          Cache &cache) noexcept {
  using U = std::remove_cvref_t<E>;
  using VT = typename U::value_type;
  if constexpr (CExpressionNode<U>) {
    using Kids = typename U::children_t;
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
// back from a root adjoint of 1.  Returns the root value; `grads` receives the
// gradient.
template <CSymbolList Syms, CExpression Expr, CNumericBuffer Seeds,
          CNumericBuffer Grads>
DIFF_ALWAYS_INLINE constexpr auto
reverse_sweep(const Expr &expr, const Seeds &seeds, Grads &grads) noexcept {
  using T = typename std::remove_cvref_t<Expr>::value_type;
  node_cache_t<Expr> cache{};
  // Store=false: the root's value is the return value, so storing it is dead.
  const T root = fill_cache<0, Syms, false>(expr, seeds, cache);
  expr.backward(Syms{}, T{1}, grads, cache);
  return root;
}

namespace detail {

// One backward sweep per colour, not per column; the harvester places the rows.
// Colours are an NTTP, not an argument: as a parameter the loop bound is a
// runtime load, the colour loop stops unrolling, and a dense Hessian measured
// slower than a plain column sweep.
template <auto Colors, CExpression Expr, CNumericBuffer Point, typename Harvest>
  requires std::invocable<
      Harvest &, std::size_t,
      const typename std::remove_cvref_t<Expr>::value_type &,
      std::array<typename std::remove_cvref_t<Expr>::value_type,
                 detail::expr_arity_v<std::remove_cvref_t<Expr>>> &>
DIFF_ALWAYS_INLINE constexpr void
color_sweeps(const Expr &expr, const Point &x, Harvest &&harvest) {
  using E = std::remove_cvref_t<Expr>;
  using T = typename E::value_type;
  using S = dual_scalar_t<T>;
  using Syms = detail::expr_symbols_t<E>;
  constexpr std::size_t N = mpl::mp_size(Syms{});

  // Only the seeded tangents move between colours.
  std::array<T, N> seeds{};
  std::ranges::transform(x, seeds.begin(), [](const auto &v) {
    return T{static_cast<S>(v), S{}};
  });

  // Seeds outside the colour are already zero, so only this one moves.
  static constexpr bool kIdentity =
      std::ranges::equal(Colors.color, std::views::iota(0uz, N));
  const auto toggle = [&](std::size_t c, const S &v) {
    if constexpr (kIdentity) {
      seeds[c].deriv() = v;
    } else {
      for (auto &&[seed, color] : std::views::zip(seeds, Colors.color)) {
        if (color == c) {
          seed.deriv() = v;
        }
      }
    }
  };

  for (std::size_t c = 0; c < Colors.count; ++c) {
    toggle(c, S{1});
    std::array<T, N> grads{};
    const T root = reverse_sweep<Syms>(expr, seeds, grads);
    harvest(c, root, grads);
    toggle(c, S{});
  }
}

} // namespace detail

// The value-array APIs take a positional array in CANONICAL (alphabetical, not
// source) order.  make_values() binds by name instead, so a missing, extra,
// duplicated or misspelled symbol is a compile error.
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

// Scalar defaults to what derivative_tensor expects; override it for the
// reverse-mode hessian path.
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

// Forward mode is reached through the drivers, which take a callable rather
// than a Mode, so it is not a value here.
enum class DiffMode { Symbolic, Reverse };

namespace detail {

// A dual-valued gradient carries the seed in its dual part; the gradient proper
// is the real part.
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

// Reverse-mode gradient.  A dual-valued expression sweeps identically and hands
// back the value level of each entry.
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
  using E = std::remove_cvref_t<Expr>;

  // Sparsity is a property of the type, so the colouring and the row -> column
  // map are constants and the sweep loop carries no search.
  static constexpr auto kPattern = hessian_pattern<E>();
  static constexpr auto kColors = color_columns<N>(kPattern);
  static constexpr auto kScatter = scatter_targets<N>(kPattern, kColors);

  // Value-initialised: the scatter writes only the pattern.  Symmetric-packed,
  // so (i, j) and (j, i) are one cell and there is nothing to symmetrise.
  nd_tensor_t<S, N, 2> H{};

  detail::color_sweeps<kColors>(
      expr, values, [&](std::size_t c, const T &, const auto &grads) {
        // At most one column per (colour, row), so that entry IS the sum.
        for (std::size_t i = 0; i < N; ++i) {
          const std::size_t target = kScatter[c][i];
          if (target != no_column) {
            H[i, target] = grads[i].template get<1>();
          }
        }
      });
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

// The cartesian product of Order copies of [0, N), in layout_right order.
template <std::size_t N, std::size_t Order>
[[nodiscard]] constexpr auto index_grid() noexcept {
  return []<std::size_t... D>(std::index_sequence<D...>) {
    return std::views::cartesian_product(
        ((void)D, std::views::iota(std::size_t{0}, N))...);
  }(std::make_index_sequence<Order>{});
}

// One canonical representative per permutation class: C(N + Order - 1, Order)
// of them.  Keep it a consteval table -- a filtered view and a lazy iterator
// were each measured 4x slower.
template <std::size_t N, std::size_t Order>
  requires(N > 0 && Order > 0)
consteval auto simplex_index_table() noexcept {
  // The layout's own cell count, so table and storage cannot disagree.
  std::array<std::array<std::size_t, Order>, binomial(N + Order - 1, Order)>
      out{};
  std::array<std::size_t, Order> idx{};
  for (auto &slot : out) {
    slot = idx;
    // Raise the rightmost position that can be raised and collapse the rest
    // onto it, which is what keeps the tuple sorted.
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

// Every variable lifted to nth_dual_t, with the tangents `idx` names on.
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

  // First order: J is a template parameter, so `k == J` folds away and the
  // passes are straight-line.  The generic loop below costs a runtime index
  // walk plus make_mixed_seed's recursive span walk for nothing.
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

// Same fold as binomial() in md/layouts.hpp.
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
[[nodiscard]] DIFF_ALWAYS_INLINE constexpr S
univariate_derivative_impl(const Expr &expr, S x0) noexcept {
  using symbols = detail::expr_symbols_t<std::remove_cvref_t<Expr>>;
  using TD = TaylorDual<S, Order>;

  TD seed;
  seed.c[0] = x0;
  seed.c[1] = S{1};

  TD result = expr.template eval_seeded<symbols>(std::array<TD, 1>{seed});

  constexpr S factorial = static_cast<S>(compile_time_factorial(Order));
  return result.c[Order] * factorial;
}

} // namespace detail

// Forward-over-reverse Hessian of an expression graph: the only O(N)-sweep
// Hessian driver.  Seeds a tangent, sweeps backward once per colour, harvests a
// whole set of rows per sweep.  hessian.hpp decides when it applies.

namespace detail {

// Seeding column j and running one backward sweep yields that whole Hessian
// column: N sweeps against the scalar driver's N(N+1)/2 probes, and the j == 0
// sweep hands back the value and gradient free.
template <CExpression Expr>
constexpr HessianStatic<
    mpl::mp_size(detail::expr_symbols_t<std::remove_cvref_t<Expr>>{})>
hessian_expr_reverse(const Expr &expr, std::span<const double> x) {
  using E = std::remove_cvref_t<Expr>;
  using T = typename E::value_type;
  using Syms = detail::expr_symbols_t<E>;
  constexpr std::size_t N = mpl::mp_size(Syms{});

  // Columns sharing no row are seeded in the same sweep.  A dense Hessian
  // colours in N, so the loop degenerates to one sweep per column, never worse.
  static constexpr auto kPattern = hessian_pattern<E>();
  static constexpr auto kColors = color_columns<N>(kPattern);
  // Value-initialised is load-bearing: the scatter writes only the pattern.
  HessianStatic<N> res{};
  auto &res_gradient = std::get<1>(res);
  auto &res_hessian = std::get<2>(res);

  color_sweeps<kColors>(expr, x, [&](std::size_t c, const T &root,
                                     const auto &grads) {
    static constexpr auto kScatter = scatter_targets<N>(kPattern, kColors);
    if (c == 0) {
      // Neither depends on the tangent seeding, so any one sweep yields both.
      std::get<0>(res) = static_cast<double>(root.template get<0>());
      std::ranges::transform(grads, res_gradient.begin(), [](const T &g) {
        return static_cast<double>(g.template get<0>());
      });
    }

    // At most one of this colour's columns is structurally nonzero in row i,
    // so the sum IS that entry; which column was resolved at compile time.
    for (auto &&[row, grad, target] :
         std::views::zip(res_hessian | std::views::chunk(N), grads,
                         kScatter[c])) {
      if (target != no_column) {
        row[static_cast<std::ranges::range_difference_t<decltype(row)>>(
            target)] = static_cast<double>(grad.template get<1>());
      }
    }
  });

  // Independent sweeps, so mirrored entries can differ in the last ULP.
  detail::symmetrize(res_hessian.data(), N);
  return res;
}

// The nonzero values in sparse_layout<Expr>() order: the same sweeps as above
// scattered straight into compressed storage.  No symmetrization -- the layout
// names each entry exactly once.  NNZ is defaulted from a consteval count, so
// the buffer is an array and this path does not allocate.
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

  // One cell past the nonzeros: the sink every structural zero maps onto.  A
  // CSC consumer reads exactly nnz entries and never sees it.
  std::array<double, NNZ + 1> values{};

  color_sweeps<kColors>(expr, x,
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

} // namespace detail


// A sparse Hessian whose structure is a property of the expression type, so
// `outer` and `inner` are static constexpr and only the nnz values are
// computed.  Together, a standard CSC triple.
template <CExpression Expr> class SparseHessian {
  using E = std::remove_cvref_t<Expr>;
  static constexpr std::size_t kN = detail::expr_arity_v<E>;
  static constexpr auto kLayout = sparse_layout<E>();
  static constexpr std::size_t kNnz = decltype(kLayout)::nnz;

  // One past nnz: the sink cell, so reading an absent (i, j) is a load.
  std::array<double, kNnz + 1> values_;

public:
  static constexpr std::size_t rows = kN;
  static constexpr std::size_t nnz = kNnz;

  explicit SparseHessian(std::array<double, kNnz + 1> values) noexcept
      : values_(values) {}

  // Column j occupies [outer()[j], outer()[j + 1]); inner()[k] is the row of
  // stored value k.
  [[nodiscard]] static constexpr std::span<const int> outer() noexcept {
    return kLayout.outer;
  }
  [[nodiscard]] static constexpr std::span<const int> inner() noexcept {
    return kLayout.inner;
  }
  [[nodiscard]] std::span<const double> values() const & noexcept {
    return std::span<const double>{values_}.first(nnz);
  }
  auto values() const && = delete;

  [[nodiscard]] auto view() const & noexcept {
    return sparse_matrix_view<E>(values_);
  }
  auto view() const && = delete;
  [[nodiscard]] double operator[](std::size_t i, std::size_t j) const noexcept {
    return view()[i, j];
  }
  // In the pattern, as opposed to reading 0.0 because it cannot be otherwise.
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

} // namespace diff::impl
