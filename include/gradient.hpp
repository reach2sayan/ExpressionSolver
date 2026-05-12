#pragma once

#include "dual.hpp"
#include "expressions.hpp"
#include "taylor_dual.hpp"
#include "traits.hpp"
#include <array>
#include <boost/mp11/algorithm.hpp>

namespace diff {

namespace mp = boost::mp11;

// Compile-time N-dimensional array: nd_array_t<S, N, Order> is std::array
// nested Order times.
template <typename S, std::size_t N, std::size_t Order> struct nd_array {
  using type = std::array<typename nd_array<S, N, Order - 1>::type, N>;
};
template <typename S, std::size_t N> struct nd_array<S, N, 0> {
  using type = S;
};
template <typename S, std::size_t N, std::size_t Order>
using nd_array_t = typename nd_array<S, N, Order>::type;

// arr[idx[0]][idx[1]]...[idx[Order-1]]
template <std::size_t Order>
constexpr auto &nd_index(auto &arr, const std::size_t *idx) {
  if constexpr (Order == 0) {
    return arr;
  } else {
    return nd_index<Order - 1>(arr[idx[0]], idx + 1);
  }
}

enum class DiffMode { Symbolic, Forward, Reverse, ParallelReverse };
namespace detail {

template <CExpression Expr, typename T = typename Expr::value_type>
  requires(!is_dual_v<T>)
[[nodiscard]] constexpr auto reverse_mode_gradient(const Expr &expr) {
  using Syms =
      extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  constexpr auto N = mp::mp_size<Syms>::value;
  std::array<T, N> grads{};
  expr.backward(Syms{}, T{1}, grads);
  return grads;
}

template <CExpression Expr, typename T = typename Expr::value_type>
  requires is_dual_v<T>
[[nodiscard]] constexpr auto reverse_mode_gradient(const Expr &expr) {
  using scalar_t = dual_scalar_t<T>;
  using Syms =
      extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  constexpr auto N = mp::mp_size<Syms>::value;
  std::array<T, N> grads{};
  expr.backward(Syms{}, T{1}, grads);
  std::array<scalar_t, N> result{};
  for (std::size_t i = 0; i < N; i++) {
    result[i] = grads[i].template get<0>();
  }
  return result;
}

template <CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = dual_scalar_t<T>,
          std::size_t N = mp::mp_size<extract_symbols_from_expr_t<
              std::remove_cvref_t<Expr>>>::value>
  requires is_dual_v<T>
[[nodiscard]] auto reverse_mode_hessian(Expr &expr, std::array<S, N> values) {
  using symbols =
      extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  std::array<std::array<S, N>, N> H{};
  std::array<T, N> seeds{};
  for (std::size_t j = 0; j < N; j++) {
    for (std::size_t i = 0; i < N; i++) {
      seeds[i] = T{values[i], i == j ? S{1} : S{}};
    }
    expr.update(symbols{}, seeds);
    std::array<T, N> grads{};
    expr.backward(symbols{}, T{1}, grads);
    for (std::size_t i = 0; i < N; i++) {
      H[i][j] = grads[i].template get<1>();
    }
  }
  for (std::size_t i = 0; i < N; i++) {
    seeds[i] = T{values[i], S{}};
  }
  expr.update(symbols{}, seeds);
  return H;
}

template <CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = dual_scalar_t<T>,
          std::size_t N = mp::mp_size<extract_symbols_from_expr_t<
              std::remove_cvref_t<Expr>>>::value>
  requires is_dual_v<T>
[[nodiscard]] auto reverse_mode_hessian(Expr &expr) {
  using symbols =
      extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  std::array<T, N> current{};
  expr.collect(symbols{}, current);
  std::array<S, N> values{};
  for (std::size_t i = 0; i < N; i++) {
    values[i] = current[i].template get<0>();
  }
  return reverse_mode_hessian(expr, values);
}

// ---------------------------------------------------------------------------
// Forward-mode: mixed-partial seed construction.
//
// make_mixed_seed<S, Depth>(value, idx, k):
//   Builds the Dual^Depth<S> seed for variable k given a multi-index
//   idx[0..Depth-1], where idx[0] is the outermost direction and
//   idx[Depth-1] is the innermost.  All cross-terms start at zero so
//   only the intended mixed partial is non-zero in the result.
//
//   Extraction: result.get<1>().get<1>()... (Depth times)
//               = ∂^Depth f / ∂x_{idx[0]} ∂x_{idx[1]} ... ∂x_{idx[Depth-1]}
// ---------------------------------------------------------------------------
template <typename S, std::size_t Depth>
constexpr nth_dual_t<S, Depth> make_mixed_seed(S value, const std::size_t *idx,
                                               std::size_t k) {
  if constexpr (Depth == 0) {
    return value;
  } else if constexpr (Depth == 1) {
    return nth_dual_t<S, 1>{value, k == idx[0] ? S{1} : S{}};
  } else {
    auto inner = make_mixed_seed<S, Depth - 1>(value, idx + 1, k);
    auto outer_tangent = embed_constant<S, Depth - 1>(k == idx[0] ? S{1} : S{});
    return nth_dual_t<S, Depth>{inner, outer_tangent};
  }
}

// Chain .get<1>() N times: extracts the order-N tangent component.
template <std::size_t N, typename T> constexpr auto extract_nth(const T &x) {
  if constexpr (N == 0)
    return x;
  else
    return extract_nth<N - 1>(x.template get<1>());
}

// ---------------------------------------------------------------------------
// Core forward derivative_tensor implementation.
// Returns nd_array_t<S, N, Order> — nested std::array of rank Order.
// result[i₁][i₂]...[iOrder] = ∂^Order f / ∂x_{i₁} ∂x_{i₂} ... ∂x_{iOrder}
// ---------------------------------------------------------------------------
template <std::size_t Order, CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = scalar_base_t<T>,
          std::size_t N = mp::mp_size<extract_symbols_from_expr_t<
              std::remove_cvref_t<Expr>>>::value>
  requires(Order > 0 && N > 0)
[[nodiscard]] auto derivative_tensor_impl(const Expr &expr,
                                          std::array<S, N> values) {
  using symbols =
      extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  using U = nth_dual_t<S, Order>;

  nd_array_t<S, N, Order> result{};

  std::size_t total = 1;
  for (std::size_t d = 0; d < Order; ++d) {
    total *= N;
  }

  for (std::size_t flat = 0; flat < total; ++flat) {
    std::array<std::size_t, Order> idx{};
    std::size_t tmp = flat;
    for (int d = (int)Order - 1; d >= 0; --d) {
      idx[d] = tmp % N;
      tmp /= N;
    }

    std::array<U, N> seeds{};
    for (std::size_t k = 0; k < N; ++k) {
      seeds[k] = make_mixed_seed<S, Order>(values[k], idx.data(), k);
    }
    U val = expr.template eval_seeded_as<U, symbols>(seeds);
    nd_index<Order>(result, idx.data()) = extract_nth<Order>(val);
  }
  return result;
}

} // namespace detail

// ===========================================================================
// Public API — reverse mode
// ===========================================================================

template <DiffMode Mode, CExpression Expr>
  requires(Mode == DiffMode::Reverse)
[[nodiscard]] constexpr auto gradient(const Expr &expr) {
  return detail::reverse_mode_gradient(expr);
}

template <DiffMode Mode, CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = dual_scalar_t<T>,
          std::size_t N = mp::mp_size<extract_symbols_from_expr_t<
              std::remove_cvref_t<Expr>>>::value>
  requires(Mode == DiffMode::Reverse && is_dual_v<T>)
[[nodiscard]] auto hessian(Expr &expr, std::array<S, N> values) {
  return detail::reverse_mode_hessian(expr, values);
}

template <DiffMode Mode, CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = dual_scalar_t<T>,
          std::size_t N = mp::mp_size<extract_symbols_from_expr_t<
              std::remove_cvref_t<Expr>>>::value>
  requires(Mode == DiffMode::Reverse && is_dual_v<T>)
[[nodiscard]] auto hessian(Expr &expr) {
  return detail::reverse_mode_hessian(expr);
}

// ===========================================================================
// Public API — forward mode (derivative_tensor)
//
// derivative_tensor<Order>(expr [, values])
//   → nd_array_t<S, N, Order>  (nested std::array, rank = Order, extent N each)
//   → result[i₁][i₂]...[iOrder] = ∂^Order f / ∂x_{i₁} ∂x_{i₂} ... ∂x_{iOrder}
//
// Works for any expression depth (Variable<T>, Variable<Dual<T>>, etc.).
// Input is always the base scalar S = scalar_base_t<value_type>; the library
// promotes internally to nth_dual_t<S, Order> via eval_seeded_as.
//
// Order = 1 → gradient (rank-1 tensor)
// Order = 2 → full Hessian including mixed partials (rank-2 tensor)
// Order = k → full k-th order derivative tensor
// ===========================================================================

template <std::size_t Order, CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = scalar_base_t<T>,
          std::size_t N = mp::mp_size<extract_symbols_from_expr_t<
              std::remove_cvref_t<Expr>>>::value>
  requires(Order > 0 && N > 0)
[[nodiscard]] auto derivative_tensor(const Expr &expr,
                                     std::array<S, N> values) {
  return detail::derivative_tensor_impl<Order>(expr, values);
}

template <std::size_t Order, CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = scalar_base_t<T>,
          std::size_t N = mp::mp_size<extract_symbols_from_expr_t<
              std::remove_cvref_t<Expr>>>::value>
  requires(Order > 0 && N > 0)
[[nodiscard]] auto derivative_tensor(const Expr &expr) {
  using symbols =
      extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  std::array<T, N> current{};
  expr.collect(symbols{}, current);
  std::array<S, N> values{};
  for (std::size_t i = 0; i < N; ++i)
    values[i] = get_real_part<dual_depth_v<T>>(current[i]);
  return detail::derivative_tensor_impl<Order>(expr, values);
}

// ===========================================================================
// univariate_derivative<N>(expr [, x0])
//
// N-th order derivative of a single-variable expression using TaylorDual<S,N>.
// Stores only N+1 coefficients (vs 2^N for nth_dual_t), making it O(N²) per
// operation instead of O(2^N).
//
// Returns the raw N-th derivative value (not normalised by N!).
// Requires the expression to contain exactly one Variable symbol.
// ===========================================================================

namespace detail {

template <typename T> constexpr T compile_time_factorial(T Order) {
  T result = 1;
  for (std::size_t i = 1; i <= Order; ++i) {
    result *= static_cast<T>(i);
  }
  return result;
}

template <std::size_t Order, CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = scalar_base_t<T>,
          std::size_t NVars = mp::mp_size<extract_symbols_from_expr_t<
              std::remove_cvref_t<Expr>>>::value>
  requires(Order > 0 && NVars == 1)
[[nodiscard]] S univariate_derivative_impl(const Expr &expr, S x0) {
  using symbols =
      extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  using TD = TaylorDual<S, Order>;

  // Seed: value = x0, first tangent = 1, higher tangents = 0.
  TD seed;
  seed.c[0] = x0;
  seed.c[1] = S{1};

  TD result =
      expr.template eval_seeded_as<TD, symbols>(std::array<TD, 1>{seed});

  // c[k] = f^(k)(x0) / k!  →  multiply by Order! to recover the derivative.
  S factorial = compile_time_factorial(Order);
  // for (std::size_t i = 1; i <= Order; ++i) factorial *= static_cast<S>(i);
  return result.c[Order] * factorial;
}

} // namespace detail

// With explicit evaluation point:
template <std::size_t Order, CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = scalar_base_t<T>,
          std::size_t NVars = mp::mp_size<extract_symbols_from_expr_t<
              std::remove_cvref_t<Expr>>>::value>
  requires(Order > 0 && NVars == 1)
[[nodiscard]] S univariate_derivative(const Expr &expr, S x0) {
  return detail::univariate_derivative_impl<Order>(expr, x0);
}

// Read evaluation point from the expression's current variable values:
template <std::size_t Order, CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = scalar_base_t<T>,
          std::size_t NVars = mp::mp_size<extract_symbols_from_expr_t<
              std::remove_cvref_t<Expr>>>::value>
  requires(Order > 0 && NVars == 1)
[[nodiscard]] S univariate_derivative(const Expr &expr) {
  using symbols =
      extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;
  std::array<T, 1> current{};
  expr.collect(symbols{}, current);
  S x0 = get_real_part<dual_depth_v<T>>(current[0]);
  return detail::univariate_derivative_impl<Order>(expr, x0);
}

} // namespace diff

#define reverse_mode_grad gradient<diff::DiffMode::Reverse>
#define reverse_mode_hess hessian<diff::DiffMode::Reverse>
