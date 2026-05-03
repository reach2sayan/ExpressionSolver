#pragma once

#include "dual.hpp"
#include "expressions.hpp"
#include "traits.hpp"
#include <array>
#include <boost/mp11/algorithm.hpp>
#include <ranges>

namespace diff {

namespace mp = boost::mp11;

enum class DiffMode { Symbolic, Forward, Reverse };

// ===========================================================================
// detail — implementation functions; use the public wrappers below.
// ===========================================================================
namespace detail {

// GCC requires always_inline on each per-pass lambda so it can constant-fold
// through the expression tree. Clang folds each pass independently at a
// manageable size without the hint — forcing always_inline merges all N passes
// into one function body that exceeds Clang's optimisation threshold.
#if defined(__GNUC__) && !defined(__clang__)
#define DIFF_PASS_INLINE __attribute__((always_inline))
#else
#define DIFF_PASS_INLINE
#endif

template <CExpression Expr, typename T = typename Expr::value_type>
  requires(!is_dual_v<T>)
[[nodiscard]] constexpr auto reverse_mode_gradient(const Expr &expr) {
  using Syms =
      typename extract_symbols_from_expr<std::remove_cvref_t<Expr>>::type;
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
      typename extract_symbols_from_expr<std::remove_cvref_t<Expr>>::type;
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
          typename TArr =
              dual_scalar_t<typename std::remove_cvref_t<Expr>::value_type>,
          std::size_t N = mp::mp_size<typename extract_symbols_from_expr<
              std::remove_cvref_t<Expr>>::type>::value>
[[nodiscard]] auto forward_mode_gradient(Expr &expr, std::array<TArr, N> values)
  requires is_dual_v<typename std::remove_cvref_t<Expr>::value_type>
{
  using expr_type = std::remove_cvref_t<Expr>;
  using value_type = typename expr_type::value_type;
  using scalar_type = dual_scalar_t<value_type>;
  using symbols = typename extract_symbols_from_expr<expr_type>::type;
  constexpr std::size_t n = mp::mp_size<symbols>::value;

  std::array<value_type, n> seeds{};
  std::array<scalar_type, n> gradients{};

  // One outer lambda exposes compile-time Js to both the init fold and the
  // per-pass lambdas. Each inner lambda is independently sized for Clang's
  // inliner; DIFF_PASS_INLINE forces GCC to inline them so it can see through
  // to seeds/expr and constant-fold compile-time-known inputs.
  [&]<std::size_t... Js>(std::index_sequence<Js...>) {
    ((seeds[Js] = value_type{values[Js], scalar_type{}}), ...);
    (
        [&]() DIFF_PASS_INLINE {
          seeds[Js] = value_type{values[Js], scalar_type{1}};
          expr.update(symbols{}, seeds);
          gradients[Js] = expr.eval().template get<1>();
          seeds[Js] = value_type{values[Js], scalar_type{}};
        }(),
        ...);
  }(std::make_index_sequence<n>{});

  expr.update(symbols{}, seeds);
  return gradients;
}

template <CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = dual_scalar_t<T>,
          std::size_t N = mp::mp_size<typename extract_symbols_from_expr<
              std::remove_cvref_t<Expr>>::type>::value>
  requires is_dual_v<T>
[[nodiscard]] auto reverse_mode_hessian(Expr &expr, std::array<S, N> values) {
  using symbols =
      typename extract_symbols_from_expr<std::remove_cvref_t<Expr>>::type;

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
          std::size_t N = mp::mp_size<typename extract_symbols_from_expr<
              std::remove_cvref_t<Expr>>::type>::value>
  requires is_dual_v<T>
[[nodiscard]] auto reverse_mode_hessian(Expr &expr) {
  using symbols =
      typename extract_symbols_from_expr<std::remove_cvref_t<Expr>>::type;
  std::array<T, N> current{};
  expr.collect(symbols{}, current);
  std::array<S, N> values{};
  for (std::size_t i = 0; i < N; i++) {
    values[i] = current[i].template get<0>();
  }
  return reverse_mode_hessian(expr, values);
}

template <CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename D = dual_scalar_t<T>, typename S = dual_scalar_t<D>,
          std::size_t N = mp::mp_size<typename extract_symbols_from_expr<
              std::remove_cvref_t<Expr>>::type>::value>
  requires is_dual_v<T> && is_dual_v<D>
[[nodiscard]] constexpr auto forward_mode_hessian(const Expr &expr,
                                                  std::array<S, N> values) {
  using symbols =
      typename extract_symbols_from_expr<std::remove_cvref_t<Expr>>::type;

  std::array<std::array<S, N>, N> H{};
  static_for<N>([&]<std::size_t I>() {
    static_for<N>([&]<std::size_t J>() {
      std::array<T, N> seeds{};
      static_for<N>([&]<std::size_t K>() {
        seeds[K] =
            T{D{values[K], K == J ? S{1} : S{}}, D{K == I ? S{1} : S{}, S{}}};
      });
      H[I][J] = expr.template eval_seeded<symbols>(seeds)
                    .template get<1>()
                    .template get<1>();
    });
  });

  return H;
}

template <CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename D = dual_scalar_t<T>, typename S = dual_scalar_t<D>,
          std::size_t N = mp::mp_size<typename extract_symbols_from_expr<
              std::remove_cvref_t<Expr>>::type>::value>
  requires is_dual_v<T> && is_dual_v<D>
[[nodiscard]] constexpr auto forward_mode_hessian(const Expr &expr) {
  using symbols =
      typename extract_symbols_from_expr<std::remove_cvref_t<Expr>>::type;
  std::array<T, N> current{};
  expr.collect(symbols{}, current);
  std::array<S, N> values{};
  for (std::size_t i = 0; i < N; i++) {
    values[i] = current[i].template get<0>().template get<0>();
  }
  return forward_mode_hessian(expr, values);
}

// ---------------------------------------------------------------------------
// Nth-derivative helpers
// ---------------------------------------------------------------------------

// Lift scalar val into nth_dual_t<T,N> with all dual parts zero (constant).
template <typename T, std::size_t N>
constexpr nth_dual_t<T, N> embed_constant(T val) {
  if constexpr (N == 0) {
    return val;
  } else {
    return nth_dual_t<T, N>{embed_constant<T, N - 1>(val),
                            nth_dual_t<T, N - 1>{}};
  }
}

// Seed for the Nth derivative of a single variable:
//   seed<1>(v) = {v, 1}
//   seed<N>(v) = {seed<N-1>(v), embed_constant<N-1>(1)}
// Chain .get<1>() N times on f(seed) to recover f^(N)(v).
template <typename T, std::size_t N>
constexpr nth_dual_t<T, N> make_nth_seed(T val) {
  if constexpr (N == 0) {
    return val;
  } else if constexpr (N == 1) {
    return nth_dual_t<T, 1>{val, T{1}};
  } else {
    return nth_dual_t<T, N>{make_nth_seed<T, N - 1>(val),
                            embed_constant<T, N - 1>(T{1})};
  }
}

// Chain .get<1>() N times: extracts the order-N tangent component.
template <std::size_t N, typename T> constexpr auto extract_nth(const T &x) {
  if constexpr (N == 0) {
    return x;
  } else {
    return extract_nth<N - 1>(x.template get<1>());
  }
}

// Chain .get<0>() N times: recovers the base scalar from a nested dual.
template <std::size_t N, typename T> constexpr auto get_real_part(const T &x) {
  if constexpr (N == 0) {
    return x;
  } else {
    return get_real_part<N - 1>(x.template get<0>());
  }
}

} // namespace detail

// ===========================================================================
// Public API — select mode with DiffMode::Forward or DiffMode::Reverse.
// ===========================================================================

template <DiffMode Mode, CExpression Expr>
  requires(Mode == DiffMode::Reverse)
[[nodiscard]] constexpr auto gradient(const Expr &expr) {
  return detail::reverse_mode_gradient(expr);
}

template <DiffMode Mode, CExpression Expr,
          typename TArr =
              dual_scalar_t<typename std::remove_cvref_t<Expr>::value_type>,
          std::size_t N = mp::mp_size<typename extract_symbols_from_expr<
              std::remove_cvref_t<Expr>>::type>::value>
  requires(Mode == DiffMode::Forward &&
           is_dual_v<typename std::remove_cvref_t<Expr>::value_type>)
[[nodiscard]] auto gradient(Expr &expr, std::array<TArr, N> values) {
  return detail::forward_mode_gradient(expr, values);
}

template <DiffMode Mode, CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = dual_scalar_t<T>,
          std::size_t N = mp::mp_size<typename extract_symbols_from_expr<
              std::remove_cvref_t<Expr>>::type>::value>
  requires(Mode == DiffMode::Reverse && is_dual_v<T>)
[[nodiscard]] auto hessian(Expr &expr, std::array<S, N> values) {
  return detail::reverse_mode_hessian(expr, values);
}

template <DiffMode Mode, CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = dual_scalar_t<T>,
          std::size_t N = mp::mp_size<typename extract_symbols_from_expr<
              std::remove_cvref_t<Expr>>::type>::value>
  requires(Mode == DiffMode::Reverse && is_dual_v<T>)
[[nodiscard]] auto hessian(Expr &expr) {
  return detail::reverse_mode_hessian(expr);
}

template <DiffMode Mode, CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename D = dual_scalar_t<T>, typename S = dual_scalar_t<D>,
          std::size_t N = mp::mp_size<typename extract_symbols_from_expr<
              std::remove_cvref_t<Expr>>::type>::value>
  requires(Mode == DiffMode::Forward && is_dual_v<T> && is_dual_v<D>)
[[nodiscard]] constexpr auto hessian(const Expr &expr,
                                     std::array<S, N> values) {
  return detail::forward_mode_hessian(expr, values);
}

template <DiffMode Mode, CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename D = dual_scalar_t<T>, typename S = dual_scalar_t<D>,
          std::size_t N = mp::mp_size<typename extract_symbols_from_expr<
              std::remove_cvref_t<Expr>>::type>::value>
  requires(Mode == DiffMode::Forward && is_dual_v<T> && is_dual_v<D>)
[[nodiscard]] constexpr auto hessian(const Expr &expr) {
  return detail::forward_mode_hessian(expr);
}

// ===========================================================================
// nth_derivative<Order> — pure Order-th partial derivative per variable.
//
// Returns std::array<S, N_vars> where entry i = ∂^Order f / ∂x_i^Order.
// Mixed partials are NOT computed; use hessian() for ∂²f/∂xᵢ∂xⱼ.
//
// Requires: Expr::value_type == nth_dual_t<S, Order>
//   i.e. the expression must be built with Order-deep Dual variables.
// ===========================================================================

template <std::size_t Order, CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = scalar_base_t<T>,
          std::size_t N = mp::mp_size<typename extract_symbols_from_expr<
              std::remove_cvref_t<Expr>>::type>::value>
  requires(dual_depth_v<T> == Order && Order > 0 && N > 0)
[[nodiscard]] constexpr auto nth_derivative(const Expr &expr,
                                            std::array<S, N> values) {
  using symbols =
      typename extract_symbols_from_expr<std::remove_cvref_t<Expr>>::type;
  std::array<S, N> result{};
  static_for<N>([&]<std::size_t I>() {
    std::array<T, N> seeds{};
    static_for<N>([&]<std::size_t J>() {
      if constexpr (I == J) {
        seeds[J] = detail::make_nth_seed<S, Order>(values[J]);
      }
      else {
        seeds[J] = detail::embed_constant<S, Order>(values[J]);
      }
    });
    result[I] =
        detail::extract_nth<Order>(expr.template eval_seeded<symbols>(seeds));
  });
  return result;
}

template <std::size_t Order, CExpression Expr,
          typename T = typename std::remove_cvref_t<Expr>::value_type,
          typename S = scalar_base_t<T>,
          std::size_t N = mp::mp_size<typename extract_symbols_from_expr<
              std::remove_cvref_t<Expr>>::type>::value>
  requires(dual_depth_v<T> == Order && Order > 0 && N > 0)
[[nodiscard]] constexpr auto nth_derivative(const Expr &expr) {
  using symbols =
      typename extract_symbols_from_expr<std::remove_cvref_t<Expr>>::type;
  std::array<T, N> current{};
  expr.collect(symbols{}, current);
  std::array<S, N> values{};
  for (std::size_t i = 0; i < N; ++i) {
    values[i] = detail::get_real_part<Order>(current[i]);
  }
  return nth_derivative<Order>(expr, values);
}

} // namespace diff

#define reverse_mode_grad gradient<diff::DiffMode::Reverse>
#define forward_mode_grad gradient<diff::DiffMode::Forward>
#define reverse_mode_hess hessian<diff::DiffMode::Reverse>
#define forward_mode_hess hessian<diff::DiffMode::Forward>
