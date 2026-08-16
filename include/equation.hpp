#pragma once
#include "dual.hpp"
#include "gradient.hpp"
#include "mpl.hpp"
#include <algorithm>
#include <array>
#include <ranges>

namespace diff {

namespace mp = diff::mpl;

namespace detail {
// Evaluate a tuple of expressions at one point, in canonical symbol order.
template <typename Syms, typename Vals, typename... Es>
constexpr auto eval_all(const Vals &vals, const Es &...es) noexcept {
  return std::array{es.template eval_seeded<Syms>(vals)...};
}
} // namespace detail

template <typename... Ts>
constexpr std::ostream &print_tup(std::ostream &out,
                                  const std::tuple<Ts...> &tup) {
  out << "(\n";
  bool first = true;

  static_for<sizeof...(Ts)>([&]<std::size_t I>() {
    if (!first) {
      out << "\n";
    }
    out << std::get<I>(tup);
    first = false;
  });
  out << "\n)";
  return out;
}

template <typename... Syms, CExpression Expr>
constexpr auto make_derivatives(mp::mp_list<Syms...>,
                                const Expr &expr) noexcept {
  return std::tuple(
      make_all_constant_except<Syms::value>(expr).derivative()...);
}

template <CExpression... Ts> class Equation;

template <typename... Syms, typename... Exprs>
constexpr auto make_jac_rows(const std::tuple<Exprs...> &es,
                             mp::mp_list<Syms...> = {}) noexcept {
  return std::apply(
      [](const auto &...exprs) noexcept {
        return std::make_tuple(
            make_derivatives(mp::mp_list<Syms...>{}, exprs)...);
      },
      es);
}

template <CExpression TFirst, CExpression... TRest>
  requires((CSameValueType<TFirst, TRest> && ...))
class Equation<TFirst, TRest...> {
public:
  using value_type = typename TFirst::value_type;
  using symbols =
      sort_tuple_t<tuple_union_t<extract_symbols_from_expr_t<TFirst>,
                                 extract_symbols_from_expr_t<TRest>...>>;

  static constexpr std::size_t output_dim = 1 + sizeof...(TRest);
  static constexpr std::size_t input_dim = mp::mp_size(symbols{});
  static constexpr std::size_t number_of_derivatives = input_dim;

private:
  using Exprs = std::tuple<TFirst, TRest...>;
  using jacobian_t = decltype(make_jac_rows(std::declval<Exprs>(), symbols{}));
  Exprs expressions;
  jacobian_t jacobian_data;

  friend std::ostream &operator<<(std::ostream &out, const Equation &ve) {
    static_for<output_dim>([&]<std::size_t I>() {
      out << "f" << I << ": " << std::get<I>(ve.expressions);
      out << " grad: ";
      print_tup(out, std::get<I>(ve.jacobian_data));
      out << '\n';
    });
    return out;
  }

  using point_t = std::array<value_type, input_dim>;

  [[nodiscard]] constexpr auto
  jacobian_symbolic(const point_t &vals) const noexcept
    requires(input_dim > 0)
  {
    std::array<std::array<value_type, input_dim>, output_dim> J{};
    static_for<output_dim>([&]<std::size_t I>() {
      J[I] = std::apply(
          [&](const auto &...ds) {
            return detail::eval_all<symbols>(vals, ds...);
          },
          std::get<I>(jacobian_data));
    });
    return J;
  }

  [[nodiscard]] constexpr auto
  jacobian_reverse_mode(const point_t &vals) const noexcept
    requires(input_dim > 0)
  {
    std::array<std::array<value_type, input_dim>, output_dim> J{};
    static_for<output_dim>([&]<std::size_t I>() {
      const auto &e = std::get<I>(expressions);
      node_cache_t<std::remove_cvref_t<decltype(e)>> cache{};
      fill_cache<0, symbols>(e, vals, cache);
      e.backward(symbols{}, value_type{1}, J[I], cache);
    });
    return J;
  }

  [[nodiscard]] constexpr auto hessian_forward_over_reverse(
      const std::array<dual_scalar_t<value_type>, input_dim> &values)
      const noexcept
    requires(DualLike<value_type> && input_dim > 0)
  {
    using S = dual_scalar_t<value_type>;
    std::array<std::array<std::array<S, input_dim>, input_dim>, output_dim> H{};

    // Values are the point and never move; only the seeded tangent does.
    point_t seeds{};
    std::ranges::transform(values, seeds.begin(),
                           [](const S &v) { return value_type{v, S{}}; });

    for (std::size_t j = 0; j < input_dim; ++j) {
      // Seed column j; one reverse sweep per output yields that column.
      seeds[j].deriv() = S{1};
      static_for<output_dim>([&]<std::size_t K>() {
        point_t grads{};
        const auto &e = std::get<K>(expressions);
        node_cache_t<std::remove_cvref_t<decltype(e)>> cache{};
        fill_cache<0, symbols>(e, seeds, cache);
        e.backward(symbols{}, value_type{1}, grads, cache);

        const auto column =
            grads | std::views::transform(
                        [](const value_type &g) { return g.template get<1>(); });
        for (auto &&[row, entry] : std::views::zip(H[K], column)) {
          row[j] = entry;
        }
      });
      seeds[j].deriv() = S{};
    }
    return H;
  }

  template <std::size_t Order>
  [[nodiscard]] constexpr auto equation_derivative_tensor_impl(
      std::array<scalar_base_t<value_type>, input_dim> values) const noexcept
    requires(input_dim > 0 && Order > 0)
  {
    using S = scalar_base_t<value_type>;
    using U = nth_dual_t<S, Order>;
    std::array<nd_array_t<S, input_dim, Order>, output_dim> result{};

    std::size_t total = 1;
    for (std::size_t d = 0; d < Order; ++d) {
      total *= input_dim;
    }

    for (std::size_t flat = 0; flat < total; ++flat) {
      std::array<std::size_t, Order> idx{};
      std::size_t tmp = flat;
      for (int d = (int)Order - 1; d >= 0; --d) {
        idx[d] = tmp % input_dim;
        tmp /= input_dim;
      }

      std::array<U, input_dim> seeds{};
      for (std::size_t k = 0; k < input_dim; ++k) {
        seeds[k] = detail::make_mixed_seed<S, Order>(values[k], idx, k);
      }

      static_for<output_dim>([&]<std::size_t OUT>() {
        U val = std::get<OUT>(expressions)
                    .template eval_seeded_as<U, symbols>(seeds);
        nd_index<Order>(result[OUT], idx) = detail::extract_nth<Order>(val);
      });
    }
    return result;
  }

public:
  constexpr Equation(TFirst first, TRest... rest) noexcept
      : expressions{std::move(first), std::move(rest)...},
        jacobian_data{make_jac_rows(expressions, symbols{})} {}

  [[nodiscard]] constexpr auto evaluate(const point_t &vals) const noexcept {
    if constexpr (output_dim == 1) {
      return std::get<0>(expressions).template eval_seeded<symbols>(vals);
    } else {
      return std::apply(
          [&](const auto &...es) {
            return detail::eval_all<symbols>(vals, es...);
          },
          expressions);
    }
  }

  [[nodiscard]] constexpr auto
  eval_derivatives(const point_t &vals) const noexcept
    requires(output_dim == 1)
  {
    const auto &row = std::get<0>(jacobian_data);
    std::array<value_type, input_dim> result{};
    static_for<input_dim>([&]<std::size_t I>() {
      result[I] = std::get<I>(row).template eval_seeded<symbols>(vals);
    });
    return result;
  }

  // Slot 0 is the expression itself; slot k>0 is d/d(k-1 th symbol), in
  // canonical symbol order.  The index is a template argument, so no tag type
  // is involved.
  // Lvalue-only: these alias `expressions` / `jacobian_data`.  The rvalue
  // overload below returns a copy instead, which post-refactor costs nothing —
  // the nodes it hands back are empty types.
  template <std::size_t N>
  [[nodiscard]] constexpr decltype(auto) get() & noexcept
    requires(output_dim == 1 && N <= input_dim)
  {
    if constexpr (N == 0) {
      return std::get<0>(expressions);
    } else {
      return std::get<N - 1>(std::get<0>(jacobian_data));
    }
  }

  template <std::size_t N>
  [[nodiscard]] constexpr decltype(auto) get() const & noexcept
    requires(output_dim == 1 && N <= input_dim)
  {
    if constexpr (N == 0) {
      return std::get<0>(expressions);
    } else {
      return std::get<N - 1>(std::get<0>(jacobian_data));
    }
  }

  template <std::size_t N>
  [[nodiscard]] constexpr auto get() const && noexcept
    requires(output_dim == 1 && N <= input_dim)
  {
    if constexpr (N == 0) {
      return std::get<0>(expressions);
    } else {
      return std::get<N - 1>(std::get<0>(jacobian_data));
    }
  }

  // Subscript spelling of the same thing.  operator[] has no template-argument
  // syntax, so the index arrives as an empty idx_t value (see idx<N>()).
  template <std::size_t N>
  [[nodiscard]] constexpr decltype(auto) operator[](idx_t<N>) & noexcept
    requires(output_dim == 1 && N <= input_dim)
  {
    return get<N>();
  }

  template <std::size_t N>
  [[nodiscard]] constexpr decltype(auto) operator[](idx_t<N>) const & noexcept
    requires(output_dim == 1 && N <= input_dim)
  {
    return get<N>();
  }

  template <std::size_t N>
  [[nodiscard]] constexpr auto operator[](idx_t<N>) const && noexcept
    requires(output_dim == 1 && N <= input_dim)
  {
    return std::move(*this).template get<N>();
  }

  template <DiffMode Mode>
  [[nodiscard]] constexpr auto jacobian(const point_t &values) const noexcept
    requires(Mode == DiffMode::Symbolic && input_dim > 0)
  {
    return jacobian_symbolic(values);
  }

  template <DiffMode Mode>
  [[nodiscard]] constexpr auto jacobian(const point_t &values) const noexcept
    requires(Mode == DiffMode::Reverse && input_dim > 0)
  {
    return jacobian_reverse_mode(values);
  }

  template <DiffMode Mode>
  [[nodiscard]] constexpr auto
  hessian(const std::array<dual_scalar_t<value_type>, input_dim> &values)
      const noexcept
    requires(Mode == DiffMode::Reverse && DualLike<value_type> && input_dim > 0)
  {
    return hessian_forward_over_reverse(values);
  }

  template <std::size_t Order>
  [[nodiscard]] constexpr auto derivative_tensor(
      std::array<scalar_base_t<value_type>, input_dim> values) const noexcept
    requires(input_dim > 0 && Order > 0)
  {
    return equation_derivative_tensor_impl<Order>(std::move(values));
  }

};

template <CExpression T, CExpression... Ts>
Equation(T, Ts...) -> Equation<T, Ts...>;

} // namespace diff

#define reverse_mode_jac jacobian<diff::DiffMode::Reverse>
#define symbolic_mode_jac jacobian<diff::DiffMode::Symbolic>
