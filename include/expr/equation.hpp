#pragma once
#include "drivers/gradient.hpp"
#include "dual/dual.hpp"
#include "expr/format.hpp"
#include "util/config.hpp"
#include "util/mpl.hpp"
#include "util/scope_guard.hpp"
#include <algorithm>
#include <array>
#include <ranges>
#include <utility>

namespace diff {

namespace mp = diff::mpl;

namespace detail {
// Evaluate a tuple of expressions at one point, in canonical symbol order.
template <CSymbolList Syms, CNumericBuffer Vals, CExpression... Es>
constexpr auto eval_all(const Vals &vals, const Es &...es) noexcept {
  return std::array{es.template eval_seeded<Syms>(vals)...};
}
} // namespace detail

template <CSymbol... Syms, CExpression Expr>
constexpr auto make_derivatives(mp::mp_list<Syms...>,
                                const Expr &expr) noexcept {
  return std::tuple(
      make_all_constant_except<Syms::value>(expr).derivative()...);
}

template <CExpression... Ts> class Equation;

template <CSymbol... Syms, CExpression... Exprs>
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

  using point_t = std::array<value_type, input_dim>;

  [[nodiscard]] constexpr auto
  jacobian_symbolic(const point_t &vals) const noexcept
    requires(input_dim > 0)
  {
    md_tensor<value_type, md::extents<std::size_t, output_dim, input_dim>> J{};
    static_for<output_dim>([&]<std::size_t I>() {
      assign_row(J, I,
                 std::apply(
                     [&](const auto &...ds) {
                       return detail::eval_all<symbols>(vals, ds...);
                     },
                     std::get<I>(jacobian_data)));
    });
    return J;
  }

  [[nodiscard]] constexpr auto
  jacobian_reverse_mode(const point_t &vals) const noexcept
    requires(input_dim > 0)
  {
    md_tensor<value_type, md::extents<std::size_t, output_dim, input_dim>> J{};
    static_for<output_dim>([&]<std::size_t I>() {
      // reverse_sweep fills a plain array (backward() writes through a
      // CNumericBuffer); the row lands in the tensor afterwards.
      std::array<value_type, input_dim> row{};
      reverse_sweep<symbols>(std::get<I>(expressions), vals, row);
      assign_row(J, I, row);
    });
    return J;
  }

  [[nodiscard]] constexpr auto hessian_forward_over_reverse(
      const std::array<dual_scalar_t<value_type>, input_dim> &values)
      const noexcept
    requires(DualLike<value_type> && input_dim > 0)
  {
    using S = dual_scalar_t<value_type>;
    nd_stack_t<S, output_dim, input_dim, 2> H{};

    // Values are the point and never move; only the seeded tangent does.
    point_t seeds{};
    std::ranges::transform(values, seeds.begin(),
                           [](const S &v) { return value_type{v, S{}}; });

    for (std::size_t j = 0; j < input_dim; ++j) {
      // Seed column j; one reverse sweep per output yields that column.
      const auto seed = scoped_seed<1>(seeds[j].deriv());
      static_for<output_dim>([&]<std::size_t K>() {
        point_t grads{};
        reverse_sweep<symbols>(std::get<K>(expressions), seeds, grads);

        const auto column =
            grads | std::views::transform([](const value_type &g) {
              return g.template get<1>();
            });
        for (auto &&[i, entry] :
             std::views::zip(std::views::iota(0uz, input_dim), column)) {
          H[K, i, j] = entry;
        }
      });
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
    nd_stack_t<S, output_dim, input_dim, Order> result{};

    for (const auto &idx : detail::symmetric_index_grid<input_dim, Order>()) {

      const auto seeds = detail::mixed_seeds<S, Order>(values, idx);

      static_for<output_dim>([&]<std::size_t OUT>() {
        U val = std::get<OUT>(expressions).template eval_seeded<symbols>(seeds);
        // The output axis leads, so the stacked index is OUT followed by the
        // multi-index.
        const auto stacked = [&]<std::size_t... K>(std::index_sequence<K...>) {
          return std::array<std::size_t, Order + 1>{OUT, idx[K]...};
        }(std::make_index_sequence<Order>{});
        result.at_index(stacked) = detail::extract_nth<Order>(val);
      });
    }
    return result;
  }

  template <std::size_t N>
  [[nodiscard]] static constexpr decltype(auto) slot(auto &&self) noexcept {
    if constexpr (N == 0) {
      return std::get<0>(std::forward<decltype(self)>(self).expressions);
    } else {
      return std::get<N - 1>(
          std::get<0>(std::forward<decltype(self)>(self).jacobian_data));
    }
  }

public:
  constexpr Equation(TFirst first, TRest... rest) noexcept
      : expressions{std::move(first), std::move(rest)...},
        jacobian_data{make_jac_rows(expressions, symbols{})} {}

  // The symbolic trees themselves.  Every other member reduces them to numbers
  // at a point; these hand them over intact, which is what the formatter below
  // needs and what makes the built Jacobian inspectable at all.
  [[nodiscard]] constexpr const Exprs &functions() const noexcept {
    return expressions;
  }
  [[nodiscard]] constexpr const jacobian_t &jacobian_rows() const noexcept {
    return jacobian_data;
  }

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
  // Subscript spelling of the same thing; the index arrives as an empty idx_t
  // value (see idx<N>()).  See DIFF_KEYED_ACCESSORS in util/config.hpp.
  DIFF_KEYED_ACCESSORS(std::size_t N, std::size_t N, N, idx_t<N>,
                       requires(output_dim == 1 && N <= input_dim))

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

template <CExpression... Ts>
std::ostream &operator<<(std::ostream &out, const Equation<Ts...> &eq) {
  return out << std::format("{}", eq);
}

} // namespace diff

// One block per output function: the function itself, then its gradient row in
// canonical symbol order.  The spec is whatever expr/format.hpp accepts and is
// forwarded to every expression printed, so "{:d}" annotates the leaves of a
// whole system and "{:t}" dumps each one as a tree.
template <diff::CExpression... Ts>
struct std::formatter<diff::Equation<Ts...>, char> {
  constexpr auto parse(std::format_parse_context &ctx) {
    spec_ = {ctx.begin(), static_cast<std::size_t>(ctx.end() - ctx.begin())};
    if (const auto close = spec_.find('}'); close != std::string_view::npos) {
      spec_ = spec_.substr(0, close);
    }
    return ctx.begin() + spec_.size();
  }

  auto format(const diff::Equation<Ts...> &eq, std::format_context &ctx) const {
    using Eq = diff::Equation<Ts...>;
    const std::string one = std::format("{{:{}}}", spec_);
    auto out = ctx.out();

    diff::static_for<Eq::output_dim>([&]<std::size_t I>() {
      out = std::format_to(out, "f{}: ", I);
      out = std::vformat_to(out, one,
                            std::make_format_args(std::get<I>(eq.functions())));
      out = std::format_to(out, "\n  grad: ");
      const auto &row = std::get<I>(eq.jacobian_rows());
      diff::static_for<Eq::input_dim>([&]<std::size_t J>() {
        if constexpr (J > 0) {
          out = std::format_to(out, ", ");
        }
        out = std::vformat_to(out, one, std::make_format_args(std::get<J>(row)));
      });
      out = std::format_to(out, "\n");
    });
    return out;
  }

private:
  std::string_view spec_{};
};

#define reverse_mode_jac jacobian<diff::DiffMode::Reverse>
#define symbolic_mode_jac jacobian<diff::DiffMode::Symbolic>
