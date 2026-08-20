#pragma once
#include "drivers/symbolic.hpp"
#include "dual/dual.hpp"
#include "expr/bound.hpp"
#include "expr/format.hpp"
#include "expr/simplify.hpp"
#include "util/config.hpp"
#include "util/mpl.hpp"
#include "util/scope_guard.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>
#include <string_view>
#include <utility>

namespace ddx::impl {

namespace mp = ddx::impl::mpl;

namespace detail {
// Evaluate a tuple of expressions at one point, in canonical symbol order.
template <CSymbolList Syms, CNumericBuffer Vals, CExpression... Es>
constexpr auto eval_all(const Vals &vals, const Es &...es) noexcept {
  return std::array{es.template eval_seeded<Syms>(vals)...};
}
} // namespace detail

namespace detail {

// The expressions are final by the time an Equation is built, so this is where
// commutative operands get ordered; folding already happened as they were built.
template <CExpression E>
using canonical_t = decltype(canonicalise(std::declval<const E &>()));

template <CSymbol... Syms, CExpression Expr>
constexpr auto make_derivatives(mp::mp_list<Syms...>,
                                const Expr &expr) noexcept {
  return std::tuple(
      canonicalise(make_all_constant_except<Syms::value>(expr).derivative())...);
}

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

} // namespace detail

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
  // symbols and input_dim stay derived from what the user wrote, so a partial
  // that folds away a symbol cannot shrink the point.
  using Exprs =
      std::tuple<detail::canonical_t<TFirst>, detail::canonical_t<TRest>...>;
  Exprs expressions;

  using point_t = std::array<value_type, input_dim>;

  [[nodiscard]] constexpr auto
  jacobian_symbolic(const point_t &vals) const noexcept
    requires(input_dim > 0)
  {
    md_tensor<value_type, md::extents<std::size_t, output_dim, input_dim>> J{};
    const auto rows = jacobian_rows();
    static_for<output_dim>([&]<std::size_t I>() {
      assign_row(J, I,
                 std::apply(
                     [&](const auto &...ds) {
                       return detail::eval_all<symbols>(vals, ds...);
                     },
                     std::get<I>(rows)));
    });
    return J;
  }

  [[nodiscard]] constexpr auto
  jacobian_reverse_mode(const point_t &vals) const noexcept
    requires(input_dim > 0)
  {
    md_tensor<value_type, md::extents<std::size_t, output_dim, input_dim>> J{};
    static_for<output_dim>([&]<std::size_t I>() {
      // reverse_sweep fills a plain array; the row lands in the tensor after.
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

    // Only the seeded tangent moves.
    point_t seeds{};
    std::ranges::transform(values, seeds.begin(),
                           [](const S &v) { return value_type{v, S{}}; });

    for (std::size_t j = 0; j < input_dim; ++j) {
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
      // Built on demand, so this returns a value, not a dangling reference.
      auto row = std::get<0>(self.jacobian_rows());
      std::tuple_element_t<N - 1, decltype(row)> d = std::get<N - 1>(row);
      return d;
    }
  }

public:
  // explicit, or `Equation<E> eq = expr;` is ambiguous with
  // EquationConvertible's conversion operator.
  constexpr explicit Equation(TFirst first, TRest... rest) noexcept
      : expressions{detail::canonicalise(first),
                    detail::canonicalise(rest)...} {}

  // The symbolic trees; every other member reduces them to numbers at a point.
  [[nodiscard]] constexpr const Exprs &functions() const noexcept {
    return expressions;
  }
  // Built on demand: storing it would instantiate the whole symbolic Jacobian
  // every time Equation<E> is named, which the reverse-mode members never touch.
  [[nodiscard]] constexpr auto jacobian_rows() const noexcept {
    return detail::make_jac_rows(expressions, symbols{});
  }

  // Every numeric member takes a point in any spelling eval() accepts.
  template <Numeric U = value_type>
  [[nodiscard]] static constexpr auto point(const CEvalArg auto &...args) noexcept {
    return detail::make_point<symbols, U, input_dim>(args...);
  }

  [[nodiscard]] constexpr auto evaluate(const CEvalArg auto &...args) const noexcept {
    const auto vals = point(args...);
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

  // Symbolic evaluates the stored partial trees; Reverse never builds them.
  template <DiffMode Mode = DiffMode::Reverse>
  [[nodiscard]] constexpr auto gradient(const CEvalArg auto &...args) const noexcept
    requires(output_dim == 1 && input_dim > 0)
  {
    const auto vals = point(args...);
    if constexpr (Mode == DiffMode::Symbolic) {
      const auto rows = jacobian_rows();
      const auto &row = std::get<0>(rows);
      std::array<value_type, input_dim> grads{};
      static_for<input_dim>([&]<std::size_t I>() {
        grads[I] = std::get<I>(row).template eval_seeded<symbols>(vals);
      });
      return detail::strip_seed(grads);
    } else {
      return detail::reverse_mode_gradient(std::get<0>(expressions), vals);
    }
  }

  // Slot 0 is the expression itself; slot k>0 is d/d(k-1 th symbol), in
  // canonical symbol order.  Both spellings; see DDX_KEYED_ACCESSORS.
  DDX_KEYED_ACCESSORS(std::size_t N, std::size_t N, N, idx_t<N>,
                       requires(output_dim == 1 && N <= input_dim))

  template <DiffMode Mode = DiffMode::Reverse>
  [[nodiscard]] constexpr auto jacobian(const CEvalArg auto &...args) const noexcept
    requires(input_dim > 0)
  {
    if constexpr (Mode == DiffMode::Symbolic) {
      return jacobian_symbolic(point(args...));
    } else {
      return jacobian_reverse_mode(point(args...));
    }
  }

  // The leading output axis only appears with more than one output, here and in
  // derivative_tensor below.
  template <DiffMode Mode = DiffMode::Reverse>
  [[nodiscard]] constexpr auto hessian(const CEvalArg auto &...args) const noexcept
    requires(Mode == DiffMode::Reverse && DualLike<value_type> && input_dim > 0)
  {
    const auto vals = point<dual_scalar_t<value_type>>(args...);
    if constexpr (output_dim == 1) {
      return detail::reverse_mode_hessian(std::get<0>(expressions), vals);
    } else {
      return hessian_forward_over_reverse(vals);
    }
  }

  template <std::size_t Order>
  [[nodiscard]] constexpr auto
  derivative_tensor(const CEvalArg auto &...args) const noexcept
    requires(input_dim > 0 && Order > 0)
  {
    const auto vals = point<scalar_base_t<value_type>>(args...);
    if constexpr (output_dim == 1) {
      return detail::derivative_tensor_impl<Order>(std::get<0>(expressions),
                                                   vals);
    } else {
      return equation_derivative_tensor_impl<Order>(vals);
    }
  }

  // One variable, one Taylor sweep: a plain number, not a one-entry tensor.
  template <std::size_t Order>
  [[nodiscard]] DDX_ALWAYS_INLINE constexpr auto
  univariate_derivative(scalar_base_t<value_type> x0) const noexcept
    requires(input_dim == 1 && output_dim == 1 && Order > 0)
  {
    return detail::univariate_derivative_impl<Order>(std::get<0>(expressions),
                                                     x0);
  }
};

template <CExpression T, CExpression... Ts>
Equation(T, Ts...) -> Equation<T, Ts...>;

// Declared back in expr/expressions.hpp, where Equation was still incomplete.
template <typename Derived>
template <typename Eq>
  requires std::same_as<Eq, Equation<Derived>>
constexpr EquationConvertible<Derived>::operator Eq() const noexcept {
  return Equation<Derived>{static_cast<const Derived &>(*this)};
}

template <CExpression... Ts>
std::ostream &operator<<(std::ostream &out, const Equation<Ts...> &eq) {
  return out << std::format("{}", eq);
}

} // namespace ddx::impl

// One block per output: the function, then its gradient row in canonical
// symbol order.  The spec is forwarded to every expression printed.
template <ddx::impl::CExpression... Ts>
struct std::formatter<ddx::impl::Equation<Ts...>, char> {
  constexpr auto parse(std::format_parse_context &ctx) {
    // (iterator, sentinel): the iterator is const char* on libstdc++ but a
    // class type on MSVC.
    spec_ = std::string_view(ctx.begin(), ctx.end());
    if (const auto close = spec_.find('}'); close != std::string_view::npos) {
      spec_ = spec_.substr(0, close);
    }
    return ctx.begin() + static_cast<std::ptrdiff_t>(spec_.size());
  }

  auto format(const ddx::impl::Equation<Ts...> &eq, std::format_context &ctx) const {
    using Eq = ddx::impl::Equation<Ts...>;
    const std::string one = std::format("{{:{}}}", spec_);
    auto out = ctx.out();

    const auto rows = eq.jacobian_rows();
    ddx::impl::static_for<Eq::output_dim>([&]<std::size_t I>() {
      out = std::format_to(out, "f{}: ", I);
      out = std::vformat_to(out, one,
                            std::make_format_args(std::get<I>(eq.functions())));
      out = std::format_to(out, "\n  grad: ");
      const auto &row = std::get<I>(rows);
      ddx::impl::static_for<Eq::input_dim>([&]<std::size_t J>() {
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


