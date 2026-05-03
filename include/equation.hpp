#pragma once
#include "dual.hpp"
#include "equation.hpp"
#include "gradient.hpp"
#include <Eigen/Dense>
#include <boost/mp11/algorithm.hpp>

namespace diff {

namespace detail {
struct eval_func_t {
  constexpr auto operator()(const auto &...exprs) const {
    return std::array{exprs.eval()...};
  }
};
template <class Syms, class Updates> struct update_func_t {
  const Syms &syms;
  const Updates &updates;
  constexpr auto operator()(auto &...ds) const {
    (ds.update(syms, updates), ...);
  }
};

inline constexpr eval_func_t eval_func{};
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
constexpr auto make_derivatives(mp::mp_list<Syms...>, const Expr &expr) {
  return std::tuple(
      make_all_constant_except<Syms::value>(expr).derivative()...);
}

template <typename T>
concept CEquation = CExpression<T> and std::constructible_from<T>;
template <CExpression... Ts> class Equation;


// Build a std::tuple of Jacobian rows — one row (std::tuple of derivatives)
// per expression in the std::tuple es.
template <typename... Syms, typename... Exprs>
constexpr auto make_jac_rows(const std::tuple<Exprs...> &es,
                             mp::mp_list<Syms...> = {}) {
  return std::apply(
      [](const auto &...exprs) {
        return std::make_tuple(make_derivatives(mp::mp_list<Syms...>{}, exprs)...);
      },
      es);
}

// ===========================================================================
// Equation<TFirst, TRest...> — f: ℝⁿ → ℝᵐ  (sizeof...(TRest) > 0).
//
// Public API:
//   evaluate()                               — std::array of output values
//   jacobian<DiffMode::Symbolic>()           — symbolic (compile-time only)
//   jacobian<DiffMode::Reverse>([values])    — reverse-mode AD
//   jacobian<DiffMode::Forward>([values])    — forward-mode AD (Dual<T>)
//   hessian<DiffMode::Reverse>([values])     — forward-over-reverse (Dual<T>)
//   hessian<DiffMode::Forward>([values])     — forward-over-forward
//   (Dual<Dual<T>>) update(syms, values) / update(values)    — update variable
//   values
// ===========================================================================
template <CExpression TFirst, CExpression... TRest>
  requires((std::same_as<typename TFirst::value_type,
                         typename TRest::value_type> &&
            ...))
class Equation<TFirst, TRest...> {
public:
  using value_type = typename TFirst::value_type;
  using symbols = sort_tuple_t<
      tuple_union_t<typename extract_symbols_from_expr<TFirst>::type,
                    typename extract_symbols_from_expr<TRest>::type...>>;

  static constexpr std::size_t output_dim = 1 + sizeof...(TRest);
  static constexpr std::size_t input_dim = mp::mp_size<symbols>::value;
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

  // --- Symbolic Jacobian (compile-time only) ---
  [[nodiscard]] auto jacobian_symbolic() const
    requires(input_dim > 0)
  {
    Eigen::Matrix<value_type, output_dim, input_dim> J;
    static_for<output_dim>([&]<std::size_t I>() {
      auto row = std::apply(detail::eval_func, std::get<I>(jacobian_data));
      for (std::size_t j = 0; j < input_dim; ++j) {
        J(I, j) = row[j];
      }
    });
    return J;
  }

  // --- Reverse-mode Jacobian (compile-time path) ---
  [[nodiscard]] auto jacobian_reverse_mode() const
    requires(input_dim > 0)
  {
    Eigen::Matrix<value_type, output_dim, input_dim> J;
    static_for<output_dim>([&]<std::size_t I>() {
      std::array<value_type, input_dim> row{};
      std::get<I>(expressions).backward(symbols{}, value_type{1}, row);
      for (std::size_t j = 0; j < input_dim; ++j) {
        J(I, j) = row[j];
      }
    });
    return J;
  }

// --- Forward-mode Jacobian ---
  [[nodiscard]] auto jacobian_forward_mode()
    requires is_dual_v<value_type> && (input_dim > 0)
  {
    using S = dual_scalar_t<value_type>;
    Eigen::Matrix<S, output_dim, input_dim> J;
    Eigen::Vector<value_type, input_dim> seeds;

    std::array<value_type, input_dim> current{};
    std::apply(
        [&](const auto &...exprs) { (exprs.collect(symbols{}, current), ...); },
        expressions);

    for (std::size_t j = 0; j < input_dim; ++j){
      for (std::size_t i = 0; i < input_dim; ++i) {
        seeds[i] =
            value_type{current[i].template get<0>(), i == j ? S{1} : S{}};
      }
      update(symbols{}, seeds);
      auto vals = evaluate();
      for (std::size_t i = 0; i < output_dim; ++i)
        J(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
            vals[i].template get<1>();
    }

    // Restore to real-part values with zero dual tangent.
    for (std::size_t i = 0; i < input_dim; ++i) {
      seeds[i] = value_type{current[i].template get<0>(), S{}};
    }
    update(symbols{}, seeds);
    return J;
  }

  // --- Forward-over-reverse Hessian ---
  [[nodiscard]] auto hessian_forward_over_reverse()
    requires is_dual_v<value_type> && (input_dim > 0)
  {
    using S = dual_scalar_t<value_type>;
    std::array<Eigen::Matrix<S, input_dim, input_dim>, output_dim> H;
    for (auto &h : H) {
      h.setZero();
    }

    std::array<value_type, input_dim> current{};
    std::apply(
        [&](const auto &...exprs) { (exprs.collect(symbols{}, current), ...); },
        expressions);

    Eigen::Vector<value_type, input_dim> seeds;
    for (std::size_t j = 0; j < input_dim; ++j){
      for (std::size_t i= 0; i < input_dim; ++i) {
        seeds[i] =
            value_type{current[i].template get<0>(), i == j ? S{1} : S{}};
      }
      update(symbols{}, seeds);
      static_for<output_dim>([&]<std::size_t K>() {
        std::array<value_type, input_dim> grads{};
        std::get<K>(expressions).backward(symbols{}, value_type{1}, grads);
        for (auto i : std::views::iota(std::size_t{0}, input_dim)) {
          H[K](static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
              grads[i].template get<1>();
        }
      });
    }

    // Restore to real-part values with zero dual tangent.
    for (auto i : std::views::iota(std::size_t{0}, input_dim)) {
      seeds[i] = value_type{current[i].template get<0>(), S{}};
    }
    update(symbols{}, seeds);

    return H;
  }

  // --- Forward-over-forward Hessian ---
  [[nodiscard]] auto hessian_forward_over_forward()
    requires is_dual_v<value_type> && is_dual_v<dual_scalar_t<value_type>> &&
             (input_dim > 0)
  {
    using D = dual_scalar_t<value_type>;
    using S = dual_scalar_t<D>;

    std::array<Eigen::Matrix<S, input_dim, input_dim>, output_dim> H;
    std::array<value_type, input_dim> current{};
    std::apply(
        [&](const auto &...exprs) { (exprs.collect(symbols{}, current), ...); },
        expressions);

    Eigen::Vector<value_type, input_dim> seeds;
    for (std::size_t i = 0; i < input_dim; ++i) {
      for (std::size_t j = 0; j < input_dim; ++j) {
        for (std::size_t k = 0; k < input_dim; ++k) {
          S real_k = current[k].template get<0>().template get<0>();
          seeds[k] = value_type{D{real_k, k == j ? S{1} : S{}},
                                D{k == i ? S{1} : S{}, S{}}};
        }
        update(symbols{}, seeds);
        auto vals = evaluate();
        static_for<output_dim>([&]<std::size_t K>() {
          H[K](static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
              vals[K].template get<1>().template get<1>();
        });
      }
    }

    // Restore to real-part values.
    for (std::size_t k = 0; k < input_dim; ++k) {
      S real_k = current[k].template get<0>().template get<0>();
      seeds[k] = value_type{D{real_k, S{}}, D{}};
    }
    update(symbols{}, seeds);

    return H;
  }

public:
  constexpr Equation(TFirst first, TRest... rest)
      : expressions{first, rest...},
        jacobian_data{make_jac_rows(expressions, symbols{})} {}

  // Evaluate all component expressions.
  [[nodiscard]] constexpr auto evaluate() const {
    if constexpr (output_dim == 1)
      return std::get<0>(expressions).eval();
    else
      return std::apply(detail::eval_func, expressions);
  }

  constexpr operator value_type() const
    requires(output_dim == 1)
  {
    return std::get<0>(expressions).eval();
  }

  [[nodiscard]] constexpr auto eval_derivatives() const
    requires(output_dim == 1)
  {
    const auto &row = std::get<0>(jacobian_data);
    std::array<value_type, input_dim> result{};
    static_for<input_dim>([&]<std::size_t I>() {
      result[I] = std::get<I>(row).eval();
    });
    return result;
  }

  template <std::size_t N>
  constexpr decltype(auto) operator[](std::integral_constant<std::size_t, N>)
    requires(output_dim == 1)
  {
    if constexpr (N == 0)
      return std::get<0>(expressions);
    else
      return std::get<N - 1>(std::get<0>(jacobian_data));
  }

  template <std::size_t N>
  constexpr decltype(auto)
  operator[](std::integral_constant<std::size_t, N>) const
    requires(output_dim == 1)
  {
    if constexpr (N == 0)
      return std::get<0>(expressions);
    else
      return std::get<N - 1>(std::get<0>(jacobian_data));
  }

  // --- jacobian<Mode>() ---

  template <DiffMode Mode>
  [[nodiscard]] auto jacobian() const
    requires(Mode == DiffMode::Symbolic && input_dim > 0)
  {
    return jacobian_symbolic();
  }

  template <DiffMode Mode>
  [[nodiscard]] auto jacobian() const
    requires(Mode == DiffMode::Reverse && input_dim > 0)
  {
    return jacobian_reverse_mode();
  }

  template <DiffMode Mode>
  [[nodiscard]] auto jacobian(Eigen::Vector<value_type, input_dim> values)
    requires(Mode == DiffMode::Reverse && input_dim > 0)
  {
    update(symbols{}, values);
    return jacobian<Mode>();
  }

  template <DiffMode Mode>
  [[nodiscard]] auto jacobian()
    requires(Mode == DiffMode::Forward && is_dual_v<value_type> &&
             input_dim > 0)
  {
    return jacobian_forward_mode();
  }

  template <DiffMode Mode>
  [[nodiscard]] auto
  jacobian(Eigen::Vector<dual_scalar_t<value_type>, input_dim> values)
    requires(Mode == DiffMode::Forward && is_dual_v<value_type> &&
             input_dim > 0)
  {
    using S = dual_scalar_t<value_type>;
    Eigen::Vector<value_type, input_dim> seeds;
    for (std::size_t i = 0; i < input_dim; ++i) {
      seeds[i] = value_type{values[i], S{}};
    }
    update(symbols{}, seeds);
    return jacobian<Mode>();
  }

  // --- hessian<Mode>() ---

  template <DiffMode Mode>
  [[nodiscard]] auto hessian()
    requires(Mode == DiffMode::Reverse && is_dual_v<value_type> &&
             input_dim > 0)
  {
    return hessian_forward_over_reverse();
  }

  template <DiffMode Mode>
  [[nodiscard]] auto
  hessian(Eigen::Vector<dual_scalar_t<value_type>, input_dim> values)
    requires(Mode == DiffMode::Reverse && is_dual_v<value_type> &&
             input_dim > 0)
  {
    using S = dual_scalar_t<value_type>;
    Eigen::Vector<value_type, input_dim> seeds;
    for (std::size_t i = 0; i < input_dim; ++i) {
      seeds[i] = value_type{values[i], S{}};
    }
    update(symbols{}, seeds);
    return hessian<Mode>();
  }

  template <DiffMode Mode>
  [[nodiscard]] auto hessian()
    requires(Mode == DiffMode::Forward && is_dual_v<value_type> &&
             is_dual_v<dual_scalar_t<value_type>> && input_dim > 0)
  {
    return hessian_forward_over_forward();
  }

  template <DiffMode Mode>
  [[nodiscard]] auto hessian(
      Eigen::Vector<dual_scalar_t<dual_scalar_t<value_type>>, input_dim> values)
    requires(Mode == DiffMode::Forward && is_dual_v<value_type> &&
             is_dual_v<dual_scalar_t<value_type>> && input_dim > 0)
  {
    using D = dual_scalar_t<value_type>;
    using S = dual_scalar_t<D>;
    Eigen::Vector<value_type, input_dim> seeds;
    for (std::size_t i = 0; i < input_dim; ++i) {
      seeds[i] = value_type{D{values[i], S{}}, D{}};
    }
    update(symbols{}, seeds);
    return hessian<Mode>();
  }

  // --- nth_derivative<Order>() ---
  // Returns std::array<S, input_dim> of ∂^Order f / ∂x_i^Order per variable.
  // Requires value_type == nth_dual_t<S, Order> (build variables with that type).

  template <std::size_t Order>
  [[nodiscard]] constexpr auto
  nth_derivative(std::array<scalar_base_t<value_type>, input_dim> values) const
    requires(output_dim == 1 && dual_depth_v<value_type> == Order && Order > 0)
  {
    return diff::nth_derivative<Order>(std::get<0>(expressions), values);
  }

  template <std::size_t Order = dual_depth_v<value_type>>
  [[nodiscard]] constexpr auto nth_derivative() const
    requires(output_dim == 1 && is_dual_v<value_type>)
  {
    return diff::nth_derivative<Order>(std::get<0>(expressions));
  }

  // Update compile-time variables in all expressions and Jacobian rows.
  constexpr void update(const symbols &syms, const auto &updates) {
    auto update_func = detail::update_func_t{syms, updates};
    auto apply_to_tuple_func = [&](auto &...jac_rows) {
      (std::apply(update_func, jac_rows), ...);
    };
    std::apply(update_func, expressions);
    std::apply(apply_to_tuple_func, jacobian_data);
  }

};

template <CExpression T, CExpression... Ts>
Equation(T, Ts...) -> Equation<T, Ts...>;

} // namespace diff

auto make_equation(auto &&...args) {
  return diff::Equation(std::forward<decltype(args)>(args)...);
}

#define forward_mode_hess hessian<diff::DiffMode::Forward>
#define reverse_mode_hess hessian<diff::DiffMode::Reverse>
#define forward_mode_jac jacobian<diff::DiffMode::Forward>
#define reverse_mode_jac jacobian<diff::DiffMode::Reverse>
#define symbolic_mode_jac jacobian<diff::DiffMode::Symbolic>
