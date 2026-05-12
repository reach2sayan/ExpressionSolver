#pragma once
#include "dual.hpp"
#include "gradient.hpp"
#include <array>
#include <barrier>
#include <boost/mp11/algorithm.hpp>
#include <ranges>
#include <thread>

namespace diff {

namespace mp = boost::mp11;

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
        return std::make_tuple(
            make_derivatives(mp::mp_list<Syms...>{}, exprs)...);
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
//   hessian<DiffMode::Reverse>([values])     — forward-over-reverse (Dual<T>)
//   derivative_tensor<Order>([values])       — forward-mode, any order
//   update(syms, values)                     — update variable values
// ===========================================================================
template <CExpression TFirst, CExpression... TRest>
  requires(
      (std::same_as<typename TFirst::value_type, typename TRest::value_type> &&
       ...))
class Equation<TFirst, TRest...> {
public:
  using value_type = typename TFirst::value_type;
  using symbols = sort_tuple_t<
      tuple_union_t<extract_symbols_from_expr_t<TFirst>,
                    extract_symbols_from_expr_t<TRest>...>>;

  static constexpr std::size_t output_dim = 1 + sizeof...(TRest);
  static constexpr std::size_t input_dim = mp::mp_size<symbols>::value;
  static constexpr std::size_t number_of_derivatives = input_dim;

private:
  using Exprs = std::tuple<TFirst, TRest...>;
  using jacobian_t = decltype(make_jac_rows(std::declval<Exprs>(), symbols{}));
  Exprs expressions;
  jacobian_t jacobian_data;

  // Parallel reverse-mode infrastructure.
  // Main thread owns row 0; each worker[i] owns row i+1.
  // Heap-allocated so Equation remains movable for output_dim == 1
  // (null par_state_ → no workers, safe to move).
  // Do NOT move an Equation after construction when output_dim > 1:
  // worker lambdas capture `this` and would reference the old address.
  static constexpr std::size_t par_nworkers =
      output_dim > 1 ? output_dim - 1 : 0;

  struct ParState {
    std::barrier<> start;
    std::barrier<> end;
    std::array<std::array<value_type, input_dim>, output_dim> results{};
    std::array<std::jthread, par_nworkers> workers;
    explicit ParState(std::ptrdiff_t n) : start{n}, end{n} {}
  };

  mutable std::unique_ptr<ParState> par_state_{nullptr};

  void spawn_parallel_workers() {
    if constexpr (output_dim > 1) {
      par_state_ =
          std::make_unique<ParState>(static_cast<std::ptrdiff_t>(output_dim));
      [this]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((par_state_->workers[Is] = std::jthread([this](std::stop_token st) {
            while (true) {
              par_state_->start.arrive_and_wait();
              if (st.stop_requested())
                return;
              std::get<Is + 1>(expressions)
                  .backward(symbols{}, value_type{1},
                            par_state_->results[Is + 1]);
              par_state_->end.arrive_and_wait();
            }
          })),
         ...);
      }(std::make_index_sequence<par_nworkers>{});
    }
  }

  friend std::ostream &operator<<(std::ostream &out, const Equation &ve) {
    static_for<output_dim>([&]<std::size_t I>() {
      out << "f" << I << ": " << std::get<I>(ve.expressions);
      out << " grad: ";
      print_tup(out, std::get<I>(ve.jacobian_data));
      out << '\n';
    });
    return out;
  }

  // --- Symbolic Jacobian ---
  [[nodiscard]] auto jacobian_symbolic() const
    requires(input_dim > 0)
  {
    std::array<std::array<value_type, input_dim>, output_dim> J{};
    static_for<output_dim>([&]<std::size_t I>() {
      J[I] = std::apply(detail::eval_func, std::get<I>(jacobian_data));
    });
    return J;
  }

  // --- Reverse-mode Jacobian ---
  [[nodiscard]] auto jacobian_reverse_mode() const
    requires(input_dim > 0)
  {
    std::array<std::array<value_type, input_dim>, output_dim> J{};
    static_for<output_dim>([&]<std::size_t I>() {
      std::get<I>(expressions).backward(symbols{}, value_type{1}, J[I]);
    });
    return J;
  }

  // --- Parallel reverse-mode Jacobian (barrier rendezvous) ---
  // Workers pre-spawned in constructor handle rows 1..output_dim-1;
  // main thread handles row 0.  No dynamic allocation in the hot path —
  // just two barrier phase transitions per call.
  [[nodiscard]] auto jacobian_reverse_parallel() const
    requires(input_dim > 0)
  {
    if constexpr (output_dim == 1) {
      return jacobian_reverse_mode();
    } else {
      par_state_->start.arrive_and_wait();
      std::get<0>(expressions)
          .backward(symbols{}, value_type{1}, par_state_->results[0]);
      par_state_->end.arrive_and_wait();
      return par_state_->results;
    }
  }

  // --- Forward-over-reverse Hessian (Dual<T> expressions) ---
  [[nodiscard]] auto hessian_forward_over_reverse()
    requires(is_dual_v<value_type> && input_dim > 0)
  {
    using S = dual_scalar_t<value_type>;
    std::array<std::array<std::array<S, input_dim>, input_dim>, output_dim> H{};

    std::array<value_type, input_dim> current{};
    std::apply(
        [&](const auto &...exprs) { (exprs.collect(symbols{}, current), ...); },
        expressions);

    std::array<value_type, input_dim> seeds{};
    for (std::size_t j = 0; j < input_dim; ++j) {
      for (std::size_t i = 0; i < input_dim; ++i) {
        seeds[i] =
            value_type{current[i].template get<0>(), i == j ? S{1} : S{}};
      }
      update(symbols{}, seeds);
      static_for<output_dim>([&]<std::size_t K>() {
        std::array<value_type, input_dim> grads{};
        std::get<K>(expressions).backward(symbols{}, value_type{1}, grads);
        for (std::size_t i = 0; i < input_dim; ++i) {
          H[K][i][j] = grads[i].template get<1>();
        }
      });
    }
    for (std::size_t i = 0; i < input_dim; ++i) {
      seeds[i] = value_type{current[i].template get<0>(), S{}};
    }
    update(symbols{}, seeds);
    return H;
  }

  // --- Forward-mode derivative tensor (any order) ---
  // Returns std::array<nd_array_t<S, input_dim, Order>, output_dim>.
  // result[out][i₁]...[iOrder] = ∂^Order f_out / ∂x_{i₁}...∂x_{iOrder}
  template <std::size_t Order>
  [[nodiscard]] auto equation_derivative_tensor_impl(
      std::array<scalar_base_t<value_type>, input_dim> values) const
    requires(input_dim > 0 && Order > 0)
  {
    using S = scalar_base_t<value_type>;
    using U = nth_dual_t<S, Order>;

    std::array<nd_array_t<S, input_dim, Order>, output_dim> result{};

    std::size_t total = 1;
    for (std::size_t d = 0; d < Order; ++d)
      total *= input_dim;

    for (std::size_t flat = 0; flat < total; ++flat) {
      std::array<std::size_t, Order> idx{};
      std::size_t tmp = flat;
      for (int d = (int)Order - 1; d >= 0; --d) {
        idx[d] = tmp % input_dim;
        tmp /= input_dim;
      }

      std::array<U, input_dim> seeds{};
      for (std::size_t k = 0; k < input_dim; ++k)
        seeds[k] = detail::make_mixed_seed<S, Order>(values[k], idx.data(), k);

      static_for<output_dim>([&]<std::size_t OUT>() {
        U val = std::get<OUT>(expressions)
                    .template eval_seeded_as<U, symbols>(seeds);
        nd_index<Order>(result[OUT], idx.data()) =
            detail::extract_nth<Order>(val);
      });
    }
    return result;
  }

public:
  Equation(TFirst first, TRest... rest)
      : expressions{first, rest...},
        jacobian_data{make_jac_rows(expressions, symbols{})} {
    spawn_parallel_workers();
  }

  // Move is only safe when there are no pinned workers (output_dim == 1).
  // For output_dim > 1, worker lambdas capture `this`; moving would dangle.
  Equation(Equation &&o) noexcept
    requires(par_nworkers == 0)
      : expressions{std::move(o.expressions)},
        jacobian_data{std::move(o.jacobian_data)},
        par_state_{std::move(o.par_state_)} {}

  Equation(const Equation &) = delete;
  Equation &operator=(const Equation &) = delete;
  Equation &operator=(Equation &&) = delete;

  ~Equation() {
    if constexpr (output_dim > 1) {
      for (auto &w : par_state_->workers)
        w.request_stop();
      par_state_->start
          .arrive_and_wait(); // wake workers so they see stop and exit
    }
  }

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
    static_for<input_dim>(
        [&]<std::size_t I>() { result[I] = std::get<I>(row).eval(); });
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
  [[nodiscard]] auto jacobian(std::array<value_type, input_dim> values)
    requires(Mode == DiffMode::Reverse && input_dim > 0)
  {
    update(symbols{}, values);
    return jacobian<Mode>();
  }

  template <DiffMode Mode>
  [[nodiscard]] auto jacobian() const
    requires(Mode == DiffMode::ParallelReverse && input_dim > 0)
  {
    return jacobian_reverse_parallel();
  }

  template <DiffMode Mode>
  [[nodiscard]] auto jacobian(std::array<value_type, input_dim> values)
    requires(Mode == DiffMode::ParallelReverse && input_dim > 0)
  {
    update(symbols{}, values);
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
  hessian(std::array<dual_scalar_t<value_type>, input_dim> values)
    requires(Mode == DiffMode::Reverse && is_dual_v<value_type> &&
             input_dim > 0)
  {
    using S = dual_scalar_t<value_type>;
    std::array<value_type, input_dim> seeds{};
    for (std::size_t i = 0; i < input_dim; ++i)
      seeds[i] = value_type{values[i], S{}};
    update(symbols{}, seeds);
    return hessian<Mode>();
  }

  // --- derivative_tensor<Order>() — forward-mode, any order ---
  // Input is always plain scalar S = scalar_base_t<value_type>.
  // Returns std::array<nd_array_t<S, input_dim, Order>, output_dim>.

  template <std::size_t Order>
  [[nodiscard]] auto derivative_tensor(
      std::array<scalar_base_t<value_type>, input_dim> values) const
    requires(input_dim > 0 && Order > 0)
  {
    return equation_derivative_tensor_impl<Order>(values);
  }

  template <std::size_t Order>
  [[nodiscard]] auto derivative_tensor() const
    requires(input_dim > 0 && Order > 0)
  {
    using S = scalar_base_t<value_type>;
    std::array<value_type, input_dim> current{};
    std::apply(
        [&](const auto &...exprs) { (exprs.collect(symbols{}, current), ...); },
        expressions);
    std::array<S, input_dim> values{};
    for (std::size_t i = 0; i < input_dim; ++i)
      values[i] = get_real_part<dual_depth_v<value_type>>(current[i]);
    return equation_derivative_tensor_impl<Order>(values);
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

#define reverse_mode_hess hessian<diff::DiffMode::Reverse>
#define reverse_mode_jac jacobian<diff::DiffMode::Reverse>
#define symbolic_mode_jac jacobian<diff::DiffMode::Symbolic>
#define parallel_reverse_jac jacobian<diff::DiffMode::ParallelReverse>
