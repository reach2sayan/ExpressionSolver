#pragma once
#include "dual.hpp"
#include "expressions.hpp"
#include "operations.hpp"
#include <boost/mp11/algorithm.hpp>
#include <concepts>
#include <format>
#include <string_view>

namespace diff {

constexpr bool PRINT_VARIABLE_VALUE = false;
constexpr bool PRINT_VARIABLE_LABEL = true;
constexpr bool PRINT_CONSTANT_VALUE = true;
constexpr bool PRINT_CONSTANT_LABEL = false;

template <typename LHS, typename RHS>
concept CompatibleValueTypes =
    std::is_same_v<typename LHS::value_type, typename RHS::value_type> ||
    std::is_convertible_v<typename LHS::value_type, typename RHS::value_type> ||
    std::is_convertible_v<typename RHS::value_type, typename LHS::value_type>;

template <char C, typename SymList> consteval std::size_t find_index_of_char() {
  return boost::mp11::mp_find<SymList, std::integral_constant<char, C>>::value;
}

struct IOperators {

  template <ExpressionConcept LHS, ExpressionConcept RHS>
    requires CompatibleValueTypes<LHS, RHS>
  friend constexpr auto operator+(const LHS &a, const RHS &b) {
    using value_type = typename LHS::value_type;
    return Expression<SumOp<value_type>, LHS, RHS>{std::move(a), std::move(b)};
  }

  template <ExpressionConcept LHS, ExpressionConcept RHS>
    requires CompatibleValueTypes<LHS, RHS>
  friend constexpr auto operator*(const LHS &a, const RHS &b) {
    using value_type = typename LHS::value_type;
    return Expression<MultiplyOp<value_type>, LHS, RHS>(std::move(a),
                                                        std::move(b));
  }

  template <ExpressionConcept LHS, ExpressionConcept RHS>
    requires CompatibleValueTypes<LHS, RHS>
  friend constexpr auto operator-(const LHS &a, const RHS &b) {
    using value_type = typename LHS::value_type;
    auto neg = MonoExpression<NegateOp<value_type>, RHS>(std::move(b));
    return Expression<SumOp<value_type>, LHS, decltype(neg)>{std::move(a),
                                                             std::move(neg)};
  }

  template <ExpressionConcept LHS, ExpressionConcept RHS>
    requires CompatibleValueTypes<LHS, RHS>
  friend constexpr auto operator/(const LHS &a, const RHS &b) {
    using value_type = typename LHS::value_type;
    return Expression<DivideOp<value_type>, LHS, RHS>{std::move(a),
                                                      std::move(b)};
  }

  template <ExpressionConcept Expr> friend constexpr auto sin(const Expr &a) {
    using value_type = typename Expr::value_type;
    return MonoExpression<SineOp<value_type>, Expr>{std::move(a)};
  }

  template <ExpressionConcept Expr> friend constexpr auto cos(const Expr &a) {
    using value_type = typename Expr::value_type;
    return MonoExpression<CosineOp<value_type>, Expr>{std::move(a)};
  }

  template <ExpressionConcept Expr> friend constexpr auto exp(const Expr &a) {
    using value_type = typename Expr::value_type;
    return MonoExpression<ExpOp<value_type>, Expr>{std::move(a)};
  }
  template <ExpressionConcept Expr> friend constexpr auto tan(const Expr &a) {
    using value_type = typename Expr::value_type;
    return MonoExpression<TanOp<value_type>, Expr>{std::move(a)};
  }
  template <ExpressionConcept Expr> friend constexpr auto log(const Expr &a) {
    using value_type = typename Expr::value_type;
    return MonoExpression<LogOp<value_type>, Expr>{std::move(a)};
  }
  template <ExpressionConcept Expr> friend constexpr auto sqrt(const Expr &a) {
    using value_type = typename Expr::value_type;
    return MonoExpression<SqrtOp<value_type>, Expr>{std::move(a)};
  }
  template <ExpressionConcept Expr> friend constexpr auto abs(const Expr &a) {
    using value_type = typename Expr::value_type;
    return MonoExpression<AbsOp<value_type>, Expr>{std::move(a)};
  }
  template <ExpressionConcept Expr> friend constexpr auto asin(const Expr &a) {
    using value_type = typename Expr::value_type;
    return MonoExpression<AsinOp<value_type>, Expr>{std::move(a)};
  }
  template <ExpressionConcept Expr> friend constexpr auto acos(const Expr &a) {
    using value_type = typename Expr::value_type;
    return MonoExpression<AcosOp<value_type>, Expr>{std::move(a)};
  }
  template <ExpressionConcept Expr> friend constexpr auto atan(const Expr &a) {
    using value_type = typename Expr::value_type;
    return MonoExpression<AtanOp<value_type>, Expr>{std::move(a)};
  }
  template <ExpressionConcept Expr> friend constexpr auto sinh(const Expr &a) {
    using value_type = typename Expr::value_type;
    return MonoExpression<SinhOp<value_type>, Expr>{std::move(a)};
  }
  template <ExpressionConcept Expr> friend constexpr auto cosh(const Expr &a) {
    using value_type = typename Expr::value_type;
    return MonoExpression<CoshOp<value_type>, Expr>{std::move(a)};
  }
  template <ExpressionConcept Expr> friend constexpr auto tanh(const Expr &a) {
    using value_type = typename Expr::value_type;
    return MonoExpression<TanhOp<value_type>, Expr>{std::move(a)};
  }

  template <typename S, ExpressionConcept RHS>
    requires std::is_arithmetic_v<S>
  friend constexpr auto operator+(S s, const RHS &b) {
    using VT = typename RHS::value_type;
    return Constant<VT>{static_cast<VT>(s)} + b;
  }
  template <typename S, ExpressionConcept RHS>
    requires std::is_arithmetic_v<S>
  friend constexpr auto operator*(S s, const RHS &b) {
    using VT = typename RHS::value_type;
    return Constant<VT>{static_cast<VT>(s)} * b;
  }
  template <typename S, ExpressionConcept RHS>
    requires std::is_arithmetic_v<S>
  friend constexpr auto operator-(S s, const RHS &b) {
    using VT = typename RHS::value_type;
    return Constant<VT>{static_cast<VT>(s)} - b;
  }
  template <typename S, ExpressionConcept RHS>
    requires std::is_arithmetic_v<S>
  friend constexpr auto operator/(S s, const RHS &b) {
    using VT = typename RHS::value_type;
    return Constant<VT>{static_cast<VT>(s)} / b;
  }

  template <ExpressionConcept LHS, typename S>
    requires std::is_arithmetic_v<S>
  friend constexpr auto operator+(const LHS &a, S s) {
    using VT = typename LHS::value_type;
    return a + Constant<VT>{static_cast<VT>(s)};
  }
  template <ExpressionConcept LHS, typename S>
    requires std::is_arithmetic_v<S>
  friend constexpr auto operator*(const LHS &a, S s) {
    using VT = typename LHS::value_type;
    return a * Constant<VT>{static_cast<VT>(s)};
  }
  template <ExpressionConcept LHS, typename S>
    requires std::is_arithmetic_v<S>
  friend constexpr auto operator-(const LHS &a, S s) {
    using VT = typename LHS::value_type;
    return a - Constant<VT>{static_cast<VT>(s)};
  }
  template <ExpressionConcept LHS, typename S>
    requires std::is_arithmetic_v<S>
  friend constexpr auto operator/(const LHS &a, S s) {
    using VT = typename LHS::value_type;
    return a / Constant<VT>{static_cast<VT>(s)};
  }
};

template <Numeric T> class Constant : public IOperators {
  const T value;
  friend std::ostream &operator<<(std::ostream &out, const Constant<T> &c) {
    if constexpr (PRINT_CONSTANT_VALUE)
      out << std::format("{}", c.value);
    if constexpr (PRINT_CONSTANT_LABEL)
      out << "_c";
    return out;
  }
  [[nodiscard]] constexpr auto eval() const { return value; }

public:
  using value_type = T;
  constexpr explicit Constant(T value) : value(value) {}
  [[nodiscard]] constexpr auto get() const { return value; }
  constexpr operator T() const { return value; }
  [[nodiscard]] constexpr auto derivative() const { return Constant{T{}}; }
  constexpr void update(const auto &, const auto &) const {}
  constexpr void collect(const auto &, auto &) const {}
  constexpr void backward(const auto &, T, auto &) const {}

  template <typename Syms, std::size_t N>
  [[nodiscard]] constexpr T eval_seeded(const std::array<T, N> &) const {
    return value;
  }

  template <std::size_t I> [[nodiscard]] constexpr auto get() const {
    static_assert(I < 2);
    if constexpr (requires { std::tuple_size<T>::value; }) {
      return eval().template get<I>();
    } else if constexpr (I == 0) {
      return eval();
    } else {
      return static_cast<T>(derivative());
    }
  }
};

template <Numeric T, char symbol> class Variable : public IOperators {
  T value;
  friend std::ostream &operator<<(std::ostream &out,
                                  const Variable<T, symbol> &c) {
    if constexpr (PRINT_VARIABLE_VALUE) {
      out << std::format("{}_", c.value);
    }
    if constexpr (PRINT_VARIABLE_LABEL) {
      out << symbol;
    }
    return out;
  }
  static constexpr inline size_t static_counter = 0;
  [[nodiscard]] constexpr T eval() const { return value; }

public:
  using value_type = T;
  constexpr explicit Variable(T value) : value(value) {}
  constexpr operator T() const { return value; }
  [[nodiscard]] constexpr auto get() const { return value; }
  template <typename U> constexpr decltype(auto) operator=(U &&v);
  constexpr void update(const auto &symbols, const auto &updates);
  constexpr void collect(const auto &symbols, auto &out) const;
  [[nodiscard]] constexpr auto derivative() const;
  constexpr void backward(const auto &syms, T adj, auto &grads) const;

  template <typename Syms, std::size_t N>
  [[nodiscard]] constexpr T eval_seeded(const std::array<T, N> &vals) const {
    constexpr auto idx = find_index_of_char<symbol, Syms>();
    return vals[idx];
  }

  template <std::size_t I> [[nodiscard]] constexpr auto get() const {
    static_assert(I < 2);
    if constexpr (requires { std::tuple_size<T>::value; }) {
      return eval().template get<I>();
    } else if constexpr (I == 0) {
      return eval();
    } else {
      return static_cast<T>(derivative());
    }
  }
};

template <Numeric T, char symbol>
template <typename U>
constexpr decltype(auto) Variable<T, symbol>::operator=(U &&v) {
  if constexpr (std::is_same_v<decltype(value),
                               std::reference_wrapper<std::decay_t<U>>>) {
    value.get() = std::forward<U>(v);
  } else if constexpr (!std::is_same_v<std::decay_t<U>, T> &&
                       std::is_constructible_v<T, U>) {
    value = T{std::forward<U>(v)};
  } else {
    value = std::forward<U>(v);
  }
  return *this;
}

template <Numeric T, char symbol>
constexpr void Variable<T, symbol>::update(const auto &symbols,
                                           const auto &updates) {
  using Syms = std::decay_t<decltype(symbols)>;
  constexpr auto index = find_index_of_char<symbol, Syms>();
  *this = updates[index];
}

template <Numeric T, char symbol>
constexpr void Variable<T, symbol>::collect(const auto &symbols,
                                            auto &out) const {
  using Syms = std::decay_t<decltype(symbols)>;
  constexpr auto index = find_index_of_char<symbol, Syms>();
  out[index] = value;
}

template <Numeric T, char symbol>
constexpr auto Variable<T, symbol>::derivative() const {
  auto ret = T{};
  return Constant{++ret};
}

template <Numeric T, char symbol>
constexpr void Variable<T, symbol>::backward(const auto &syms, T adj,
                                             auto &grads) const {
  using Syms = std::decay_t<decltype(syms)>;
  constexpr auto idx = find_index_of_char<symbol, Syms>();
  grads[idx] += adj;
}

#define DEFINE_CONST_UDL(type, suffix)                                         \
  consteval diff::Constant<type> operator"" _##suffix(                         \
      unsigned long long val) {                                                \
    return diff::Constant<type>{static_cast<type>(val)};                       \
  }                                                                            \
  consteval diff::Constant<type> operator"" _##suffix(long double val) {       \
    return diff::Constant<type>{static_cast<type>(val)};                       \
  }

#define DEFINE_VAR_UDL(type, suffix, label)                                    \
  consteval auto operator"" _##suffix(unsigned long long val) {                \
    return diff::Variable<type, label>{static_cast<type>(val)};                \
  }                                                                            \
  consteval auto operator"" _##suffix(long double val) {                       \
    return diff::Variable<type, label>{static_cast<type>(val)};                \
  }

} // namespace diff

DEFINE_CONST_UDL(int, ci)
DEFINE_CONST_UDL(double, cd)
DEFINE_VAR_UDL(int, vi, 'c')
DEFINE_VAR_UDL(double, vd, 'v')

namespace std {
template <diff::Numeric T>
struct tuple_size<diff::Constant<T>> : integral_constant<std::size_t, 2> {};

template <std::size_t I, diff::Numeric T>
struct tuple_element<I, diff::Constant<T>> {
  using type = typename diff::detail::expression_element<T, I>::type;
};

template <diff::Numeric T, char C>
struct tuple_size<diff::Variable<T, C>> : integral_constant<std::size_t, 2> {};

template <std::size_t I, diff::Numeric T, char C>
struct tuple_element<I, diff::Variable<T, C>> {
  using type = typename diff::detail::expression_element<T, I>::type;
};
} // namespace std

#define PDV(x, label)                                                          \
  diff::Variable<diff::Dual<decltype(x)>, label>(diff::Dual<decltype(x)>{x, 0})
#define PV(x, label) diff::Variable<decltype(x), label>(x)
#define PC(x) diff::Constant(x)
