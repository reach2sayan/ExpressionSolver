#pragma once

#include <concepts>
#include <ostream>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace diff {

template <typename T>
concept Numeric = std::is_arithmetic_v<T> || requires(T a, T b) {
  T{};
  { a + b } -> std::convertible_to<T>;
  { a - b } -> std::convertible_to<T>;
  { a * b } -> std::convertible_to<T>;
  { a / b } -> std::convertible_to<T>;
  { -a } -> std::convertible_to<T>;
};

template <typename O>
concept COperation =
    requires { typename O::value_type; } && Numeric<typename O::value_type>;

template <COperation Op, typename LHS, typename RHS> class Expression;
template <COperation Op, typename Exp> class MonoExpression;
template <Numeric T> class Constant;

template <std::size_t N> struct FixedString {
  char data[N];
  constexpr FixedString(const char (&str)[N]) noexcept {
    for (std::size_t i = 0; i < N; ++i)
      data[i] = str[i];
  }
  constexpr bool operator==(const FixedString &) const noexcept = default;
  template <std::size_t M>
  constexpr bool operator==(const FixedString<M> &) const noexcept {
    return false;
  }
  constexpr std::string_view view() const noexcept { return {data, N - 1}; }
};
template <std::size_t N> FixedString(const char (&)[N]) -> FixedString<N>;

namespace detail {
template <typename T> struct is_fixed_string_t : std::false_type {};
template <std::size_t N>
struct is_fixed_string_t<FixedString<N>> : std::true_type {};
} // namespace detail

template <typename T>
concept CFixedString = detail::is_fixed_string_t<T>::value;

template <auto S> struct symbol_type {
  static constexpr auto value = S;
  static constexpr std::string_view name = S.view();
};

template <typename H, typename T>
concept CHook = requires(H h, T adj) {
  { h.get_f() } -> std::convertible_to<T>;
  h.accum_df(adj);
  h.zero_df();
};

template <Numeric T, CFixedString auto, typename Storage = T> class Variable;

template <typename T> struct is_expression_type : std::false_type {};
template <Numeric T> struct is_expression_type<Constant<T>> : std::true_type {};

template <Numeric T, CFixedString auto C, typename S>
struct is_expression_type<Variable<T, C, S>> : std::true_type {};

template <COperation Op, typename LHS, typename RHS>
struct is_expression_type<Expression<Op, LHS, RHS>> : std::true_type {};

template <COperation Op, typename Exp>
struct is_expression_type<MonoExpression<Op, Exp>> : std::true_type {};

template <typename T>
concept CExpression = is_expression_type<std::remove_cvref_t<T>>::value;

template <typename T> struct is_constant : std::false_type {};
template <Numeric T> struct is_constant<Constant<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_constant_v =
    is_constant<std::remove_cvref_t<T>>::value;

template <Numeric T> struct EvalResult {
  using value_type = T;
  T value;
  [[nodiscard]] constexpr T eval() const noexcept { return value; }
  constexpr operator T() const noexcept { return value; }
};

template <Numeric T>
struct is_expression_type<EvalResult<T>> : std::true_type {};

template <COperation Op> struct BaseExpression {
  using value_type = typename Op::value_type;
};

// ===========================================================================
// ExpressionOps — shared CRTP base implementing every fan-out algorithm once.
//
// A unary MonoExpression and a binary Expression evaluate, differentiate, seed,
// collect and back-propagate identically: unpack the children and hand them to
// the operator.  The only thing that varies is the arity, which the derived
// node exposes through `children()` as a tuple of references; std::apply then
// adapts the call for free.  Add a new node arity and the algorithms come along
// for free — there is nothing per-node to re-implement.
// ===========================================================================
template <typename Derived, COperation Op>
class ExpressionOps : public BaseExpression<Op> {
  [[nodiscard]] constexpr const Derived &self() const noexcept {
    return static_cast<const Derived &>(*this);
  }
  [[nodiscard]] constexpr Derived &self() noexcept {
    return static_cast<Derived &>(*this);
  }

public:
  using op_type = Op;
  using value_type = typename BaseExpression<Op>::value_type;

  template <std::size_t I> [[nodiscard]] constexpr auto get() const noexcept {
    static_assert(I < 2);
    if constexpr (requires { std::tuple_size<value_type>::value; }) {
      return eval().template get<I>();
    } else if constexpr (I == 0) {
      return eval();
    } else {
      return static_cast<value_type>(derivative());
    }
  }

  [[nodiscard]] constexpr auto eval() const noexcept {
    return std::apply([](const auto &...e) noexcept { return Op::eval(e...); },
                      self().children());
  }
  constexpr operator value_type() const noexcept { return eval(); }

  [[nodiscard]] constexpr auto derivative() const noexcept {
    return std::apply(
        [](const auto &...e) noexcept { return Op::derivative(e...); },
        self().children());
  }

  template <typename Syms, std::size_t N>
  [[nodiscard]] constexpr auto
  eval_seeded(const std::array<value_type, N> &vals) const noexcept {
    return std::apply(
        [&](const auto &...e) noexcept {
          return Op::eval(
              EvalResult<value_type>{e.template eval_seeded<Syms>(vals)}...);
        },
        self().children());
  }

  // eval_seeded_as<U>: evaluate with seeds of a deeper dual type U.
  template <typename U, typename Syms, std::size_t N>
  [[nodiscard]] constexpr U
  eval_seeded_as(const std::array<U, N> &vals) const noexcept {
    return std::apply(
        [&](const auto &...e) noexcept {
          return Op::eval(
              EvalResult<U>{e.template eval_seeded_as<U, Syms>(vals)}...);
        },
        self().children());
  }

  constexpr void update(const auto &symbols, const auto &updates) noexcept {
    std::apply([&](auto &...e) noexcept { (e.update(symbols, updates), ...); },
               self().children());
  }

  constexpr void collect(const auto &symbols, auto &out) const noexcept {
    std::apply(
        [&](const auto &...e) noexcept { (e.collect(symbols, out), ...); },
        self().children());
  }

  constexpr void backward(const auto &syms, value_type adj,
                          auto &grads) const noexcept {
    std::apply(
        [&](const auto &...e) noexcept { Op::backward(e..., adj, syms, grads); },
        self().children());
  }
};

// ===========================================================================
// MonoExpression — unary node (one child).
// ===========================================================================
template <COperation Op, typename Exp>
class MonoExpression : public ExpressionOps<MonoExpression<Op, Exp>, Op> {
  Exp expression;
  friend constexpr std::ostream &operator<<(std::ostream &out,
                                            const MonoExpression &e) {
    out << e.expression;
    return out;
  }

  friend ExpressionOps<MonoExpression<Op, Exp>, Op>;
  [[nodiscard]] constexpr auto children() const noexcept {
    return std::tie(expression);
  }
  [[nodiscard]] constexpr auto children() noexcept {
    return std::tie(expression);
  }

public:
  using lhs_type = Exp;
  constexpr MonoExpression(Exp expr) noexcept : expression{std::move(expr)} {}

  [[nodiscard]] constexpr const auto &expressions() const noexcept {
    return expression;
  }
};

// ===========================================================================
// Expression — binary expression node (two children).
// ===========================================================================
template <COperation Op, typename LHS, typename RHS>
class Expression : public ExpressionOps<Expression<Op, LHS, RHS>, Op> {
  std::pair<LHS, RHS> inner_expressions;
  friend std::ostream &operator<<(std::ostream &out, const Expression &e) {
    out << '(';
    std::apply([&out](const auto &...e) { Op::print(out, e...); },
               e.inner_expressions);
    out << ')';
    return out;
  }

  friend ExpressionOps<Expression<Op, LHS, RHS>, Op>;
  [[nodiscard]] constexpr auto children() const noexcept {
    return std::tie(inner_expressions.first, inner_expressions.second);
  }
  [[nodiscard]] constexpr auto children() noexcept {
    return std::tie(inner_expressions.first, inner_expressions.second);
  }

public:
  using lhs_type = LHS;
  using rhs_type = RHS;
  constexpr Expression(LHS lhs, RHS rhs) noexcept
      : inner_expressions({std::move(lhs), std::move(rhs)}) {}

  [[nodiscard]] constexpr const auto &expressions() const noexcept {
    return inner_expressions;
  }
};

namespace detail {
template <typename V, std::size_t I, typename = void>
struct expression_element {
  using type = V;
};

template <typename V, std::size_t I>
struct expression_element<V, I,
                          std::void_t<typename std::tuple_element_t<I, V>>> {
  using type = std::tuple_element_t<I, V>;
};
} // namespace detail

} // namespace diff

namespace std {
template <diff::COperation Op, typename LHS, typename RHS>
struct tuple_size<diff::Expression<Op, LHS, RHS>>
    : integral_constant<size_t, 2> {};

template <size_t I, diff::COperation Op, typename LHS, typename RHS>
struct tuple_element<I, diff::Expression<Op, LHS, RHS>> {
  using type = typename diff::detail::expression_element<
      typename diff::Expression<Op, LHS, RHS>::value_type, I>::type;
};

template <diff::COperation Op, typename Exp>
struct tuple_size<diff::MonoExpression<Op, Exp>>
    : integral_constant<size_t, 2> {};

template <size_t I, diff::COperation Op, typename Exp>
struct tuple_element<I, diff::MonoExpression<Op, Exp>> {
  using type = typename diff::detail::expression_element<
      typename diff::MonoExpression<Op, Exp>::value_type, I>::type;
};
} // namespace std
