#pragma once

#include "symbolic/symbol.hpp"       // CSymbol, CSymbolList
#include "ops/numeric.hpp"       // Numeric, COperation, CExpression
#include "util/fixed_string.hpp" // FixedString, CFixedString

#include <concepts>
#include <ranges>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ddx::impl {

template <FixedString S, typename V>
  requires std::is_object_v<V>
struct NamedValue;

// A symbol-list element: the label lifted to a type.
template <auto S> struct symbol_type {
  static constexpr auto value = S;
  static constexpr std::string_view name = S.view();

  // "x"_s = 1.5 -- one entry of a point or a map, with the symbol in hand
  // rather than in a template argument.  Assignment rather than a call because
  // the pair reads as one at the call site; a symbol carries no state for a
  // real assignment to overwrite.  Defined in named_value.hpp, where the thing
  // it returns is.
  template <typename V>
    requires std::is_object_v<V>
  constexpr NamedValue<S, V> operator=(V v) const;
};

// The same symbol as a value, for operator[], which takes no template argument.
// FixedString, not `CFixedString auto`: CTAD has to turn the literal into a
// FixedString rather than decay it to const char*.
template <FixedString S> inline constexpr symbol_type<S> sym{};

namespace literals {

template <FixedString S> [[nodiscard]] consteval auto operator""_s() noexcept {
  return sym<S>;
}

} // namespace literals

template <Numeric T, CFixedString auto, bool Frozen = false> class Variable;

// Unconstrained, so include/rt can specialise it for a runtime graph.  The
// compile-time definition in expr/equation.hpp carries the CExpression
// constraint itself, and every use of Equation<...> in this header and that one
// is constrained too, so nothing here widens.
template <typename... Ts> class Equation;

// Reaches Equation from a bare expression.  Declared here and defined out of
// line, since Equation is still incomplete.  Keep it a constrained *template*:
// a plain operator Equation<Derived>() is a candidate for every is_convertible
// query, and answering one completes the return type, which asks again.
template <typename Derived> struct EquationConvertible {
  template <typename Eq>
    requires std::same_as<Eq, Equation<Derived>>
  constexpr operator Eq() const noexcept;
};

template <Numeric T, auto... V>
inline constexpr bool is_expression_type_v<Lit<T, V...>> = true;

template <Numeric T, CFixedString auto C, bool F>
inline constexpr bool is_expression_type_v<Variable<T, C, F>> = true;

template <COperation Op, CExpression... Children>
inline constexpr bool is_expression_type_v<Expression<Op, Children...>> = true;

template <bool S, COperation Op, CExpression... Children>
inline constexpr bool is_expression_type_v<ExpressionImpl<S, Op, Children...>> =
    true;

template <typename T> inline constexpr bool is_constant_v = false;
template <Numeric T, auto... V>
inline constexpr bool is_constant_v<Lit<T, V...>> = true;
template <typename T>
concept CConstant = is_constant_v<std::remove_cvref_t<T>>;

template <typename T> inline constexpr bool is_lit_v = false;
template <Numeric T, auto V> inline constexpr bool is_lit_v<Lit<T, V>> = true;

template <typename T>
concept CLit = is_lit_v<std::remove_cvref_t<T>>;

// `children_t` is exactly the branch/leaf discriminator.
template <typename T>
concept CExpressionNode =
    requires { typename std::remove_cvref_t<T>::children_t; };

template <Numeric T> struct EvalResult {
  using value_type = T;
  T value;
  constexpr operator T() const noexcept { return value; }
};

template <Numeric T>
inline constexpr bool is_expression_type_v<EvalResult<T>> = true;

template <COperation Op> struct BaseExpression {
  using value_type = typename Op::value_type;
};

template <CExpression T> consteval std::size_t node_count_fn() {
  using U = std::remove_cvref_t<T>;
  if constexpr (CExpressionNode<U>) {
    return
        []<CExpression... C>(std::type_identity<std::tuple<C...>>) consteval {
          return (1 + ... + node_count_fn<std::remove_cvref_t<C>>());
        }(std::type_identity<typename U::children_t>{});
  } else {
    return 1; // constant / variable
  }
}

template <CExpression T>
inline constexpr std::size_t node_count_v =
    node_count_fn<std::remove_cvref_t<T>>();

// Preorder cache slot of the I-th child of a node at `Base1
template <std::size_t Base, CTupleLike Kids, std::size_t I>
consteval std::size_t child_base_at() {
  std::size_t off = Base + 1;
  static_for<I>([&]<std::size_t K>() {
    off += node_count_v<std::tuple_element_t<K, Kids>>;
  });
  return off;
}

// One element of a point: a bare number, a range of them, a ValueMap, or a
// named value.
template <typename T>
concept CEvalArg = Numeric<T> || std::ranges::input_range<T> || requires {
  typename std::remove_cvref_t<T>::symbols;
} || requires { std::remove_cvref_t<T>::symbol; };

namespace detail {
template <CExpression Expr, CEvalArg... Args>
[[nodiscard]] constexpr auto eval_dispatch(const Expr &e,
                                           const Args &...args) noexcept;

template <auto Seed, CExpression Expr, CEvalArg... Args>
[[nodiscard]] constexpr auto tangent_dispatch(const Expr &e,
                                              const Args &...args);
} // namespace detail

template <CExpression Derived, COperation Op>
class ExpressionOps : public BaseExpression<Op>,
                      public EquationConvertible<Derived> {
  [[nodiscard]] constexpr const Derived &self() const noexcept {
    return static_cast<const Derived &>(*this);
  }

public:
  using op_type = Op;
  using value_type = typename BaseExpression<Op>::value_type;

  // The tree stores no values, so evaluation always takes a point.
  [[nodiscard]] constexpr auto eval(const CEvalArg auto &...args) const {
    return detail::eval_dispatch(self(), args...);
  }

  [[nodiscard]] constexpr auto derivative() const noexcept {
    return std::apply(
        [](const auto &...e) noexcept { return Op::derivative(e...); },
        self().expressions());
  }

  template <FixedString Seed>
  [[nodiscard]] constexpr auto
  eval_with_tangent(const CEvalArg auto &...args) const {
    return detail::tangent_dispatch<Seed>(self(), args...);
  }

  // The one seeded sweep.  U is deduced from the point, so the arithmetic
  // follows whatever was handed in and only the leaves see the difference.
  template <CSymbolList Syms, Numeric U, std::size_t N>
  [[nodiscard]] constexpr U
  eval_seeded(const std::array<U, N> &vals) const noexcept {
    return std::apply(
        [&](const auto &...e) noexcept {
          return Op::eval(EvalResult<U>{e.template eval_seeded<Syms>(vals)}...);
        },
        self().expressions());
  }

  template <std::size_t Base = 0>
  constexpr void backward(const auto &syms, value_type adj, auto &grads,
                          const auto &cache) const noexcept {
    using Kids = typename Derived::children_t;
    index_apply<std::tuple_size_v<Kids>>([&]<std::size_t... I>() noexcept {
      auto child_adj =
          Op::template adjoints<Base, child_base_at<Base, Kids, I>()...>(adj,
                                                                         cache);
      (std::get<I>(self().expressions())
           .template backward<child_base_at<Base, Kids, I>()>(
               syms, std::move(child_adj[I]), grads, cache),
       ...);
    });
  }
};

// The CRTP parameter is the public Expression type, not this base.
template <COperation Op, CExpression... Children>
class ExpressionImpl<true, Op, Children...>
    : public ExpressionOps<Expression<Op, Children...>, Op> {
  friend ExpressionOps<Expression<Op, Children...>, Op>;

public:
  using children_t = std::tuple<Children...>;

  constexpr ExpressionImpl() noexcept = default;
  constexpr ExpressionImpl(Children...) noexcept {}

  [[nodiscard]] constexpr children_t expressions() const noexcept { return {}; }
};

template <COperation Op, CExpression... Children>
class ExpressionImpl<false, Op, Children...>
    : public ExpressionOps<Expression<Op, Children...>, Op> {
  std::tuple<Children...> operands;
  friend ExpressionOps<Expression<Op, Children...>, Op>;

public:
  using children_t = std::tuple<Children...>;

  constexpr ExpressionImpl(Children... c) noexcept
      : operands{std::move(c)...} {}
  [[nodiscard]] constexpr const children_t &expressions() const noexcept {
    return operands;
  }
};

// Derives from the storage form rather than aliasing it, so Expression stays a
// class template that partial specialisations can match.
template <COperation Op, CExpression... Children>
class Expression : public ExpressionImpl<(std::is_empty_v<Children> && ...), Op,
                                         Children...> {
  using base_type =
      ExpressionImpl<(std::is_empty_v<Children> && ...), Op, Children...>;

public:
  using base_type::base_type;
  using children_t = typename base_type::children_t;
  static constexpr bool stateless = (std::is_empty_v<Children> && ...);
};

namespace detail {
template <Numeric V, std::size_t I, typename = void> struct expression_element {
  using type = V;
};

template <Numeric V, std::size_t I>
struct expression_element<V, I,
                          std::void_t<typename std::tuple_element_t<I, V>>> {
  using type = std::tuple_element_t<I, V>;
};
} // namespace detail

} // namespace ddx::impl

// At GLOBAL scope: the body opens namespace ddx::impl.  Variadic, so a
// template-id containing commas survives the preprocessor.
#define DDX_COMMUTATIVE_MULTIPLY(...)                                          \
  namespace ddx::impl {                                                        \
  template <>                                                                  \
  inline constexpr bool is_commutative_multiply_v<__VA_ARGS__> = true;         \
  }

namespace std {
template <ddx::impl::COperation Op, ddx::impl::CExpression... Children>
struct tuple_size<ddx::impl::Expression<Op, Children...>>
    : integral_constant<size_t, 2> {};

template <size_t I, ddx::impl::COperation Op,
          ddx::impl::CExpression... Children>
struct tuple_element<I, ddx::impl::Expression<Op, Children...>> {
  using type = typename ddx::impl::detail::expression_element<
      typename ddx::impl::Expression<Op, Children...>::value_type, I>::type;
};
} // namespace std
