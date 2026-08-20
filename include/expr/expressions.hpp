#pragma once

#include "util/fixed_string.hpp" // FixedString, CFixedString
#include "util/mpl.hpp"          // CSymbol, CSymbolList

#include <concepts>
#include <ranges>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace diff {

// The standard library has no `arithmetic` concept; this composes the two that
// partition it.
template <typename T>
concept CArithmetic = std::integral<std::remove_cvref_t<T>> ||
                      std::floating_point<std::remove_cvref_t<T>>;

// Closed under the four arithmetic operators and negation (Dual, TaylorDual,
// TaylorDual, ...).
template <typename T>
// constructible_from<T, int> is not decoration: differentiation manufactures
// exactly 0 and 1 -- the reverse sweep seeds T{1}, and the int spelling of
// Lit<T, V> is `T(V)`.  A type that cannot spell them cannot be differentiated,
// so the concept says so instead of failing deep inside a sweep.
concept CFieldLike = std::default_initializable<T> &&
                     std::constructible_from<T, int> && requires(T a, T b) {
  { a + b } -> std::convertible_to<T>;
  { a - b } -> std::convertible_to<T>;
  { a * b } -> std::convertible_to<T>;
  { a / b } -> std::convertible_to<T>;
  { -a } -> std::convertible_to<T>;
};

template <typename T>
concept Numeric = CArithmetic<T> || CFieldLike<T>;

// Does multiplication commute for this scalar?  Numeric admits matrices and
// quaternions, so this is declared rather than deduced; the default is the safe
// answer (non-commutative costs a canonicalisation share, never a wrong result).
// Only expr/simplify.hpp reads it, to decide whether a product's operands may be
// put in canonical order.
//
// Specialise it directly for a class template, propagating from the scalar
// underneath:
//
//   template <Numeric T>
//   inline constexpr bool is_commutative_multiply_v<MyDual<T>> =
//       is_commutative_multiply_v<T>;
//
// or reach for DIFF_COMMUTATIVE_MULTIPLY(MyScalar) below for a concrete type.
template <typename T>
inline constexpr bool is_commutative_multiply_v = CArithmetic<T>;

template <typename T>
concept CCommutativeMultiply =
    is_commutative_multiply_v<std::remove_cvref_t<T>>;

// Decomposable into {value, derivative} via std::tuple_size / get<I> — the
// protocol Dual, Constant, Variable and Expression all opt into.
template <typename T>
concept CTupleLike =
    requires { std::tuple_size<std::remove_cvref_t<T>>::value; };

// A random-access buffer of numbers: every sweep threads two of these, the point
// (in canonical symbol order) and the node-value cache.
template <typename R>
concept CNumericBuffer = std::ranges::random_access_range<R> &&
                         Numeric<std::ranges::range_value_t<R>>;

namespace detail {
template <typename F, typename Seq>
inline constexpr bool index_invocable_v = false;
template <typename F, std::size_t... Is>
inline constexpr bool index_invocable_v<F, std::index_sequence<Is...>> =
    (requires(F &&f) { static_cast<F &&>(f).template operator()<Is>(); } &&
     ...);
} // namespace detail

// Invocable as f.template operator()<I>() for every I in [0, N) — the shape
// static_for folds over, and the reason those lambdas take their index as a
// template parameter rather than an argument.
template <typename F, std::size_t N>
concept CIndexedCallable =
    detail::index_invocable_v<F, std::make_index_sequence<N>>;

template <typename O>
concept COperation =
    requires { typename O::value_type; } && Numeric<typename O::value_type>;

template <typename T> inline constexpr bool is_expression_type_v = false;
template <typename T>
concept CExpression = is_expression_type_v<std::remove_cvref_t<T>>;

template <COperation Op, CExpression... Children> class Expression;

// Expression's storage base; also the CRTP `Derived`, so it has to satisfy
// CExpression in its own right.
template <bool Stateless, COperation Op, CExpression... Children>
class ExpressionImpl;

template <COperation Op, CExpression Exp>
using MonoExpression = Expression<Op, Exp>;

template <Numeric T, auto... V> class Lit;
template <Numeric T> using Constant = Lit<T>;

// A symbol-list element: the label lifted to a type, with the name in a form
// the ordering predicate can compare.  See fixed_string.hpp for the label type.
template <auto S> struct symbol_type {
  static constexpr auto value = S;
  static constexpr std::string_view name = S.view();
};

// The same symbol carried as a *value*, for the one place that cannot take a
// template argument: operator[].  Prefer m.get<"x">() except in the assignment
// spelling m["x"_s] = v.  FixedString, not `CFixedString auto`: the placeholder
// has to be the class template so CTAD turns the literal into a FixedString
// rather than decaying it to const char*.
template <FixedString S> inline constexpr symbol_type<S> sym{};

namespace literals {

template <FixedString S> [[nodiscard]] consteval auto operator""_s() noexcept {
  return sym<S>;
}

} // namespace literals

template <Numeric T, CFixedString auto, bool Frozen = false> class Variable;

template <CExpression... Ts> class Equation;

// Every derivative entry point is an Equation member, so a bare expression has
// to reach one.  Declared here (Equation is still incomplete; including
// expr/equation.hpp would be a cycle) and defined out of line.  A mixin rather
// than part of ExpressionOps, because the leaves do not derive from it.
//
// Keep it a *constrained template*.  A plain `operator Equation<Derived>()` is a
// candidate for every is_convertible query, and answering one completes its
// return type -- which canonicalises, which std::applys over the children, which
// asks is_convertible again.  Deducing the target and rejecting it with same_as
// answers those queries without ever naming Equation's members.
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

// Either form of constant leaf: a value with no symbols under it.
template <typename T> inline constexpr bool is_constant_v = false;
template <Numeric T, auto... V>
inline constexpr bool is_constant_v<Lit<T, V...>> = true;
template <typename T>
concept CConstant = is_constant_v<std::remove_cvref_t<T>>;

template <typename T> inline constexpr bool is_lit_v = false;
template <Numeric T, auto V> inline constexpr bool is_lit_v<Lit<T, V>> = true;

template <typename T>
concept CLit = is_lit_v<std::remove_cvref_t<T>>;

// An internal node: Expression<Op, Children...> is the only node that carries
// children, so `children_t` is exactly the branch/leaf discriminator.
template <typename T>
concept CExpressionNode =
    requires { typename std::remove_cvref_t<T>::children_t; };

template <Numeric T> struct EvalResult {
  using value_type = T;
  T value;
  constexpr operator T() const noexcept { return value; }
};

template <Numeric T> inline constexpr bool is_expression_type_v<EvalResult<T>> =
    true;

template <COperation Op> struct BaseExpression {
  using value_type = typename Op::value_type;
};

template <CExpression T> consteval std::size_t node_count_fn() {
  using U = std::remove_cvref_t<T>;
  if constexpr (CExpressionNode<U>) {
    return []<CExpression... C>(
               std::type_identity<std::tuple<C...>>) consteval {
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
  [&]<std::size_t... K>(std::index_sequence<K...>) {
    ((off += node_count_v<std::tuple_element_t<K, Kids>>), ...);
  }(std::make_index_sequence<I>{});
  return off;
}

// One element of a point, in any of the spellings eval() accepts:
// (a) a bare number (positional),
// (b) a range of them,
// (c) a ValueMap,
// (d) or a named value.
template <typename T>
concept CEvalArg = Numeric<T> || std::ranges::input_range<T> ||
                   requires { typename std::remove_cvref_t<T>::symbols; } ||
                   requires { std::remove_cvref_t<T>::symbol; };

namespace detail {
template <CExpression Expr, CEvalArg... Args>
[[nodiscard]] constexpr auto eval_dispatch(const Expr &e, const Args &...args);

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

  // Evaluation always takes a point: the tree stores no values.  Op::eval is
  // reached only through the seeded sweeps below, where every child has
  // already been reduced to an EvalResult.
  [[nodiscard]] constexpr auto eval(const CEvalArg auto &...args) const {
    return detail::eval_dispatch(self(), args...);
  }

  [[nodiscard]] constexpr auto derivative() const noexcept {
    return std::apply(
        [](const auto &...e) noexcept { return Op::derivative(e...); },
        self().expressions());
  }

  // Fused forward sweep at a point; the symbol list is deduced.
  template <FixedString Seed>
  [[nodiscard]] constexpr auto
  eval_with_tangent(const CEvalArg auto &...args) const {
    return detail::tangent_dispatch<Seed>(self(), args...);
  }

  // The one seeded sweep.  The seed type U is deduced from the point, so the
  // same call evaluates in the expression's own scalar type or in a deeper
  // dual type (Dual, TaylorDual, ...) — the arithmetic follows
  // whatever was handed in, and only the leaves care about the difference.
  template <CSymbolList Syms, Numeric U, std::size_t N>
  [[nodiscard]] constexpr U
  eval_seeded(const std::array<U, N> &vals) const noexcept {
    return std::apply(
        [&](const auto &...e) noexcept {
          return Op::eval(
              EvalResult<U>{e.template eval_seeded<Syms>(vals)}...);
        },
        self().expressions());
  }

  template <std::size_t Base = 0>
  constexpr void backward(const auto &syms, value_type adj, auto &grads,
                          const auto &cache) const noexcept {
    using Kids = typename Derived::children_t;
    [&]<std::size_t... I>(std::index_sequence<I...>) noexcept {
      auto child_adj =
          Op::template adjoints<Base, child_base_at<Base, Kids, I>()...>(adj,
                                                                         cache);
      (std::get<I>(self().expressions())
           .template backward<child_base_at<Base, Kids, I>()>(
               syms, std::move(child_adj[I]), grads, cache),
       ...);
    }(std::make_index_sequence<std::tuple_size_v<Kids>>{});
  }
};

// The CRTP parameter is the public Expression type, not this base, so every
// type-level pattern match sees Expression<Op, C...>.
template <COperation Op, CExpression... Children>
class ExpressionImpl<true, Op, Children...>
    : public ExpressionOps<Expression<Op, Children...>, Op> {
  friend ExpressionOps<Expression<Op, Children...>, Op>;

public:
  using children_t = std::tuple<Children...>;

  constexpr ExpressionImpl() noexcept = default;
  constexpr ExpressionImpl(Children...) noexcept {}

  // Children are empty, so materialising them is free.
  [[nodiscard]] constexpr children_t expressions() const noexcept { return {}; }
};

template <COperation Op, CExpression... Children>
class ExpressionImpl<false, Op, Children...>
    : public ExpressionOps<Expression<Op, Children...>, Op> {
  std::tuple<Children...> operands;
  friend ExpressionOps<Expression<Op, Children...>, Op>;

public:
  using children_t = std::tuple<Children...>;

  constexpr ExpressionImpl(Children... c) noexcept : operands{std::move(c)...} {}
  [[nodiscard]] constexpr const children_t &expressions() const noexcept {
    return operands;
  }
};

// The public node type.  It derives from the storage form rather than aliasing
// it so Expression stays a class template and can be pattern-matched by partial
// specialisations.  The base is empty whenever the children are.
template <COperation Op, CExpression... Children>
class Expression
    : public ExpressionImpl<(std::is_empty_v<Children> && ...), Op,
                            Children...> {
  using base_type =
      ExpressionImpl<(std::is_empty_v<Children> && ...), Op, Children...>;

public:
  using base_type::base_type;
  using children_t = typename base_type::children_t;
  static constexpr bool stateless = (std::is_empty_v<Children> && ...);
};

namespace detail {
template <Numeric V, std::size_t I, typename = void>
struct expression_element {
  using type = V;
};

template <Numeric V, std::size_t I>
struct expression_element<V, I,
                          std::void_t<typename std::tuple_element_t<I, V>>> {
  using type = std::tuple_element_t<I, V>;
};
} // namespace detail

} // namespace diff

// Declare a concrete user type's multiplication commutative, so canonicalisation
// may reorder the operands of a product of it.  Write it at GLOBAL scope -- the
// body opens namespace diff.  Variadic, so a template-id containing commas
// survives the preprocessor.  For a class template, specialise
// is_commutative_multiply_v directly; a macro cannot express a partial
// specialisation.
#define DIFF_COMMUTATIVE_MULTIPLY(...)                                         \
  namespace diff {                                                             \
  template <>                                                                  \
  inline constexpr bool is_commutative_multiply_v<__VA_ARGS__> = true;         \
  }

namespace std {
template <diff::COperation Op, diff::CExpression... Children>
struct tuple_size<diff::Expression<Op, Children...>>
    : integral_constant<size_t, 2> {};

template <size_t I, diff::COperation Op, diff::CExpression... Children>
struct tuple_element<I, diff::Expression<Op, Children...>> {
  using type = typename diff::detail::expression_element<
      typename diff::Expression<Op, Children...>::value_type, I>::type;
};
} // namespace std
