#pragma once

#include "fixed_string.hpp" // FixedString, CFixedString
#include "mpl.hpp"          // CSymbol, CSymbolList

#include <concepts>
#include <ostream>
#include <ranges>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace diff {

// The standard library has no `arithmetic` concept, so this is the one gap we
// fill ourselves — and only by composing the two std concepts that partition
// it.
template <typename T>
concept CArithmetic = std::integral<std::remove_cvref_t<T>> ||
                      std::floating_point<std::remove_cvref_t<T>>;

// Closed under the four arithmetic operators and negation (Dual, TaylorDual,
// VectorDual, ...).
template <typename T>
concept CFieldLike = std::default_initializable<T> && requires(T a, T b) {
  { a + b } -> std::convertible_to<T>;
  { a - b } -> std::convertible_to<T>;
  { a * b } -> std::convertible_to<T>;
  { a / b } -> std::convertible_to<T>;
  { -a } -> std::convertible_to<T>;
};

template <typename T>
concept Numeric = CArithmetic<T> || CFieldLike<T>;

// Decomposable into {value, derivative} via std::tuple_size / get<I> — the
// protocol Dual, Constant, Variable and Expression all opt into.
template <typename T>
concept CTupleLike =
    requires { std::tuple_size<std::remove_cvref_t<T>>::value; };

// A random-access buffer of numbers.  Every sweep threads two of these — the
// point (in canonical symbol order) and the node-value cache — and both happen
// to be std::array, but nothing in the sweeps requires that.
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
template <Numeric T> class Constant;

// A symbol-list element: the label lifted to a type, with the name in a form
// the ordering predicate can compare.  See fixed_string.hpp for the label type.
template <auto S> struct symbol_type {
  static constexpr auto value = S;
  // `name` views into S.data, and S is a template-parameter object, so it has
  // static storage duration.  That is what makes the views symbol_order<Expr>()
  // returns safe to keep indefinitely — it is the library's only API that hands
  // back a view rather than owned storage.
  static constexpr std::string_view name = S.view();
};

// `Frozen` holds a symbol constant for partial differentiation: it still reads
// its value from the seed array, but differentiates to zero.  Because freezing
// needs no value, it is a pure type transform.
template <Numeric T, CFixedString auto, bool Frozen = false> class Variable;

// A literal whose value lives in the type, so the object is empty.  This is
// what derivative() manufactures: the 0s and 1s that flood a Jacobian cost
// nothing.  User-written constants stay in Constant<T>, which stores a value
// and can therefore hold any runtime number without minting a type per value.
template <Numeric T, T V> struct Lit;

template <Numeric T> inline constexpr bool is_expression_type_v<Constant<T>> =
    true;

template <Numeric T, T V>
inline constexpr bool is_expression_type_v<Lit<T, V>> = true;

template <Numeric T, CFixedString auto C, bool F>
inline constexpr bool is_expression_type_v<Variable<T, C, F>> = true;

template <COperation Op, CExpression... Children>
inline constexpr bool is_expression_type_v<Expression<Op, Children...>> = true;

template <bool S, COperation Op, CExpression... Children>
inline constexpr bool is_expression_type_v<ExpressionImpl<S, Op, Children...>> =
    true;

template <typename T> inline constexpr bool is_constant_v = false;
template <Numeric T> inline constexpr bool is_constant_v<Constant<T>> = true;
template <Numeric T, T V> inline constexpr bool is_constant_v<Lit<T, V>> = true;
template <typename T>
concept CConstant = is_constant_v<std::remove_cvref_t<T>>;

template <typename T> inline constexpr bool is_lit_v = false;
template <Numeric T, T V> inline constexpr bool is_lit_v<Lit<T, V>> = true;
// A literal whose value is a template argument, so folding two of them yields
// another empty node rather than a stored one.  This is what keeps derivative
// trees — which are mostly 0s and 1s — free.
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
  [[nodiscard]] constexpr T eval() const noexcept { return value; }
  constexpr operator T() const noexcept { return value; }
};

// Result of a fused forward-mode sweep (eval_with_tangent)
template <Numeric T> struct Tangent {
  T value;
  T deriv;
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

// One element of a point, in any of the spellings eval() accepts: a bare
// number (positional), a range of them, a ValueMap, or a named value.  The
// last two are matched structurally rather than by name — they are defined
// further down the include chain (bound.hpp, gradient.hpp), and this concept
// has to be visible on the eval_dispatch declaration below.
template <typename T>
concept CEvalArg = Numeric<T> || std::ranges::input_range<T> ||
                   requires { typename std::remove_cvref_t<T>::symbols; } ||
                   requires { std::remove_cvref_t<T>::symbol; };

namespace detail {
// Defined in bound.hpp.  Declared here so ExpressionOps::eval can forward to
// it; the definition needs the symbol machinery from traits.hpp, which cannot
// be included from this header.
template <CExpression Expr, CEvalArg... Args>
[[nodiscard]] constexpr auto eval_dispatch(const Expr &e, const Args &...args);

template <auto Seed, CExpression Expr, CEvalArg... Args>
[[nodiscard]] constexpr auto tangent_dispatch(const Expr &e,
                                              const Args &...args);
} // namespace detail

template <CExpression Derived, COperation Op>
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

  // Evaluation always takes a point: the tree stores no values.  Op::eval is
  // reached only through the seeded sweeps below, where every child has
  // already been reduced to an EvalResult.
  //
  // The member forwards to detail::eval_dispatch (declared above, defined in
  // bound.hpp) rather than to the free eval(): a member named `eval` hides the
  // free function, so unqualified lookup here would never reach it.
  // Nullary is legal exactly when the expression has no free symbols (a
  // constant-folded tree); otherwise a point is required.
  template <typename... Args>
  [[nodiscard]] constexpr auto eval(const Args &...args) const {
    return detail::eval_dispatch(self(), args...);
  }

  [[nodiscard]] constexpr auto derivative() const noexcept {
    return std::apply(
        [](const auto &...e) noexcept { return Op::derivative(e...); },
        self().expressions());
  }

  // Fused forward sweep at a point; the symbol list is deduced.
  template <FixedString Seed, typename... Args>
  [[nodiscard]] constexpr auto eval_with_tangent(const Args &...args) const {
    return detail::tangent_dispatch<Seed>(self(), args...);
  }

  // Fused forward-mode sweep: one recursion that carries {value, tangent}
  // upward through the tree, seeded on the variable named `Seed`.  Walks the
  // existing tree structurally and hands each Op the already-computed child
  // Tangents via Op::forward.
  template <FixedString Seed, CSymbolList Syms, std::size_t N>
  [[nodiscard]] constexpr auto
  tangent_seeded(const std::array<value_type, N> &vals) const noexcept {
    return std::apply(
        [&](const auto &...e) noexcept {
          return Op::forward(
              e.template tangent_seeded<Seed, Syms>(vals)...);
        },
        self().expressions());
  }

  template <CSymbolList Syms, std::size_t N>
  [[nodiscard]] constexpr auto
  eval_seeded(const std::array<value_type, N> &vals) const noexcept {
    return std::apply(
        [&](const auto &...e) noexcept {
          return Op::eval(
              EvalResult<value_type>{e.template eval_seeded<Syms>(vals)}...);
        },
        self().expressions());
  }

  // eval_seeded_as<U>: evaluate with seeds of a deeper dual type U.
  template <Numeric U, CSymbolList Syms, std::size_t N>
  [[nodiscard]] constexpr U
  eval_seeded_as(const std::array<U, N> &vals) const noexcept {
    return std::apply(
        [&](const auto &...e) noexcept {
          return Op::eval(
              EvalResult<U>{e.template eval_seeded_as<U, Syms>(vals)}...);
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

// Operand storage.
//
// When every child is empty the node stores NOTHING — the children live only
// as template arguments, so they are never subobjects and nothing can collide.
// This matters because empty-base-optimisation does *not* save you here:
// std::tuple<X,X> is 2 bytes, not 1, since both slots ultimately inherit the
// same X and two base subobjects of one type need distinct addresses.  An
// F4-shaped tree keeping the tuple costs 6 bytes; dropping the member costs 1.
//
// A child that genuinely holds data (Constant<T>) forces the storing form.
template <bool Stateless, COperation Op, CExpression... Children>
class ExpressionImpl;

// The CRTP parameter is the public Expression type, not this base, so that
// every type-level pattern match sees Expression<Op, C...> as it always has.
template <COperation Op, CExpression... Children>
class ExpressionImpl<true, Op, Children...>
    : public ExpressionOps<Expression<Op, Children...>, Op> {
  friend std::ostream &operator<<(std::ostream &out, const ExpressionImpl &e) {
    if constexpr (sizeof...(Children) == 1) {
      out << std::get<0>(e.expressions());
    } else {
      out << '(';
      std::apply([&out](const auto &...c) { Op::print(out, c...); },
                 e.expressions());
      out << ')';
    }
    return out;
  }

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
  friend std::ostream &operator<<(std::ostream &out, const ExpressionImpl &e) {
    if constexpr (sizeof...(Children) == 1) {
      out << std::get<0>(e.operands);
    } else {
      out << '(';
      std::apply([&out](const auto &...c) { Op::print(out, c...); },
                 e.operands);
      out << ')';
    }
    return out;
  }

  friend ExpressionOps<Expression<Op, Children...>, Op>;

public:
  using children_t = std::tuple<Children...>;

  constexpr ExpressionImpl(Children... c) noexcept : operands{std::move(c)...} {}
  [[nodiscard]] constexpr const children_t &expressions() const noexcept {
    return operands;
  }
  [[nodiscard]] constexpr children_t &expressions() noexcept {
    return operands;
  }
};

// The public node type.  It derives from the storage form rather than aliasing
// it so that Expression stays a class template and can be pattern-matched by
// partial specialisations.  The base is empty whenever the children are, so
// this adds no bytes.
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
