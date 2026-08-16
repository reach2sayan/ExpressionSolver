#pragma once

#include <concepts>
#include <ostream>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace diff {

// A built-in arithmetic scalar (cv/ref stripped): the scalar half of every
// mixed scalar/expression and scalar/dual operation.  The standard library has
// no `arithmetic` concept, so this is the one gap we fill ourselves — and only
// by composing the two std concepts that partition it.
template <typename T>
concept CArithmetic = std::integral<std::remove_cvref_t<T>> ||
                      std::floating_point<std::remove_cvref_t<T>>;

// Closed under the four arithmetic operators and negation (Dual, TaylorDual,
// VectorDual, ...).  Arithmetic scalars qualify outright.
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

template <typename O>
concept COperation =
    requires { typename O::value_type; } && Numeric<typename O::value_type>;

template <typename T> struct is_expression_type : std::false_type {};
template <typename T>
concept CExpression = is_expression_type<std::remove_cvref_t<T>>::value;

template <COperation Op, CExpression... Children> class Expression;

// Expression's storage base; also the CRTP `Derived`, so it has to satisfy
// CExpression in its own right.
template <bool Stateless, COperation Op, CExpression... Children>
class ExpressionImpl;

template <COperation Op, CExpression Exp>
using MonoExpression = Expression<Op, Exp>;
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

// `Frozen` holds a symbol constant for partial differentiation: it still reads
// its value from the seed array, but differentiates to zero.  Because freezing
// needs no value, it is a pure type transform.
template <Numeric T, CFixedString auto, bool Frozen = false> class Variable;

// A literal whose value lives in the type, so the object is empty.  This is
// what derivative() manufactures: the 0s and 1s that flood a Jacobian cost
// nothing.  User-written constants stay in Constant<T>, which stores a value
// and can therefore hold any runtime number without minting a type per value.
template <Numeric T, T V> struct Lit;

template <Numeric T> struct is_expression_type<Constant<T>> : std::true_type {};

template <Numeric T, T V>
struct is_expression_type<Lit<T, V>> : std::true_type {};

template <Numeric T, CFixedString auto C, bool F>
struct is_expression_type<Variable<T, C, F>> : std::true_type {};

template <COperation Op, CExpression... Children>
struct is_expression_type<Expression<Op, Children...>> : std::true_type {};

template <bool S, COperation Op, CExpression... Children>
struct is_expression_type<ExpressionImpl<S, Op, Children...>>
    : std::true_type {};

template <typename T> struct is_constant : std::false_type {};
template <Numeric T> struct is_constant<Constant<T>> : std::true_type {};
template <Numeric T, T V> struct is_constant<Lit<T, V>> : std::true_type {};
template <typename T>
concept CConstant = is_constant<std::remove_cvref_t<T>>::value;

template <typename T> struct is_lit : std::false_type {};
template <Numeric T, T V> struct is_lit<Lit<T, V>> : std::true_type {};
// A literal whose value is a template argument, so folding two of them yields
// another empty node rather than a stored one.  This is what keeps derivative
// trees — which are mostly 0s and 1s — free.
template <typename T>
concept CLit = is_lit<std::remove_cvref_t<T>>::value;

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

// Result of a fused forward-mode sweep (eval_with_tangent): the node's value
// and its tangent w.r.t. the seed variable, accumulated together in one tree
// walk. Plain aggregate so `auto [v, dv] = expr.eval_with_tangent<"x">();`
// works.
template <Numeric T> struct Tangent {
  T value;
  T deriv;
};

template <Numeric T>
struct is_expression_type<EvalResult<T>> : std::true_type {};

template <COperation Op> struct BaseExpression {
  using value_type = typename Op::value_type;
};

template <typename T> consteval std::size_t node_count_fn() {
  using U = std::remove_cvref_t<T>;
  if constexpr (CExpressionNode<U>) {
    return []<typename... C>(std::type_identity<std::tuple<C...>>) consteval {
      return (1 + ... + node_count_fn<std::remove_cvref_t<C>>());
    }(std::type_identity<typename U::children_t>{});
  } else {
    return 1; // constant / variable
  }
}

template <typename T>
inline constexpr std::size_t node_count_v =
    node_count_fn<std::remove_cvref_t<T>>();

// Preorder cache slot of the I-th child of a node at `Base1
template <std::size_t Base, typename Kids, std::size_t I>
consteval std::size_t child_base_at() {
  std::size_t off = Base + 1;
  [&]<std::size_t... K>(std::index_sequence<K...>) {
    ((off += node_count_v<std::tuple_element_t<K, Kids>>), ...);
  }(std::make_index_sequence<I>{});
  return off;
}

namespace detail {
// Defined in bound.hpp.  Declared here so ExpressionOps::eval can forward to
// it; the definition needs the symbol machinery from traits.hpp, which cannot
// be included from this header.
template <CExpression Expr, typename... Args>
[[nodiscard]] constexpr auto eval_dispatch(const Expr &e, const Args &...args);

template <auto Seed, CExpression Expr, typename... Args>
[[nodiscard]] constexpr auto tangent_dispatch(const Expr &e,
                                              const Args &...args);
} // namespace detail

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
  // upward through the tree, seeded on the variable named `Seed`.  Unlike
  // derivative() (which builds a separate derivative tree) and unlike the
  // Dual<T> path (which substitutes value_type), this walks the existing tree
  // structurally and hands each Op the already-computed child Tangents via
  // Op::forward.
  template <FixedString Seed, typename Syms, std::size_t N>
  [[nodiscard]] constexpr auto
  tangent_seeded(const std::array<value_type, N> &vals) const noexcept {
    return std::apply(
        [&](const auto &...e) noexcept {
          return Op::forward(
              e.template tangent_seeded<Seed, Syms>(vals)...);
        },
        self().expressions());
  }

  template <typename Syms, std::size_t N>
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
  template <typename U, typename Syms, std::size_t N>
  [[nodiscard]] constexpr U
  eval_seeded_as(const std::array<U, N> &vals) const noexcept {
    return std::apply(
        [&](const auto &...e) noexcept {
          return Op::eval(
              EvalResult<U>{e.template eval_seeded_as<U, Syms>(vals)}...);
        },
        self().expressions());
  }

  constexpr void update(const auto &symbols, const auto &updates) noexcept {
    std::apply([&](auto &...e) noexcept { (e.update(symbols, updates), ...); },
               self().expressions());
  }

  constexpr void collect(const auto &symbols, auto &out) const noexcept {
    std::apply(
        [&](const auto &...e) noexcept { (e.collect(symbols, out), ...); },
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
template <diff::COperation Op, diff::CExpression... Children>
struct tuple_size<diff::Expression<Op, Children...>>
    : integral_constant<size_t, 2> {};

template <size_t I, diff::COperation Op, diff::CExpression... Children>
struct tuple_element<I, diff::Expression<Op, Children...>> {
  using type = typename diff::detail::expression_element<
      typename diff::Expression<Op, Children...>::value_type, I>::type;
};
} // namespace std
