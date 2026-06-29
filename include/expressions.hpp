#pragma once

#include <concepts>
#include <ostream>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

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

// is_expression_type is the single source of truth for "is this a node?", and
// the CExpression concept reads it.  Both are declared up here (the per-type
// specialisations come below, once the node types are declared) so that the
// Expression node can constrain its children to themselves be expressions.
template <typename T> struct is_expression_type : std::false_type {};
template <typename T>
concept CExpression = is_expression_type<std::remove_cvref_t<T>>::value;

template <COperation Op, CExpression... Children> class Expression;
// A unary node is just a one-child Expression.  Kept as an alias so the rest of
// the codebase (and the arity-specialised traits) can still spell "unary" while
// there is a single node template underneath.
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

template <typename H, typename T>
concept CHook = requires(H h, T adj) {
  { h.get_f() } -> std::convertible_to<T>;
  h.accum_df(adj);
  h.zero_df();
};

template <Numeric T, CFixedString auto, typename Storage = T> class Variable;

template <Numeric T> struct is_expression_type<Constant<T>> : std::true_type {};

template <Numeric T, CFixedString auto C, typename S>
struct is_expression_type<Variable<T, C, S>> : std::true_type {};

template <COperation Op, CExpression... Children>
struct is_expression_type<Expression<Op, Children...>> : std::true_type {};

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
// Every node — unary, binary, n-ary — evaluates, differentiates, seeds,
// collects and back-propagates identically: unpack the operands and hand them
// to the operator.  The only thing that varies is the arity, which the derived
// node exposes through `expressions()` as its operand tuple; std::apply then
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
                      self().expressions());
  }
  constexpr operator value_type() const noexcept { return eval(); }

  [[nodiscard]] constexpr auto derivative() const noexcept {
    return std::apply(
        [](const auto &...e) noexcept { return Op::derivative(e...); },
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

  // Reverse sweep.  `Base` is this node's preorder slot in the value cache;
  // Op::backward derives its children's slots from it (see node_count_v /
  // child_base_v / rhs_base_v below) and reads operand values from `cache`
  // instead of recomputing them.
  template <std::size_t Base = 0>
  constexpr void backward(const auto &syms, value_type adj, auto &grads,
                          const auto &cache) const noexcept {
    std::apply(
        [&](const auto &...e) noexcept {
          Op::template backward<Base>(e..., adj, syms, grads, cache);
        },
        self().expressions());
  }
};

// ===========================================================================
// Expression — n-ary expression node.
//
// A single variadic node covers every arity: a unary node is a one-child
// Expression (spelled MonoExpression via the alias above), a binary node has
// two children, and so on.  Operands live in one std::tuple; ExpressionOps
// folds over it with std::apply, so eval/derivative/seed/collect/backward are
// arity-agnostic.  There is no separate children() accessor any more — the
// operand tuple is the storage, exposed directly through expressions().
// ===========================================================================
template <COperation Op, CExpression... Children>
class Expression : public ExpressionOps<Expression<Op, Children...>, Op> {
  std::tuple<Children...> operands;
  friend std::ostream &operator<<(std::ostream &out, const Expression &e) {
    if constexpr (sizeof...(Children) == 1) {
      out << std::get<0>(e.operands);
    } else {
      out << '(';
      std::apply([&out](const auto &...c) { Op::print(out, c...); }, e.operands);
      out << ')';
    }
    return out;
  }

  friend ExpressionOps<Expression<Op, Children...>, Op>;

public:
  using children_t = std::tuple<Children...>;
  // Back-compat aliases for the arity-specialised traits.  lhs_type is the
  // first operand; rhs_type is the second when binary (and harmlessly the first
  // for a unary node, where the binary traits never read it).
  using lhs_type = std::tuple_element_t<0, children_t>;
  using rhs_type =
      std::tuple_element_t<(sizeof...(Children) > 1 ? 1 : 0), children_t>;

  constexpr Expression(Children... c) noexcept : operands{std::move(c)...} {}

  [[nodiscard]] constexpr const children_t &expressions() const noexcept {
    return operands;
  }
  [[nodiscard]] constexpr children_t &expressions() noexcept { return operands; }
};

// ===========================================================================
// Node counting + preorder cache offsets.
//
// Lives here (not traits.hpp) so operations.hpp can use it in Op::backward
// without an include cycle (operations -> traits -> values -> operations).
// Every node — internal or leaf — gets one cache slot.  A node at preorder
// slot `Base` owns slot `Base`; a unary child starts at `Base + 1`; a binary
// node's lhs subtree starts at `Base + 1` and its rhs subtree at
// `Base + 1 + node_count_v<LHS>`.  The forward fill (fill_cache) and the
// reverse sweep (Op::backward) MUST agree on these offsets — both go through
// child_base_v / rhs_base_v.
// ===========================================================================
// One rule: a node's count is 1 + the sum over its children.  An internal node
// exposes its operand types as children_t; leaves (Constant/Variable) don't, so
// the `requires` falls through to 1.  The fold over the child type pack is the
// reduce -- std::algorithm can't apply because the children are heterogeneous
// types, not a homogeneous range.  Adding a new node arity needs no change here,
// matching how the rest of the file folds over the operand tuple.
template <typename T> consteval std::size_t node_count_fn() {
  using U = std::remove_cvref_t<T>;
  if constexpr (requires { typename U::children_t; }) {
    return []<typename... C>(std::type_identity<std::tuple<C...>>) consteval {
      return (1 + ... + node_count_fn<std::remove_cvref_t<C>>());
    }(std::type_identity<typename U::children_t>{});
  } else {
    return 1;
  }
}

template <typename T>
inline constexpr std::size_t node_count_v = node_count_fn<std::remove_cvref_t<T>>();

template <std::size_t Base>
inline constexpr std::size_t child_base_v = Base + 1;
template <std::size_t Base, typename LHS>
inline constexpr std::size_t rhs_base_v = Base + 1 + node_count_v<LHS>;

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

// The {value, derivative} get<>() interface — tuple_size is 2 for any node,
// independent of operand arity.
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
