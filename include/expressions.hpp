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

template <typename T> struct is_expression_type : std::false_type {};
template <typename T>
concept CExpression = is_expression_type<std::remove_cvref_t<T>>::value;

template <COperation Op, CExpression... Children> class Expression;

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
// Node counting + preorder cache offsets.
//
// Every node — internal or leaf — gets one cache slot.  A node at preorder slot
// `Base` owns slot `Base`; its I-th child's subtree starts after this node and
// every earlier sibling's subtree (child_base_at).  The forward fill
// (fill_cache) and the reverse sweep (ExpressionOps::backward) both derive child
// slots through child_base_at, so they always agree.
//
// node_count is one rule: 1 + the sum over the operand pack (children_t).  A
// leaf has no children_t, so the `requires` falls through to 1.
// ===========================================================================
template <typename T> consteval std::size_t node_count_fn() {
  using U = std::remove_cvref_t<T>;
  if constexpr (requires { typename U::children_t; }) {
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

// Preorder cache slot of the I-th child of a node at `Base`: skip this node (the
// +1) and every earlier sibling's whole subtree.
template <std::size_t Base, typename Kids, std::size_t I>
consteval std::size_t child_base_at() {
  std::size_t off = Base + 1;
  [&]<std::size_t... K>(std::index_sequence<K...>) {
    ((off += node_count_v<std::tuple_element_t<K, Kids>>), ...);
  }(std::make_index_sequence<I>{});
  return off;
}

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

  // Reverse sweep.  `Base` is this node's preorder slot.  The op only supplies
  // the per-child adjoint (incoming adj times the local partial, read from the
  // value cache); the offset arithmetic and the recursion into each child are
  // the same for every arity and live here, once.
  template <std::size_t Base = 0>
  constexpr void backward(const auto &syms, value_type adj, auto &grads,
                          const auto &cache) const noexcept {
    using Kids = typename Derived::children_t;
    [&]<std::size_t... I>(std::index_sequence<I...>) noexcept {
      // Each child's adjoint is consumed exactly once, so move it into the
      // recursive call (a no-op for the trivially-copyable scalar/dual value
      // types, but correct intent and free if value_type ever grows heavier).
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

template <COperation Op, CExpression... Children>
class Expression : public ExpressionOps<Expression<Op, Children...>, Op> {
  std::tuple<Children...> operands;
  friend std::ostream &operator<<(std::ostream &out, const Expression &e) {
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

  constexpr Expression(Children... c) noexcept : operands{std::move(c)...} {}
  [[nodiscard]] constexpr const children_t &expressions() const noexcept {
    return operands;
  }
  [[nodiscard]] constexpr children_t &expressions() noexcept {
    return operands;
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
template <diff::COperation Op, diff::CExpression... Children>
struct tuple_size<diff::Expression<Op, Children...>>
    : integral_constant<size_t, 2> {};

template <size_t I, diff::COperation Op, diff::CExpression... Children>
struct tuple_element<I, diff::Expression<Op, Children...>> {
  using type = typename diff::detail::expression_element<
      typename diff::Expression<Op, Children...>::value_type, I>::type;
};
} // namespace std
