#pragma once
// The scalar vocabulary the whole library is written against, and the only
// thing `ops/` needs from below it.  Nothing here knows what an expression tree
// is: `Numeric` admits a double, a dual, a matrix or a quaternion alike, and
// every operation descriptor in operations.hpp and unary_math.hpp is written
// against that concept rather than against a scalar type.  That is what lets
// the compile-time graph, the runtime graph and the JIT share one set of rules.

#include <algorithm>
#include <concepts>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>

namespace ddx::impl {


template <typename T>
concept CArithmetic = std::integral<std::remove_cvref_t<T>> ||
                      std::floating_point<std::remove_cvref_t<T>>;

// Closed under the four arithmetic operators and negation.
template <typename T>
// constructible_from<T, int>: differentiation manufactures exactly 0 and 1.
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

// Declared, not deduced: Numeric admits matrices and quaternions, and the
// default is the safe answer.  Only simplify.hpp reads it.  Specialise directly
// for a class template; DDX_COMMUTATIVE_MULTIPLY below covers concrete types.
template <typename T>
inline constexpr bool is_commutative_multiply_v = CArithmetic<T>;

template <typename T>
concept CCommutativeMultiply =
    is_commutative_multiply_v<std::remove_cvref_t<T>>;

// Decomposable into {value, derivative} via std::tuple_size / get<I>.
template <typename T>
concept CTupleLike =
    requires { std::tuple_size<std::remove_cvref_t<T>>::value; };

// Every sweep threads two of these: the point and the node-value cache.
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

template <typename F, typename Seq>
inline constexpr bool pack_invocable_v = false;
template <typename F, std::size_t... Is>
inline constexpr bool pack_invocable_v<F, std::index_sequence<Is...>> =
    requires(F &&f) { static_cast<F &&>(f).template operator()<Is...>(); };
} // namespace detail

// The two shapes every 0..N expansion in this library takes.  static_for hands
// over one index at a time, for a fold of statements; index_apply hands over
// the whole pack at once, for the far commoner case of building one expression
// out of it.  Both exist so that `std::make_index_sequence<N>{}` is not spelled
// at each of them.
template <std::size_t N, typename F>
  requires detail::index_invocable_v<F, std::make_index_sequence<N>>
constexpr void static_for(F &&f) noexcept {
  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    (static_cast<F &&>(f).template operator()<Is>(), ...);
  }(std::make_index_sequence<N>{});
}

template <std::size_t N, typename F>
  requires detail::pack_invocable_v<F, std::make_index_sequence<N>>
[[nodiscard]] constexpr decltype(auto) index_apply(F &&f) {
  return [&]<std::size_t... Is>(std::index_sequence<Is...>) -> decltype(auto) {
    return static_cast<F &&>(f).template operator()<Is...>();
  }(std::make_index_sequence<N>{});
}

template <typename O>
concept COperation =
    requires { typename O::value_type; } && Numeric<typename O::value_type>;

template <typename T> inline constexpr bool is_expression_type_v = false;
template <typename T>
concept CExpression = is_expression_type_v<std::remove_cvref_t<T>>;

// The expression tree's vocabulary, declared here rather than with its
// definitions in expr/expressions.hpp.  The operation descriptors below are
// written against these names -- the sign and power rules return Lit<T, 0>,
// Lit<T, 1>, Lit<T, 2>, and the abs/max/min rules build a MonoExpression over
// SignOp -- so a declaration is all they need to be *written*.  Only a
// translation unit that instantiates them needs the tree itself, and that is
// the compile-time layer, which has it.  The runtime graph reuses the same
// descriptors through their functors and instantiates none of these.
template <COperation Op, CExpression... Children> class Expression;

// Expression's storage base, and the CRTP `Derived`.
template <bool Stateless, COperation Op, CExpression... Children>
class ExpressionImpl;

template <COperation Op, CExpression Exp>
using MonoExpression = Expression<Op, Exp>;

template <Numeric T, auto... V> class Lit;
template <Numeric T> using Constant = Lit<T>;


namespace detail {
// Same fold as binomial() in md/layouts.hpp.
template <CArithmetic T> consteval T compile_time_factorial(T Order) {
  return std::ranges::fold_left(std::views::iota(T{1}, T(Order + 1)), T{1},
                                std::multiplies{});
}
static_assert(compile_time_factorial(3) == 6);
static_assert(compile_time_factorial(5) == 120);
static_assert(compile_time_factorial(7) == 5040);
} // namespace detail

} // namespace ddx::impl
