#pragma once
#include "dual/dual.hpp"
#include "expr/expressions.hpp"
#include "expr/operations.hpp"
#include "expr/simplify.hpp"
#include "util/mpl.hpp"
#include <utility>

namespace diff::impl {

// Both sides carry the same value_type.
template <typename LHS, typename RHS>
concept CSameValueType =
    std::same_as<typename LHS::value_type, typename RHS::value_type>;

// ... or at least one side's value_type converts to the other's.
template <typename LHS, typename RHS>
concept CompatibleValueTypes =
    CSameValueType<LHS, RHS> ||
    std::convertible_to<typename LHS::value_type, typename RHS::value_type> ||
    std::convertible_to<typename RHS::value_type, typename LHS::value_type>;

template <CFixedString auto S, CSymbolList SymList>
consteval std::size_t find_index_of_symbol() noexcept {
  return diff::impl::mpl::mp_find<S>(SymList{});
}

// Promote a bare scalar into value_type as a zero-derivative constant;
// ConstantEmbedder recurses through every Dual<> nesting level.
template <Numeric VT, CArithmetic S>
constexpr Constant<VT> promote_scalar(S s) noexcept {
  return Constant<VT>{
      ConstantEmbedder<VT>::embed(static_cast<scalar_base_t<VT>>(s))};
}

// Each arithmetic operator in its three shapes: expression OP expression (with
// the folding ladder) and the two scalar-promoted forms.  Only the node branch
// differs, so that is the macro argument.
//
// Folding keeps a literal-only subtree from becoming a node.  A T-valued
// template argument exists only for structural T, so a dual scalar can put back
// only 0 and 1 (the int spelling) -- which are the only values differentiation
// manufactures.
#define DIFF_EXPR_BINOP(OP, ...)                                               \
  template <CExpression LHS, CExpression RHS>                                  \
    requires CompatibleValueTypes<LHS, RHS>                                    \
  constexpr auto operator OP(const LHS &a, const RHS &b) noexcept {            \
    using value_type = typename LHS::value_type;                               \
    if constexpr (CLit<LHS> && CLit<RHS>) {                                    \
      constexpr auto folded = std::remove_cvref_t<LHS>::value OP               \
                              std::remove_cvref_t<RHS>::value;                 \
      if constexpr (CArithmetic<value_type>) {                                 \
        return Lit<value_type, static_cast<value_type>(folded)>{};             \
      } else if constexpr (folded == value_type(0)) {                          \
        return Lit<value_type, 0>{};                                           \
      } else if constexpr (folded == value_type(1)) {                          \
        return Lit<value_type, 1>{};                                           \
      } else {                                                                 \
        return Constant<value_type>{folded};                                   \
      }                                                                        \
    } else if constexpr (CConstant<LHS> && CConstant<RHS>) {                   \
      return Constant<value_type>{a.get() OP b.get()};                         \
    } else {                                                                   \
      __VA_ARGS__                                                              \
    }                                                                          \
  }                                                                            \
  template <CArithmetic S, CExpression RHS>                                    \
  constexpr auto operator OP(S s, const RHS &b) noexcept {                     \
    return promote_scalar<typename RHS::value_type>(s) OP b;                   \
  }                                                                            \
  template <CExpression LHS, CArithmetic S>                                    \
  constexpr auto operator OP(const LHS &a, S s) noexcept {                     \
    return a OP promote_scalar<typename LHS::value_type>(s);                   \
  }

DIFF_EXPR_BINOP(+, return detail::simplify_node<SumOp<value_type>>(a, b);)
DIFF_EXPR_BINOP(*, return detail::simplify_node<MultiplyOp<value_type>>(a, b);)
DIFF_EXPR_BINOP(/, return detail::simplify_node<DivideOp<value_type>>(a, b);)
// a - b is a + (-b), so the reverse sweep needs one adjoint rule instead of
// two, and both operators get to apply their folding rules.
DIFF_EXPR_BINOP(-, return detail::simplify_node<SumOp<value_type>>(a, -b);)
#undef DIFF_EXPR_BINOP

template <CExpression Expr> constexpr auto operator-(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  if constexpr (CLit<Expr> && CArithmetic<value_type>) {
    return Lit<value_type,
               static_cast<value_type>(-std::remove_cvref_t<Expr>::value)>{};
  } else {
    return detail::simplify_mono<NegateOp<value_type>>(a);
  }
}

// One factory per unary math function, generated from the registry.
#define DIFF_EXPR_UNFN(FN, OP, LABEL)                                          \
  template <CExpression Expr> constexpr auto FN(const Expr &a) noexcept {      \
    return MonoExpression<OP<typename Expr::value_type>, Expr>{a};             \
  }
DIFF_UNARY_MATH_TABLE(DIFF_EXPR_UNFN)
// abs has no descriptor, so its factory is spelled out.
DIFF_EXPR_UNFN(abs, AbsOp, "abs")
#undef DIFF_EXPR_UNFN

// Function-style binary ops, plus scalar-promotion overloads.
#define DIFF_EXPR_BINFN(NAME, OP)                                              \
  template <CExpression LHS, CExpression RHS>                                  \
    requires CompatibleValueTypes<LHS, RHS>                                    \
  constexpr auto NAME(const LHS &a, const RHS &b) noexcept {                   \
    using value_type = typename LHS::value_type;                               \
    return Expression<OP<value_type>, LHS, RHS>{a, b};                         \
  }                                                                            \
  template <CArithmetic S, CExpression RHS>                                    \
  constexpr auto NAME(S s, const RHS &b) noexcept {                            \
    return NAME(promote_scalar<typename RHS::value_type>(s), b);               \
  }                                                                            \
  template <CExpression LHS, CArithmetic S>                                    \
  constexpr auto NAME(const LHS &a, S s) noexcept {                            \
    return NAME(a, promote_scalar<typename LHS::value_type>(s));               \
  }
DIFF_EXPR_BINFN(pow, PowOp)
DIFF_EXPR_BINFN(atan2, Atan2Op)
DIFF_EXPR_BINFN(hypot, HypotOp)
DIFF_EXPR_BINFN(max, MaxOp)
DIFF_EXPR_BINFN(min, MinOp)
#undef DIFF_EXPR_BINFN

namespace detail {

// Both forms of constant answer every sweep identically, so read() is the only
// member a specialisation supplies.  CRTP rather than a data member: the
// compile-time form has to stay std::is_empty_v, which is what selects the
// stateless node storage, and a member defeats it even when empty.
template <typename Derived, Numeric T>
class ConstantOps : public EquationConvertible<Derived> {
  [[nodiscard]] constexpr const Derived &self() const noexcept {
    return static_cast<const Derived &>(*this);
  }

public:
  using value_type = T;

  [[nodiscard]] constexpr T get() const noexcept { return self().read(); }
  [[nodiscard]] constexpr T eval() const noexcept { return get(); }
  constexpr operator T() const noexcept { return get(); }

  // The int spelling of Lit works for every Numeric T.
  [[nodiscard]] constexpr auto derivative() const noexcept {
    return Lit<T, 0>{};
  }

  template <std::size_t Base = 0>
  constexpr void backward(const auto &, T, auto &,
                          const auto &) const noexcept {}

  // At the constant's own type the value passes through verbatim; at a deeper
  // type ConstantEmbedder<U> embeds it with zero dual parts.
  template <CSymbolList Syms, Numeric U, std::size_t N>
  [[nodiscard]] constexpr U
  eval_seeded(const std::array<U, N> &) const noexcept {
    if constexpr (std::same_as<U, T>) {
      return get();
    } else {
      using S = scalar_base_t<U>;
      return ConstantEmbedder<U>::embed(
          static_cast<S>(get_real_part<dual_depth_v<T>>(get())));
    }
  }

  template <std::size_t I> [[nodiscard]] constexpr auto get() const noexcept {
    static_assert(I < 2);
    if constexpr (CTupleLike<T>) {
      return eval().template get<I>();
    } else if constexpr (I == 0) {
      return eval();
    } else {
      return static_cast<T>(derivative());
    }
  }

  template <CEvalArg... Args>
    requires(sizeof...(Args) > 0)
  [[nodiscard]] constexpr auto eval(const Args &...args) const {
    return detail::eval_dispatch(self(), args...);
  }

  template <FixedString Seed, CEvalArg... Args>
  [[nodiscard]] constexpr auto eval_with_tangent(const Args &...args) const {
    return detail::tangent_dispatch<Seed>(self(), args...);
  }
};

} // namespace detail

// A literal carried in the type, so the object is empty -- which is what makes
// the 0s and 1s derivative() manufactures free.
template <Numeric T, auto V>
  requires std::same_as<std::remove_cv_t<decltype(V)>, T>
class Lit<T, V> : public detail::ConstantOps<Lit<T, V>, T> {
  friend detail::ConstantOps<Lit<T, V>, T>;
  [[nodiscard]] constexpr T read() const noexcept { return V; }

public:
  static constexpr T value = V;
};

// Keyed on an int, for a T that could never be a template argument itself.
// Disjoint from the specialisation above, which requires decltype(V) to be T.
template <Numeric T, auto V>
  requires(std::same_as<std::remove_cv_t<decltype(V)>, int> &&
           !std::same_as<T, int>)
class Lit<T, V> : public detail::ConstantOps<Lit<T, V>, T> {
  friend detail::ConstantOps<Lit<T, V>, T>;
  [[nodiscard]] constexpr T read() const noexcept { return T(V); }

public:
  static constexpr T value = T(V);
};

// The storing form: one type per value_type, so it can hold a runtime number.
template <Numeric T> class Lit<T> : public detail::ConstantOps<Lit<T>, T> {
  friend detail::ConstantOps<Lit<T>, T>;
  T value;
  [[nodiscard]] constexpr T read() const noexcept { return value; }

public:
  constexpr explicit Lit(T value) noexcept : value(value) {}
};

template <Numeric T> Lit(T) -> Lit<T>;

// `Frozen`: still reads its slot from the seed array, but differentiates to
// zero.
template <Numeric T, CFixedString auto symbol, bool Frozen>
class Variable : public EquationConvertible<Variable<T, symbol, Frozen>> {
public:
  static constexpr auto label = symbol;
  static constexpr bool frozen = Frozen;
  using value_type = T;

  [[nodiscard]] constexpr auto derivative() const noexcept {
    return Lit<T, Frozen ? 0 : 1>{};
  }

  template <std::size_t Base = 0>
  constexpr void backward(const auto &syms, T adj, auto &grads,
                          const auto &) const noexcept {
    if constexpr (!Frozen) {
      using Syms = std::decay_t<decltype(syms)>;
      constexpr auto idx = find_index_of_symbol<symbol, Syms>();
      // `+` and assignment, not `+=`: CFieldLike promises only a + b.  Do not
      // "simplify" this back to `+=`.
      grads[idx] = std::move(grads[idx]) + adj;
    }
  }

  // A frozen symbol takes the seed's value and drops its derivative slots.
  // All three engines guard on Frozen or they disagree.
  template <CSymbolList Syms, Numeric U, std::size_t N>
  [[nodiscard]] constexpr U
  eval_seeded(const std::array<U, N> &vals) const noexcept {
    constexpr auto idx = find_index_of_symbol<symbol, Syms>();
    static_assert(idx < N, "eval: no value supplied for this symbol");
    if constexpr (Frozen) {
      return ConstantEmbedder<U>::embed(get_real_part<dual_depth_v<U>>(vals[idx]));
    } else {
      return vals[idx];
    }
  }

  [[nodiscard]] constexpr auto eval(const CEvalArg auto &...args) const
    requires(sizeof...(args) > 0)
  {
    return detail::eval_dispatch(*this, args...);
  }

  template <FixedString Seed>
  [[nodiscard]] constexpr auto
  eval_with_tangent(const CEvalArg auto &...args) const {
    return detail::tangent_dispatch<Seed>(*this, args...);
  }
};

#define DEFINE_CONST_UDL(type, suffix)                                         \
  consteval diff::impl::Constant<type> operator""_##suffix(unsigned long long val) { \
    return diff::impl::Constant<type>{static_cast<type>(val)};                       \
  }                                                                            \
  consteval diff::impl::Constant<type> operator""_##suffix(long double val) {        \
    return diff::impl::Constant<type>{static_cast<type>(val)};                       \
  }

// The literal's value is unused; only its type selects value_type.
#define DEFINE_VAR_UDL(type, suffix, label)                                    \
  consteval auto operator""_##suffix(unsigned long long) {                     \
    return diff::impl::Variable<type, diff::impl::FixedString{label}>{};                   \
  }                                                                            \
  consteval auto operator""_##suffix(long double) {                            \
    return diff::impl::Variable<type, diff::impl::FixedString{label}>{};                   \
  }

// var<"x"> — name a symbol.
template <FixedString S, Numeric T = double>
inline constexpr Variable<T, S> var{};

} // namespace diff::impl

DEFINE_CONST_UDL(int, ci)
DEFINE_CONST_UDL(double, cd)
DEFINE_VAR_UDL(int, vi, "c")
DEFINE_VAR_UDL(double, vd, "v")

namespace std {
template <diff::impl::Numeric T, auto... V>
struct tuple_size<diff::impl::Lit<T, V...>> : integral_constant<std::size_t, 2> {};

template <std::size_t I, diff::impl::Numeric T, auto... V>
struct tuple_element<I, diff::impl::Lit<T, V...>> {
  using type = typename diff::impl::detail::expression_element<T, I>::type;
};

template <diff::impl::Numeric T, diff::impl::CFixedString auto C, bool F>
struct tuple_size<diff::impl::Variable<T, C, F>> : integral_constant<std::size_t, 2> {
};

template <std::size_t I, diff::impl::Numeric T, diff::impl::CFixedString auto C, bool F>
struct tuple_element<I, diff::impl::Variable<T, C, F>> {
  using type = typename diff::impl::detail::expression_element<T, I>::type;
};
} // namespace std

#define PDV(x, label)                                                          \
  diff::impl::Variable<diff::impl::Dual<std::decay_t<decltype(x)>>,                        \
                 diff::impl::FixedString{label}> {}
#define PV(x, label)                                                           \
  diff::impl::Variable<std::decay_t<decltype(x)>, diff::impl::FixedString{label}> {}
// Lit(x), not Constant(x): deducing through an alias template is P1814, which
// Clang lacks before 19.  Same resulting type.
#define PC(x) diff::impl::Lit(x)
