#pragma once
#include "dual/dual.hpp"
#include "expr/expressions.hpp"
#include "expr/operations.hpp"
#include "expr/simplify.hpp"
#include "util/mpl.hpp"
#include <utility>

namespace diff {

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
  return diff::mpl::mp_find<S>(SymList{});
}

// Promote a bare scalar `s` into the expression's value_type as a
// zero-derivative constant.  ConstantEmbedder recurses through every Dual<>
// nesting level, so this is correct even when VT is a multi-level dual (e.g.
// Dual<Dual<double>>); for non-dual VT it is just a cast.
template <Numeric VT, CArithmetic S>
constexpr Constant<VT> promote_scalar(S s) noexcept {
  return Constant<VT>{
      ConstantEmbedder<VT>::embed(static_cast<scalar_base_t<VT>>(s))};
}

// The four arithmetic operators, each in its three shapes: expression OP
// expression (with the constant-folding ladder) and the two scalar-promoted
// forms.  Only the node branch differs, so that is the macro argument.
//
// Folding keeps a literal-only subtree from ever becoming a node: two Lits fold
// to a Lit at compile time, two Constants to a Constant.  A T-valued template
// argument only exists for structural T, so for a dual scalar only 0 and 1 can
// go back into the type (via the int spelling) -- but those are the only values
// differentiation manufactures, and checking for them is what keeps a
// literal-only dual subtree empty rather than storing.
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
// Subtraction has no node of its own: a - b is a + (-b), so the reverse sweep
// needs one adjoint rule instead of two.  Spelling it with the operators lets
// both apply their rules -- a - 0 is a, a - (-b) is a + b.
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

// One expression factory per unary math function, generated from the registry
// in expr/unary_math.hpp so the name list lives in exactly one place.
#define DIFF_EXPR_UNFN(FN, OP, LABEL)                                          \
  template <CExpression Expr> constexpr auto FN(const Expr &a) noexcept {      \
    return MonoExpression<OP<typename Expr::value_type>, Expr>{a};             \
  }
DIFF_UNARY_MATH_TABLE(DIFF_EXPR_UNFN)
// abs has no descriptor (its derivative is a sign), so its factory is spelled
// out; AbsOp itself is likewise hand-written.
DIFF_EXPR_UNFN(abs, AbsOp, "abs")
#undef DIFF_EXPR_UNFN

// Function-style binary ops (pow, atan2, hypot, max, min): an all-expression
// form plus scalar-promotion overloads so either operand may be a bare scalar.
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

// The constant-leaf protocol, written once.  Both forms of constant answer every
// sweep identically -- the value, and a derivative of zero -- so read() is the
// only member the specialisations supply.
//
// CRTP rather than a data member, because the compile-time form has to stay
// std::is_empty_v: that is what selects the stateless node storage in
// expressions.hpp, and a member defeats it even when it is itself empty and
// [[no_unique_address]].
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

  // The int spelling of Lit works for every Numeric T, so a constant's
  // derivative is empty whatever it is a constant of.
  [[nodiscard]] constexpr auto derivative() const noexcept {
    return Lit<T, 0>{};
  }

  // Reverse sweep leaf: no symbol underneath, so nothing to accumulate into.
  template <std::size_t Base = 0>
  constexpr void backward(const auto &, T, auto &,
                          const auto &) const noexcept {}

  // Seeded sweep leaf.  At the constant's own type the stored value passes
  // through verbatim, dual parts and all; at a deeper type it is embedded with
  // zero dual parts via ConstantEmbedder<U>, which a custom numeric type may
  // specialise.
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

  // Leaves are expressions too: same eval(...) surface as ExpressionOps.
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

// A literal carried in the type, so the object is empty.  This is what
// derivative() manufactures: the 0s and 1s that flood a Jacobian cost nothing,
// and a node whose children are all empty stores nothing.
template <Numeric T, auto V>
  requires std::same_as<std::remove_cv_t<decltype(V)>, T>
class Lit<T, V> : public detail::ConstantOps<Lit<T, V>, T> {
  friend detail::ConstantOps<Lit<T, V>, T>;
  [[nodiscard]] constexpr T read() const noexcept { return V; }

public:
  static constexpr T value = V;
};

// The same thing keyed on an int, for a T that could never be a template
// argument itself (Dual and friends are not structural).  0 and 1 are exact in
// every Numeric T, so this keeps the dual-valued trees empty too.  Disjoint from
// the specialisation above: that one requires decltype(V) to be T, this one
// requires int and a T that is not int.
template <Numeric T, auto V>
  requires(std::same_as<std::remove_cv_t<decltype(V)>, int> &&
           !std::same_as<T, int>)
class Lit<T, V> : public detail::ConstantOps<Lit<T, V>, T> {
  friend detail::ConstantOps<Lit<T, V>, T>;
  [[nodiscard]] constexpr T read() const noexcept { return T(V); }

public:
  static constexpr T value = T(V);
};

// The storing form: one type per value_type rather than one per value, so it can
// hold a number that only exists at run time.
template <Numeric T> class Lit<T> : public detail::ConstantOps<Lit<T>, T> {
  friend detail::ConstantOps<Lit<T>, T>;
  T value;
  [[nodiscard]] constexpr T read() const noexcept { return value; }

public:
  constexpr explicit Lit(T value) noexcept : value(value) {}
};

template <Numeric T> Lit(T) -> Lit<T>;

// `Frozen` marks a variable that has been held constant for the purpose of
// partial differentiation: it still reads its value from the seed array like
// any other symbol, but its derivative is zero.
template <Numeric T, CFixedString auto symbol, bool Frozen>
class Variable : public EquationConvertible<Variable<T, symbol, Frozen>> {
public:
  static constexpr auto label = symbol;
  static constexpr bool frozen = Frozen;
  using value_type = T;

  // The int spelling of Lit covers every Numeric T, so no structural fork.
  [[nodiscard]] constexpr auto derivative() const noexcept {
    return Lit<T, Frozen ? 0 : 1>{};
  }

  template <std::size_t Base = 0>
  constexpr void backward(const auto &syms, T adj, auto &grads,
                          const auto &) const noexcept {
    if constexpr (!Frozen) {
      using Syms = std::decay_t<decltype(syms)>;
      constexpr auto idx = find_index_of_symbol<symbol, Syms>();
      // `+` and assignment, not `+=`: CFieldLike asks for a + b and says
      // nothing about compound assignment.  std::move needs no guard -- without
      // a move constructor the rvalue binds to operator+'s const& or selects the
      // copy constructor.  Do not "simplify" this back to `+=`.
      grads[idx] = std::move(grads[idx]) + adj;
    }
  }

  // Seeded sweep leaf: read this symbol's slot, whatever the seed type is.  A
  // frozen symbol is a constant -- same value lookup, zero derivative -- so it
  // takes the seed's value and drops its derivative slots, exactly as
  // Constant::eval_seeded embeds a bare scalar.  All three engines guard on
  // Frozen or they disagree.
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

  // Leaves are expressions too: same eval(...) surface as ExpressionOps.
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
  consteval diff::Constant<type> operator""_##suffix(unsigned long long val) { \
    return diff::Constant<type>{static_cast<type>(val)};                       \
  }                                                                            \
  consteval diff::Constant<type> operator""_##suffix(long double val) {        \
    return diff::Constant<type>{static_cast<type>(val)};                       \
  }

// A variable is a pure symbol, so the literal's value is unused; only its type
// selects the variable's value_type.
#define DEFINE_VAR_UDL(type, suffix, label)                                    \
  consteval auto operator""_##suffix(unsigned long long) {                     \
    return diff::Variable<type, diff::FixedString{label}>{};                   \
  }                                                                            \
  consteval auto operator""_##suffix(long double) {                            \
    return diff::Variable<type, diff::FixedString{label}>{};                   \
  }

// var<"x"> — name a symbol.
template <FixedString S, Numeric T = double>
inline constexpr Variable<T, S> var{};

} // namespace diff

DEFINE_CONST_UDL(int, ci)
DEFINE_CONST_UDL(double, cd)
DEFINE_VAR_UDL(int, vi, "c")
DEFINE_VAR_UDL(double, vd, "v")

namespace std {
template <diff::Numeric T, auto... V>
struct tuple_size<diff::Lit<T, V...>> : integral_constant<std::size_t, 2> {};

template <std::size_t I, diff::Numeric T, auto... V>
struct tuple_element<I, diff::Lit<T, V...>> {
  using type = typename diff::detail::expression_element<T, I>::type;
};

template <diff::Numeric T, diff::CFixedString auto C, bool F>
struct tuple_size<diff::Variable<T, C, F>> : integral_constant<std::size_t, 2> {
};

template <std::size_t I, diff::Numeric T, diff::CFixedString auto C, bool F>
struct tuple_element<I, diff::Variable<T, C, F>> {
  using type = typename diff::detail::expression_element<T, I>::type;
};
} // namespace std

#define PDV(x, label)                                                          \
  diff::Variable<diff::Dual<std::decay_t<decltype(x)>>,                        \
                 diff::FixedString{label}> {}
#define PV(x, label)                                                           \
  diff::Variable<std::decay_t<decltype(x)>, diff::FixedString{label}> {}
// diff::Lit(x), not diff::Constant(x): deducing through an alias template is
// P1814, which Clang lacks before 19.  Same resulting type, Lit<T> == Constant<T>.
#define PC(x) diff::Lit(x)
