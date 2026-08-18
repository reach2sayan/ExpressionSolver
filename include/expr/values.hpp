#pragma once
#include "dual/dual.hpp"
#include "expr/expressions.hpp"
#include "expr/operations.hpp"
#include "util/mpl.hpp"
#include <format>

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
// expression (with the constant-folding ladder), and the two scalar-promoted
// forms.  Only the node branch differs between them, so that is the argument --
// everything else was four, then eight, copies of the same text.
//
// Folding is what keeps a literal-only subtree from ever becoming a node: two
// Lits fold to a Lit at compile time, two Constants to a Constant.
#define DIFF_EXPR_BINOP(OP, ...)                                               \
  template <CExpression LHS, CExpression RHS>                                  \
    requires CompatibleValueTypes<LHS, RHS>                                    \
  constexpr auto operator OP(const LHS &a, const RHS &b) noexcept {            \
    using value_type = typename LHS::value_type;                               \
    if constexpr (CLit<LHS> && CLit<RHS>) {                                    \
      return Lit<value_type,                                                   \
                 static_cast<value_type>(std::remove_cvref_t<LHS>::value OP    \
                                         std::remove_cvref_t<RHS>::value)>{};  \
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

DIFF_EXPR_BINOP(+, return Expression<SumOp<value_type>, LHS, RHS>{a, b};)
DIFF_EXPR_BINOP(*, return Expression<MultiplyOp<value_type>, LHS, RHS>{a, b};)
DIFF_EXPR_BINOP(/, return Expression<DivideOp<value_type>, LHS, RHS>{a, b};)
// Subtraction has no node of its own: a - b is a + (-b), which is what keeps
// the reverse sweep down to one adjoint rule instead of two.
DIFF_EXPR_BINOP(-,
                auto neg = MonoExpression<NegateOp<value_type>, RHS>{b};
                return Expression<SumOp<value_type>, LHS, decltype(neg)>{
                    a, std::move(neg)};)
#undef DIFF_EXPR_BINOP

template <CExpression Expr> constexpr auto operator-(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  if constexpr (CLit<Expr>) {
    return Lit<value_type,
               static_cast<value_type>(-std::remove_cvref_t<Expr>::value)>{};
  } else {
    return MonoExpression<NegateOp<value_type>, Expr>{a};
  }
}

// One expression factory per unary math function, generated from the registry
// in expr/unary_math.hpp so the name list lives in exactly one place.
#define DIFF_EXPR_UNFN(FN, OP, LABEL)                                          \
  template <CExpression Expr> constexpr auto FN(const Expr &a) noexcept {      \
    return MonoExpression<OP<typename Expr::value_type>, Expr>{a};             \
  }
DIFF_UNARY_MATH_TABLE(DIFF_EXPR_UNFN)
// abs is not in the registry (no descriptor -- its derivative is a sign), so
// its factory is spelled out; AbsOp itself is likewise hand-written.
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

// The constant-leaf protocol, written once.
//
// Both forms of constant answer every sweep identically — the value, and a
// derivative of zero — so the only thing that differs is where the value comes
// from.  That is the single member the specialisations below supply (read());
// everything else lives here.
//
// CRTP rather than a data member holding the value, because the compile-time
// form has to stay std::is_empty_v: that predicate is what selects the
// stateless node storage in expressions.hpp, and a member would defeat it even
// if the member were itself empty and marked [[no_unique_address]] — occupying
// no bytes and being no member are different things, and only the second one
// satisfies is_empty_v.
template <typename Derived, Numeric T> class ConstantOps {
  [[nodiscard]] constexpr const Derived &self() const noexcept {
    return static_cast<const Derived &>(*this);
  }

  // Hidden friend on Derived, not on this base: ADL still finds it (a class's
  // associated entities include its bases), and taking the leaf type exactly
  // keeps it clear of the operator T() conversion below.
  friend std::ostream &operator<<(std::ostream &out, const Derived &c) {
    return out << std::format("{}", c.get());
  }

public:
  using value_type = T;

  [[nodiscard]] constexpr T get() const noexcept { return self().read(); }
  [[nodiscard]] constexpr T eval() const noexcept { return get(); }
  constexpr operator T() const noexcept { return get(); }

  // Lit carries its value as a template argument, so it is only available for
  // structural types; a dual-valued constant falls back to a stored zero.
  [[nodiscard]] constexpr auto derivative() const noexcept {
    if constexpr (CArithmetic<T>) {
      return Lit<T, T{0}>{};
    } else {
      return Constant<T>{T{0}};
    }
  }

  // Reverse sweep leaf: no symbol underneath, so nothing to accumulate into.
  template <std::size_t Base = 0>
  constexpr void backward(const auto &, T, auto &,
                          const auto &) const noexcept {}

  // Seeded sweep leaf.  When the seed type is the constant's own type there is
  // nothing to convert and the stored value passes through verbatim — dual
  // parts and all.  Seeded with a deeper type, the constant is embedded with
  // zero dual parts via ConstantEmbedder<U>, so custom numeric types (e.g.
  // TaylorDual) can specialise the embedding without touching this code.
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
// and a node all of whose children are empty stores nothing at all.
template <Numeric T, auto V>
  requires std::same_as<std::remove_cv_t<decltype(V)>, T>
class Lit<T, V> : public detail::ConstantOps<Lit<T, V>, T> {
  friend detail::ConstantOps<Lit<T, V>, T>;
  [[nodiscard]] constexpr T read() const noexcept { return V; }

public:
  static constexpr T value = V;
};

// The storing form.  One type per value_type rather than one per value, which
// is what lets it hold a number that only exists at run time — and a T that
// could never be a template argument in the first place.
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
template <Numeric T, CFixedString auto symbol, bool Frozen> class Variable {
  // A Variable is stateless, so the label is all there is to print.
  friend std::ostream &operator<<(std::ostream &out,
                                  const Variable<T, symbol, Frozen> &) {
    return out << symbol.view();
  }

public:
  static constexpr auto label = symbol;
  static constexpr bool frozen = Frozen;
  using value_type = T;

  // Lit carries its value as a template argument, so it is only available for
  // structural types; a dual-valued variable falls back to a stored Constant.
  [[nodiscard]] constexpr auto derivative() const noexcept {
    if constexpr (CArithmetic<T>) {
      if constexpr (Frozen) {
        return Lit<T, T{0}>{};
      } else {
        return Lit<T, T{1}>{};
      }
    } else {
      return Constant<T>{Frozen ? T{0} : T{1}};
    }
  }

  template <std::size_t Base = 0>
  constexpr void backward(const auto &syms, T adj, auto &grads,
                          const auto &) const noexcept {
    if constexpr (!Frozen) {
      using Syms = std::decay_t<decltype(syms)>;
      constexpr auto idx = find_index_of_symbol<symbol, Syms>();
      grads[idx] += adj;
    }
  }

  // Seeded sweep leaf: read this symbol's slot, whatever the seed type is.
  //
  // A frozen symbol is a constant -- same value lookup, zero derivative -- so
  // it takes the seed's value and drops its derivative slots, exactly as
  // Constant::eval_seeded embeds a bare scalar.  Without this the three engines
  // disagree: derivative() and backward() both guard on Frozen, so only this
  // one would have reported a nonzero derivative for a variable that was
  // explicitly frozen.
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
#define PC(x) diff::Constant(x)
