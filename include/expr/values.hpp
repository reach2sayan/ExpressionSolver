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
// expression (with the constant-folding ladder), and the two scalar-promoted
// forms.  Only the node branch differs between them, so that is the argument --
// everything else was four, then eight, copies of the same text.
//
// Folding is what keeps a literal-only subtree from ever becoming a node: two
// Lits fold to a Lit at compile time, two Constants to a Constant.
//
// A T-valued template argument only exists for structural T, so for a dual
// scalar the folded value cannot go back into the type in general -- but 0 and
// 1 can, via the int spelling, and those are the only values differentiation
// manufactures.  Checking for them keeps a literal-only dual subtree empty.
// Without that check the pair falls to the Constant branch below, which stores,
// and the whole spine above it leaves the stateless node form.
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
// Subtraction has no node of its own: a - b is a + (-b), which is what keeps
// the reverse sweep down to one adjoint rule instead of two.  Spelling it with
// the operators rather than by hand means both of them get to apply their
// rules, so a - 0 is a and a - (-b) is a + b.
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

  // The int spelling of Lit works for every Numeric T, dual types included, so
  // a constant's derivative is empty no matter what it is a constant of.
  [[nodiscard]] constexpr auto derivative() const noexcept {
    return Lit<T, 0>{};
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

// The same thing keyed on an int, for a T that could never be a template
// argument itself: Dual and friends are not structural, so Lit<Dual<double>,
// V> above is unspellable.  0 and 1 -- the only values differentiation
// manufactures -- are exact in every Numeric T, so this covers the whole
// dual-valued path and keeps those trees empty too.  Disjoint from the
// specialisation above by construction: that one requires decltype(V) to be T,
// this one requires it to be int and T to be something else.
template <Numeric T, auto V>
  requires(std::same_as<std::remove_cv_t<decltype(V)>, int> &&
           !std::same_as<T, int>)
class Lit<T, V> : public detail::ConstantOps<Lit<T, V>, T> {
  friend detail::ConstantOps<Lit<T, V>, T>;
  [[nodiscard]] constexpr T read() const noexcept { return T(V); }

public:
  static constexpr T value = T(V);
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
template <Numeric T, CFixedString auto symbol, bool Frozen>
class Variable : public EquationConvertible<Variable<T, symbol, Frozen>> {
public:
  static constexpr auto label = symbol;
  static constexpr bool frozen = Frozen;
  using value_type = T;

  // The int spelling of Lit is available for every Numeric T, so this is one
  // line rather than a structural/non-structural fork.
  [[nodiscard]] constexpr auto derivative() const noexcept {
    return Lit<T, Frozen ? 0 : 1>{};
  }

  template <std::size_t Base = 0>
  constexpr void backward(const auto &syms, T adj, auto &grads,
                          const auto &) const noexcept {
    if constexpr (!Frozen) {
      using Syms = std::decay_t<decltype(syms)>;
      constexpr auto idx = find_index_of_symbol<symbol, Syms>();
      // Spelled with + and assignment rather than +=, because that is all
      // CFieldLike actually asks of a scalar -- it requires a + b and says
      // nothing about compound assignment, so a += here quietly demanded more
      // than the concept advertised and a conforming scalar could fail to
      // compile through this line.
      //
      // std::move needs no is_move_constructible guard: on a type without a
      // move constructor the rvalue simply binds to operator+'s const& or
      // selects its copy constructor, so this degrades to a copy on its own.
      // It buys nothing for the scalars shipped today -- Dual, TaylorDual and
      // VectorDual are trivially copyable arrays -- and costs nothing either;
      // it is here for a scalar that owns storage.
      grads[idx] = std::move(grads[idx]) + adj;
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
// diff::Lit(x), not diff::Constant(x): Constant is an alias template, and
// deducing template arguments through one is P1814, which Clang did not
// implement until 19 -- so the alias spelling is a hard error on older Clang
// while the class template it aliases deduces fine everywhere (see the
// deduction guide on Lit above).  Same resulting type, Lit<T> == Constant<T>.
#define PC(x) diff::Lit(x)
