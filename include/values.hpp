#pragma once
#include "dual.hpp"
#include "expressions.hpp"
#include "operations.hpp"
#include <boost/mp11/algorithm.hpp>
#include <concepts>
#include <format>
#include <string_view>

namespace diff {

constexpr bool PRINT_VARIABLE_VALUE = false;
constexpr bool PRINT_VARIABLE_LABEL = true;
constexpr bool PRINT_CONSTANT_VALUE = true;
constexpr bool PRINT_CONSTANT_LABEL = false;

template <typename LHS, typename RHS>
concept CompatibleValueTypes =
    std::is_same_v<typename LHS::value_type, typename RHS::value_type> ||
    std::is_convertible_v<typename LHS::value_type, typename RHS::value_type> ||
    std::is_convertible_v<typename RHS::value_type, typename LHS::value_type>;

template <CFixedString auto S, typename SymList>
consteval std::size_t find_index_of_symbol() noexcept {
  return boost::mp11::mp_find<SymList, symbol_type<S>>::value;
}

template <CExpression LHS, CExpression RHS>
  requires CompatibleValueTypes<LHS, RHS>
constexpr auto operator+(const LHS &a, const RHS &b) noexcept {
  using value_type = typename LHS::value_type;
  if constexpr (is_constant_v<LHS> && is_constant_v<RHS>) {
    return Constant<value_type>{a.get() + b.get()};
  } else {
    return Expression<SumOp<value_type>, LHS, RHS>{a, b};
  }
}

template <CExpression LHS, CExpression RHS>
  requires CompatibleValueTypes<LHS, RHS>
constexpr auto operator*(const LHS &a, const RHS &b) noexcept {
  using value_type = typename LHS::value_type;
  if constexpr (is_constant_v<LHS> && is_constant_v<RHS>) {
    return Constant<value_type>{a.get() * b.get()};
  } else {
    return Expression<MultiplyOp<value_type>, LHS, RHS>{a, b};
  }
}

template <CExpression LHS, CExpression RHS>
  requires CompatibleValueTypes<LHS, RHS>
constexpr auto operator-(const LHS &a, const RHS &b) noexcept {
  using value_type = typename LHS::value_type;
  if constexpr (is_constant_v<LHS> && is_constant_v<RHS>) {
    return Constant<value_type>{a.get() - b.get()};
  } else {
    auto neg = MonoExpression<NegateOp<value_type>, RHS>{b};
    return Expression<SumOp<value_type>, LHS, decltype(neg)>{a, std::move(neg)};
  }
}

template <CExpression LHS, CExpression RHS>
  requires CompatibleValueTypes<LHS, RHS>
constexpr auto operator/(const LHS &a, const RHS &b) noexcept {
  using value_type = typename LHS::value_type;
  if constexpr (is_constant_v<LHS> && is_constant_v<RHS>) {
    return Constant<value_type>{a.get() / b.get()};
  } else {
    return Expression<DivideOp<value_type>, LHS, RHS>{a, b};
  }
}

template <CExpression Expr> constexpr auto operator-(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  return MonoExpression<NegateOp<value_type>, Expr>{a};
}

template <CExpression Expr> constexpr auto sin(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  return MonoExpression<SineOp<value_type>, Expr>{a};
}

template <CExpression Expr> constexpr auto cos(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  return MonoExpression<CosineOp<value_type>, Expr>{a};
}

template <CExpression Expr> constexpr auto exp(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  return MonoExpression<ExpOp<value_type>, Expr>{a};
}

template <CExpression Expr> constexpr auto tan(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  return MonoExpression<TanOp<value_type>, Expr>{a};
}

template <CExpression Expr> constexpr auto log(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  return MonoExpression<LogOp<value_type>, Expr>{a};
}

template <CExpression Expr> constexpr auto sqrt(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  return MonoExpression<SqrtOp<value_type>, Expr>{a};
}

template <CExpression Expr> constexpr auto abs(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  return MonoExpression<AbsOp<value_type>, Expr>{a};
}

template <CExpression Expr> constexpr auto asin(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  return MonoExpression<AsinOp<value_type>, Expr>{a};
}

template <CExpression Expr> constexpr auto acos(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  return MonoExpression<AcosOp<value_type>, Expr>{a};
}

template <CExpression Expr> constexpr auto atan(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  return MonoExpression<AtanOp<value_type>, Expr>{a};
}

template <CExpression Expr> constexpr auto sinh(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  return MonoExpression<SinhOp<value_type>, Expr>{a};
}

template <CExpression Expr> constexpr auto cosh(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  return MonoExpression<CoshOp<value_type>, Expr>{a};
}

template <CExpression Expr> constexpr auto tanh(const Expr &a) noexcept {
  using value_type = typename Expr::value_type;
  return MonoExpression<TanhOp<value_type>, Expr>{a};
}

// Promote a bare scalar `s` into the expression's value_type as a
// zero-derivative constant.  ConstantEmbedder recurses through every Dual<>
// nesting level, so this is correct even when VT is a multi-level dual (e.g.
// Dual<Dual<double>>); for non-dual VT it is just a cast.
template <typename VT, typename S>
constexpr Constant<VT> promote_scalar(S s) noexcept {
  return Constant<VT>{
      ConstantEmbedder<VT>::embed(static_cast<scalar_base_t<VT>>(s))};
}

template <typename S, CExpression RHS>
  requires std::is_arithmetic_v<S>
constexpr auto operator+(S s, const RHS &b) noexcept {
  return promote_scalar<typename RHS::value_type>(s) + b;
}

template <typename S, CExpression RHS>
  requires std::is_arithmetic_v<S>
constexpr auto operator*(S s, const RHS &b) noexcept {
  return promote_scalar<typename RHS::value_type>(s) * b;
}

template <typename S, CExpression RHS>
  requires std::is_arithmetic_v<S>
constexpr auto operator-(S s, const RHS &b) noexcept {
  return promote_scalar<typename RHS::value_type>(s) - b;
}

template <typename S, CExpression RHS>
  requires std::is_arithmetic_v<S>
constexpr auto operator/(S s, const RHS &b) noexcept {
  return promote_scalar<typename RHS::value_type>(s) / b;
}

template <CExpression LHS, typename S>
  requires std::is_arithmetic_v<S>
constexpr auto operator+(const LHS &a, S s) noexcept {
  return a + promote_scalar<typename LHS::value_type>(s);
}

template <CExpression LHS, typename S>
  requires std::is_arithmetic_v<S>
constexpr auto operator*(const LHS &a, S s) noexcept {
  return a * promote_scalar<typename LHS::value_type>(s);
}

template <CExpression LHS, typename S>
  requires std::is_arithmetic_v<S>
constexpr auto operator-(const LHS &a, S s) noexcept {
  return a - promote_scalar<typename LHS::value_type>(s);
}

template <CExpression LHS, typename S>
  requires std::is_arithmetic_v<S>
constexpr auto operator/(const LHS &a, S s) noexcept {
  return a / promote_scalar<typename LHS::value_type>(s);
}

template <Numeric T> class Constant {
  const T value;
  friend std::ostream &operator<<(std::ostream &out, const Constant<T> &c) {
    if constexpr (PRINT_CONSTANT_VALUE) {
      out << std::format("{}", c.value);
    }
    if constexpr (PRINT_CONSTANT_LABEL) {
      out << "_c";
    }
    return out;
  }

public:
  [[nodiscard]] constexpr auto eval() const noexcept { return value; }
  using value_type = T;
  constexpr explicit Constant(T value) noexcept : value(value) {}
  [[nodiscard]] constexpr auto get() const noexcept { return value; }
  constexpr operator T() const noexcept { return value; }
  [[nodiscard]] constexpr auto derivative() const noexcept {
    return Constant{T{}};
  }
  constexpr void update(const auto &, const auto &) const noexcept {}
  constexpr void collect(const auto &, auto &) const noexcept {}
  constexpr void backward(const auto &, T, auto &) const noexcept {}

  template <typename Syms, std::size_t N>
  [[nodiscard]] constexpr T
  eval_seeded(const std::array<T, N> &) const noexcept {
    return value;
  }

  // eval_seeded_as<U>: embed constant into the deeper type U with zero dual
  // parts.  Uses ConstantEmbedder<U> so custom numeric types (e.g. TaylorDual)
  // can specialise the embedding without touching this code.
  template <typename U, typename Syms, std::size_t N>
  [[nodiscard]] constexpr U
  eval_seeded_as(const std::array<U, N> &) const noexcept {
    using S = scalar_base_t<U>;
    return ConstantEmbedder<U>::embed(
        static_cast<S>(get_real_part<dual_depth_v<T>>(value)));
  }

  template <std::size_t I> [[nodiscard]] constexpr auto get() const noexcept {
    static_assert(I < 2);
    if constexpr (requires { std::tuple_size<T>::value; }) {
      return eval().template get<I>();
    } else if constexpr (I == 0) {
      return eval();
    } else {
      return static_cast<T>(derivative());
    }
  }
};

template <Numeric T, typename GetF, typename AccumDF, typename ZeroDF>
struct FuncHook {
  [[no_unique_address]] GetF get_f_fn;
  [[no_unique_address]] AccumDF accum_df_fn;
  [[no_unique_address]] ZeroDF zero_df_fn;

  [[nodiscard]] constexpr T get_f() const noexcept { return get_f_fn(); }
  constexpr void accum_df(T adj) const noexcept { accum_df_fn(adj); }
  constexpr void zero_df() const noexcept { zero_df_fn(); }
};

template <typename GetF, typename AccumDF, typename ZeroDF>
FuncHook(GetF, AccumDF, ZeroDF)
    -> FuncHook<std::invoke_result_t<GetF>, GetF, AccumDF, ZeroDF>;

// VectorFuncHook: indexed callable hook.
// element(i) returns a FuncHook bound to slot i — all types deduced.
template <Numeric T, typename GetF, typename AccumDF, typename ZeroDF>
struct VectorFuncHook {
  [[no_unique_address]] GetF get_f_fn;
  [[no_unique_address]] AccumDF accum_df_fn;
  [[no_unique_address]] ZeroDF zero_df_fn;

  [[nodiscard]] constexpr T get_f(std::size_t i) const noexcept {
    return get_f_fn(i);
  }
  constexpr void accum_df(std::size_t i, T adj) const noexcept {
    accum_df_fn(i, adj);
  }
  constexpr void zero_df() const noexcept { zero_df_fn(); }

  [[nodiscard]] constexpr auto element(std::size_t i) const noexcept {
    return FuncHook{[gf = get_f_fn, i] { return gf(i); },
                    [ad = accum_df_fn, i](T adj) { ad(i, adj); },
                    [zd = zero_df_fn] { zd(); }};
  }
};

template <typename GetF, typename AccumDF, typename ZeroDF>
VectorFuncHook(GetF, AccumDF, ZeroDF)
    -> VectorFuncHook<std::invoke_result_t<GetF, std::size_t>, GetF, AccumDF,
                      ZeroDF>;

template <Numeric T, CFixedString auto symbol, typename Storage>
class Variable {
  Storage storage;
  friend std::ostream &operator<<(std::ostream &out,
                                  const Variable<T, symbol, Storage> &c) {
    if constexpr (PRINT_VARIABLE_VALUE) {
      out << std::format("{}_", c.eval());
    }
    if constexpr (PRINT_VARIABLE_LABEL) {
      out << symbol.view();
    }
    return out;
  }

public:
  static constexpr auto label = symbol;
  [[nodiscard]] constexpr T eval() const noexcept {
    if constexpr (CHook<Storage, T>)
      return storage.get_f();
    else
      return storage;
  }
  using value_type = T;
  constexpr explicit Variable(Storage s) noexcept : storage(std::move(s)) {}
  constexpr operator T() const noexcept { return eval(); }
  [[nodiscard]] constexpr auto get() const noexcept { return eval(); }
  template <typename U> constexpr decltype(auto) operator=(U &&v) noexcept {
    if constexpr (CHook<Storage, T>) {
      if constexpr (requires { storage.set_f(T{}); })
        storage.set_f(static_cast<T>(std::forward<U>(v)));
    } else if constexpr (std::is_same_v<Storage, std::reference_wrapper<
                                                     std::decay_t<U>>>) {
      storage.get() = std::forward<U>(v);
    } else if constexpr (!std::is_same_v<std::decay_t<U>, T> &&
                         std::is_constructible_v<T, U>) {
      storage = T{std::forward<U>(v)};
    } else {
      storage = std::forward<U>(v);
    }
    return *this;
  }
  constexpr void update(const auto &symbols, const auto &updates) noexcept {
    if constexpr (!CHook<Storage, T>) {
      using Syms = std::decay_t<decltype(symbols)>;
      constexpr auto index = find_index_of_symbol<symbol, Syms>();
      *this = updates[index];
    }
  }
  constexpr void collect(const auto &symbols, auto &out) const noexcept {
    if constexpr (!CHook<Storage, T>) {
      using Syms = std::decay_t<decltype(symbols)>;
      constexpr auto index = find_index_of_symbol<symbol, Syms>();
      out[index] = storage;
    }
  }
  [[nodiscard]] constexpr auto derivative() const noexcept {
    auto ret = T{};
    return Constant{++ret};
  }
  constexpr void backward(const auto &syms, T adj, auto &grads) const noexcept {
    if constexpr (CHook<Storage, T>) {
      storage.accum_df(adj);
    } else {
      using Syms = std::decay_t<decltype(syms)>;
      constexpr auto idx = find_index_of_symbol<symbol, Syms>();
      grads[idx] += adj;
    }
  }

  template <typename Syms, std::size_t N>
  [[nodiscard]] constexpr T
  eval_seeded(const std::array<T, N> &vals) const noexcept {
    constexpr auto idx = find_index_of_symbol<symbol, Syms>();
    return vals[idx];
  }

  template <typename U, typename Syms, std::size_t N>
  [[nodiscard]] constexpr U
  eval_seeded_as(const std::array<U, N> &vals) const noexcept {
    constexpr auto idx = find_index_of_symbol<symbol, Syms>();
    return vals[idx];
  }

  template <std::size_t I> [[nodiscard]] constexpr auto get() const noexcept {
    static_assert(I < 2);
    if constexpr (requires { std::tuple_size<T>::value; }) {
      return eval().template get<I>();
    } else if constexpr (I == 0) {
      return eval();
    } else {
      return static_cast<T>(derivative());
    }
  }
};

#define DEFINE_CONST_UDL(type, suffix)                                         \
  consteval diff::Constant<type> operator"" _##suffix(                         \
      unsigned long long val) {                                                \
    return diff::Constant<type>{static_cast<type>(val)};                       \
  }                                                                            \
  consteval diff::Constant<type> operator"" _##suffix(long double val) {       \
    return diff::Constant<type>{static_cast<type>(val)};                       \
  }

#define DEFINE_VAR_UDL(type, suffix, label)                                    \
  consteval auto operator"" _##suffix(unsigned long long val) {                \
    return diff::Variable<type, diff::FixedString{label}>{                     \
        static_cast<type>(val)};                                               \
  }                                                                            \
  consteval auto operator"" _##suffix(long double val) {                       \
    return diff::Variable<type, diff::FixedString{label}>{                     \
        static_cast<type>(val)};                                               \
  }

} // namespace diff

DEFINE_CONST_UDL(int, ci)
DEFINE_CONST_UDL(double, cd)
DEFINE_VAR_UDL(int, vi, "c")
DEFINE_VAR_UDL(double, vd, "v")

namespace std {
template <diff::Numeric T>
struct tuple_size<diff::Constant<T>> : integral_constant<std::size_t, 2> {};

template <std::size_t I, diff::Numeric T>
struct tuple_element<I, diff::Constant<T>> {
  using type = typename diff::detail::expression_element<T, I>::type;
};

template <diff::Numeric T, diff::CFixedString auto C, typename S>
struct tuple_size<diff::Variable<T, C, S>> : integral_constant<std::size_t, 2> {
};

template <std::size_t I, diff::Numeric T, diff::CFixedString auto C, typename S>
struct tuple_element<I, diff::Variable<T, C, S>> {
  using type = typename diff::detail::expression_element<T, I>::type;
};
} // namespace std

#define PDV(x, label)                                                          \
  diff::Variable<diff::Dual<decltype(x)>, diff::FixedString{label}>(           \
      diff::Dual<decltype(x)>{x, 0})
#define PV(x, label) diff::Variable<decltype(x), diff::FixedString{label}>(x)
#define PC(x) diff::Constant(x)
