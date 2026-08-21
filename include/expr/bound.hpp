#pragma once

#include "drivers/symbolic.hpp"
#include "expr/named_value.hpp"
#include "expr/traits.hpp"
#include "util/config.hpp"
#include "util/error.hpp"
#include "util/fixed_string.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ddx::impl {

// A compile-time label-keyed map: one slot per symbol.
template <Numeric Scalar, CSymbolList SymList> struct ValueMap {
  using symbols = SymList;
  using value_type = Scalar;

  static constexpr bool kValueMap = true;
  static constexpr std::size_t arity = mp::mp_size<SymList>::value;

  std::array<Scalar, arity> slots{};

  template <FixedString S>
  [[nodiscard]] static constexpr decltype(auto) slot(auto &&self) noexcept {
    constexpr auto idx = find_index_of_symbol<S, SymList>();
    static_assert(idx < arity, "ValueMap: symbol not present in map");
    if constexpr (std::is_lvalue_reference_v<decltype(self)>) {
      return std::forward<decltype(self)>(self).slots[idx];
    } else {
      return Scalar{self.slots[idx]};
    }
  }

  // Subscript spelling: m["x"_s] = v.  See DDX_KEYED_ACCESSORS.
  DDX_KEYED_ACCESSORS(FixedString S, CFixedString auto S, S, symbol_type<S>)
  template <FixedString S> constexpr void set(const Scalar &v) noexcept {
    slot<S>(*this) = v;
  }
};

template <typename T>
concept CValueMap = requires {
  typename std::remove_cvref_t<T>::symbols;
  requires std::remove_cvref_t<T>::kValueMap;
};

namespace detail {

// For each canonical slot, which argument position supplies it.
template <CSymbolList SymList, std::size_t N, CSymbol... ArgSyms>
consteval std::array<std::size_t, N> arg_of_canonical() noexcept {
  const std::array<std::size_t, N> canonical_of_arg{
      find_index_of_symbol<ArgSyms::value, SymList>()...};
  std::array<std::size_t, N> out{};
  for (std::size_t j = 0; j < N; ++j) {
    out[canonical_of_arg[j]] = j;
  }
  return out;
}

// Canonical symbol order to the map's; the map may be a superset.
template <CSymbolList ExprSyms, CSymbolList MapSyms, std::size_t N>
consteval std::array<std::size_t, N> symbol_permutation() noexcept {
  std::array<std::size_t, N> p{};
  static_for<N>([&]<std::size_t I>() {
    p[I] = find_index_of_symbol<mp::mp_at_c<ExprSyms, I>::value, MapSyms>();
  });
  return p;
}

} // namespace detail

// values(named<"x">(1.0), named<"y">(0.5))
template <FixedString... Syms, Numeric... Vs>
[[nodiscard]] constexpr auto values(NamedValue<Syms, Vs>... nv) noexcept {
  using SymList = unique_tuple_t<mp::mp_list<symbol_type<Syms>...>>;
  constexpr std::size_t N = sizeof...(Syms);
  static_assert(mp::mp_size<SymList>::value == N, "values: duplicate symbol");

  using Scalar = std::common_type_t<Vs...>;
  constexpr auto pos =
      detail::arg_of_canonical<SymList, N, symbol_type<Syms>...>();
  const auto args = std::tuple<Vs...>{nv.value...};
  return index_apply<N>([&]<std::size_t... I>() {
    return ValueMap<Scalar, SymList>{
        std::array<Scalar, N>{static_cast<Scalar>(std::get<pos[I]>(args))...}};
  });
}

// An expression paired with a point; the expression is empty.
template <CExpression Expr, CValueMap Map> struct Bound {
  static_assert(!std::is_reference_v<Expr> && !std::is_reference_v<Map>,
                "Bound stores the expression and the map by value");
  using expr_type = std::remove_cvref_t<Expr>;
  using map_type = std::remove_cvref_t<Map>;
  using symbols = detail::expr_symbols_t<expr_type>;
  using value_type = typename expr_type::value_type;

  static constexpr std::size_t arity = mp::mp_size<symbols>::value;

  static constexpr auto kPerm =
      detail::symbol_permutation<symbols, typename map_type::symbols, arity>();

  static_assert(std::ranges::all_of(kPerm,
                                    [](std::size_t i) {
                                      return i < map_type::arity;
                                    }),
                "bind: the map does not supply every symbol the expression "
                "uses");

  [[no_unique_address]] expr_type expr{};
  map_type map{};

  template <Numeric U = value_type>
  [[nodiscard]] constexpr std::array<U, arity> point() const noexcept {
    std::array<U, arity> out{};
    std::ranges::transform(kPerm, out.begin(), [&](std::size_t i) {
      return static_cast<U>(map.slots[i]);
    });
    return out;
  }

  [[nodiscard]] constexpr value_type eval() const noexcept {
    return expr.template eval_seeded<symbols>(point());
  }

  template <Numeric U>
  [[nodiscard]] constexpr U
  eval_as(const std::array<U, arity> &seed) const noexcept {
    return expr.template eval_seeded<symbols>(seed);
  }

  template <FixedString S>
  constexpr void set(const typename map_type::value_type &v) noexcept {
    map.template set<S>(v);
  }

  template <FixedString S>
  [[nodiscard]] static constexpr decltype(auto) slot(auto &&self) noexcept {
    return map_type::template slot<S>(std::forward<decltype(self)>(self).map);
  }

  DDX_KEYED_ACCESSORS(FixedString S, CFixedString auto S, S, symbol_type<S>)
};

template <CExpression Expr, CValueMap Map> Bound(Expr, Map) -> Bound<Expr, Map>;

template <CExpression Expr, CValueMap Map>
[[nodiscard]] constexpr auto bind(Expr &&e, Map &&m) noexcept {
  return Bound<std::remove_cvref_t<Expr>, std::remove_cvref_t<Map>>{
      static_cast<Expr &&>(e), static_cast<Map &&>(m)};
}

// bind(expr, named<"x">(1.0), named<"y">(0.5))
template <CExpression Expr, FixedString... Syms, Numeric... Vs>
[[nodiscard]] constexpr auto bind(Expr &&e,
                                  NamedValue<Syms, Vs>... nv) noexcept {
  return bind(static_cast<Expr &&>(e), values(nv...));
}

template <CExpression Expr>
[[nodiscard]] consteval auto symbol_order(const Expr &) noexcept {
  return symbol_order<std::remove_cvref_t<Expr>>();
}

namespace detail {

// The one spelling of a point whose length is not known until it arrives: a
// single range that is not tuple-like.  Every other spelling is counted by a
// static_assert, so this is the only one that can answer with an error.
template <typename... Args>
concept CDynamicPoint =
    sizeof...(Args) == 1 && (std::ranges::input_range<Args> && ...) &&
    !(CTupleLike<Args> && ...);

// Every spelling of "a point" reduced to an array of N values in canonical
// symbol order.  Written against a symbol list, not an expression, because
// Equation supplies its own.
template <CSymbolList Syms, Numeric U, std::size_t N, CEvalArg... Args>
  requires(!CDynamicPoint<Args...>)
[[nodiscard]] constexpr std::array<U, N>
make_point(const Args &...args) noexcept {
  // clang-format off
  if constexpr (sizeof...(Args) == 0) { // constant-folded: no free symbols
    static_assert(N == 0,
                  "eval: this expression has free symbols, so it needs a point "
                  "(see symbol_order<Expr>())");
    return {};
  }

  // point(map) / point(named<"x">(..), ..) -- read by name.
  else if constexpr ((CValueMap<Args> && ...) || (is_named_value_v<std::remove_cvref_t<Args>> && ...)) {
    const auto map = [&] {
      if constexpr ((CValueMap<Args> && ...)) {
        static_assert(sizeof...(Args) == 1, "eval: pass a single ValueMap");
        return std::get<0>(std::tuple{args...});
      } else {
        return values(args...);
      }
    }();
    std::array<U, N> vals{};
    static_for<N>([&]<std::size_t I>() {
      vals[I] = static_cast<U>(map.template get<mp::mp_at_c<Syms, I>::value>());
    });
    return vals;
  }

  // point(tuple-like range) -- the size is in the type, so it is a diagnostic.
  else if constexpr (sizeof...(Args) == 1 && (std::ranges::input_range<Args> && ...)) {
    return [&](const auto &r) {
      static_assert(
          std::tuple_size_v<std::remove_cvref_t<decltype(r)>> == N,
          "eval: range size must equal the expression's symbol count "
          "(see symbol_order<Expr>())");
      std::array<U, N> vals{};
      std::size_t i = 0;
      std::ranges::for_each(r | std::views::take(N), [&](const auto &v) {
        vals[i++] = static_cast<U>(v);
      });
      return vals;
    }(args...);
  }

  // point(x, y, z) -- positional, in canonical order
  else {
    static_assert(sizeof...(Args) == N,
                  "eval: supply exactly one value per symbol, in canonical "
                  "order (see symbol_order<Expr>())");
    return std::array<U, N>{static_cast<U>(args)...};
  }
  //clang-format on
}

// point(range) -- the length arrives with the range, so a short one is the one
// wrong point this library cannot catch at compile time.
template <CSymbolList Syms, Numeric U, std::size_t N, CEvalArg... Args>
  requires(CDynamicPoint<Args...>)
[[nodiscard]] constexpr result<std::array<U, N>>
make_point(const Args &...args) noexcept {
  return [&](const auto &r) -> result<std::array<U, N>> {
    std::array<U, N> vals{};
    std::size_t i = 0;
    std::ranges::for_each(
        r | std::views::take(N),
        [&](const auto &v) { vals[i++] = static_cast<U>(v); });
    if (i != N) {
      return fail(errc::short_point);
    }
    return vals;
  }(args...);
}

// Run `body` on the point.  Whether the point is an array or an error is
// settled here and nowhere else, so every numeric member below reads as though
// it were always an array -- and returns result<T> exactly when its caller
// spelled the point as a range.
template <CSymbolList Syms, Numeric U, std::size_t N, typename Body,
          CEvalArg... Args>
[[nodiscard]] constexpr auto with_point(Body &&body,
                                        const Args &...args) noexcept {
  if constexpr (CDynamicPoint<Args...>) {
    return make_point<Syms, U, N>(args...).transform(
        static_cast<Body &&>(body));
  } else {
    return static_cast<Body &&>(body)(make_point<Syms, U, N>(args...));
  }
}

template <CExpression Expr, CEvalArg... Args>
[[nodiscard]] constexpr auto eval_dispatch(const Expr &e,
                                           const Args &...args) noexcept {
  using VT = typename std::remove_cvref_t<Expr>::value_type;
  using Syms = expr_symbols_t<Expr>;
  return with_point<Syms, VT, expr_arity_v<Expr>>(
      [&e](const auto &vals) { return e.template eval_seeded<Syms>(vals); },
      args...);
}

// Forward-mode sweep seeded on `Seed`: the ordinary seeded sweep with Dual<VT>,
// so there is no separate forward engine.  Slot k gets a unit tangent exactly
// when symbol k is the one being differentiated.
template <auto Seed, CExpression Expr, CEvalArg... Args>
[[nodiscard]] constexpr auto tangent_dispatch(const Expr &e,
                                              const Args &...args) {
  using VT = typename std::remove_cvref_t<Expr>::value_type;
  using Syms = expr_symbols_t<Expr>;
  constexpr std::size_t N = expr_arity_v<Expr>;
  static_assert(sizeof...(Args) == N,
                "eval_with_tangent: supply exactly one value per symbol, in "
                "canonical order (see symbol_order<Expr>())");
  const std::array<VT, N> vals{static_cast<VT>(args)...};
  std::array<Dual<VT>, N> seeds{};
  static_for<N>([&]<std::size_t I>() {
    constexpr bool seeded = mp::mp_at_c<Syms, I>::value == Seed;
    seeds[I] = Dual<VT>{vals[I], seeded ? VT{1} : VT{}};
  });
  return e.template eval_seeded<Syms>(seeds);
}

} // namespace detail

template <CExpression Expr, CEvalArg... Args>
  requires(sizeof...(Args) > 0)
[[nodiscard]] constexpr auto eval(const Expr &e,
                                  const Args &...args) noexcept {
  return detail::eval_dispatch(e, args...);
}

} // namespace ddx::impl
