#pragma once

#include "gradient.hpp" // NamedValue, named, find_index_of_symbol
#include "traits.hpp"   // extract_symbols_from_expr_t, unique_tuple_t

#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <type_traits>

// ===========================================================================
// Root-held values.
//
// Expression leaves carry no value of their own; a point is supplied at the
// root instead.  The saving is not in *having* values, it is in holding one
// slot per VARIABLE rather than one per leaf occurrence: an expression that
// mentions `x` sixteen times still costs a single slot.
//
// ValueMap is make_values() (gradient.hpp) made storable — a plain array of
// values in canonical symbol order, with a consteval label->index lookup.
//
// Bound<Expr, Map> pairs an (empty) expression with a map and so keeps eval()
// nullary.  Because the expression contributes no bytes, sizeof(Bound) is just
// the map.
// ===========================================================================

namespace diff {

// A compile-time label-keyed map: one slot per symbol, stored in canonical
// (alphabetical) order so it lines up with every other array in the library.
// Lookup is a consteval index, so `get<"x">()` compiles to a direct member
// access — this is not a hash table despite the name.
//
// Aggregate with a single public array member, so it is a structural type and
// can be used as a non-type template parameter.
template <typename Scalar, typename SymList> struct ValueMap {
  using symbols = SymList;
  using value_type = Scalar;

  static constexpr bool kValueMap = true;
  static constexpr std::size_t arity = mp::mp_size(SymList{});

  std::array<Scalar, arity> slots{};

  template <FixedString S>
  [[nodiscard]] constexpr const Scalar &get() const noexcept {
    constexpr auto idx = find_index_of_symbol<S, SymList>();
    static_assert(idx < arity, "ValueMap: symbol not present in map");
    return slots[idx];
  }

  template <FixedString S> constexpr void set(const Scalar &v) noexcept {
    constexpr auto idx = find_index_of_symbol<S, SymList>();
    static_assert(idx < arity, "ValueMap: symbol not present in map");
    slots[idx] = v;
  }

  [[nodiscard]] constexpr const std::array<Scalar, arity> &
  to_array() const noexcept {
    return slots;
  }
};

// No deduction guide for ValueMap: SymList is a type-level symbol list and
// cannot be recovered from the slot array, so values() is the way to build one.

// Anything carrying the tag; deliberately narrow so a map is never also
// matched as an input_range by the eval overloads.
template <typename T>
concept CValueMap = requires {
  typename std::remove_cvref_t<T>::symbols;
  requires std::remove_cvref_t<T>::kValueMap;
};

template <typename T> struct is_named_value : std::false_type {};
template <FixedString S, typename V>
struct is_named_value<NamedValue<S, V>> : std::true_type {};
template <typename T>
concept CNamedValue = is_named_value<std::remove_cvref_t<T>>::value;

namespace detail {

// For each canonical slot, which argument position supplies it.
template <typename SymList, std::size_t N, typename... ArgSyms>
consteval std::array<std::size_t, N> arg_of_canonical() noexcept {
  const std::array<std::size_t, N> canonical_of_arg{
      find_index_of_symbol<ArgSyms::value, SymList>()...};
  std::array<std::size_t, N> out{};
  // Scatter: invert canonical_of_arg.
  std::ranges::for_each(std::views::iota(std::size_t{0}, N),
                        [&](std::size_t j) { out[canonical_of_arg[j]] = j; });
  return out;
}

// Permutation taking the expression's canonical symbol order to the map's.
// The map may legitimately be a superset of the expression's symbols.
template <typename ExprSyms, typename MapSyms, std::size_t N>
consteval std::array<std::size_t, N> symbol_permutation() noexcept {
  std::array<std::size_t, N> p{};
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    ((p[I] = find_index_of_symbol<mp::mp_at_c<ExprSyms, I>::value, MapSyms>()),
     ...);
  }(std::make_index_sequence<N>{});
  return p;
}

} // namespace detail

// values(named<"x">(1.0), named<"y">(0.5))
template <FixedString... Syms, typename... Vs>
[[nodiscard]] constexpr auto values(NamedValue<Syms, Vs>... nv) noexcept {
  using SymList = unique_tuple_t<mp::mp_list<symbol_type<Syms>...>>;
  constexpr std::size_t N = sizeof...(Syms);
  static_assert(mp::mp_size(SymList{}) == N, "values: duplicate symbol");

  using Scalar = std::common_type_t<Vs...>;
  constexpr auto pos =
      detail::arg_of_canonical<SymList, N, symbol_type<Syms>...>();
  const auto args = std::tuple<Vs...>{nv.value...};
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    return ValueMap<Scalar, SymList>{
        std::array<Scalar, N>{static_cast<Scalar>(std::get<pos[I]>(args))...}};
  }(std::make_index_sequence<N>{});
}

// An expression paired with a point.  The expression is empty, so this costs
// exactly the map.
template <typename Expr, typename Map> struct Bound {
  using expr_type = std::remove_cvref_t<Expr>;
  using map_type = std::remove_cvref_t<Map>;
  using symbols = extract_symbols_from_expr_t<expr_type>;
  using value_type = typename expr_type::value_type;

  static constexpr std::size_t arity = mp::mp_size(symbols{});

  static constexpr auto kPerm =
      detail::symbol_permutation<symbols, typename map_type::symbols, arity>();

  static_assert(std::ranges::all_of(
                    kPerm, [](std::size_t i) { return i < map_type::arity; }),
                "bind: the map does not supply every symbol the expression "
                "uses");

  [[no_unique_address]] expr_type expr{};
  map_type map{};

  // The point, in the expression's canonical symbol order.
  template <typename U = value_type>
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

  // Evaluate with a deeper numeric type (Dual, TaylorDual, VectorDual, ...).
  template <typename U>
  [[nodiscard]] constexpr U
  eval_as(const std::array<U, arity> &seed) const noexcept {
    return expr.template eval_seeded_as<U, symbols>(seed);
  }

  template <FixedString S>
  constexpr void set(const typename map_type::value_type &v) noexcept {
    map.template set<S>(v);
  }
  template <FixedString S>
  [[nodiscard]] constexpr decltype(auto) get() const noexcept {
    return map.template get<S>();
  }
};

// Lets `Bound{expr, map}` work without naming either parameter.
template <CExpression Expr, CValueMap Map>
Bound(Expr, Map) -> Bound<Expr, Map>;

template <CExpression Expr, CValueMap Map>
[[nodiscard]] constexpr auto bind(Expr &&e, Map &&m) noexcept {
  return Bound<std::remove_cvref_t<Expr>, std::remove_cvref_t<Map>>{
      static_cast<Expr &&>(e), static_cast<Map &&>(m)};
}

// bind(expr, named<"x">(1.0), named<"y">(0.5))
template <CExpression Expr, FixedString... Syms, typename... Vs>
[[nodiscard]] constexpr auto bind(Expr &&e,
                                  NamedValue<Syms, Vs>... nv) noexcept {
  return bind(static_cast<Expr &&>(e), values(nv...));
}

// ===========================================================================
// Positional evaluation.
//
// Values are supplied in CANONICAL symbol order — alphabetical by name, not
// source order.  Use symbol_order<Expr>() (gradient.hpp) to see that order, or
// the named/map forms above to be immune to it.
//
// Both forms stay constexpr.  A `throw` inside a constexpr function is only a
// problem if it is actually reached during constant evaluation, so a
// too-short range is a *compile error* at compile time and an exception at
// run time.  That is why these are not noexcept: silently evaluating at the
// wrong point is the worst outcome an AD library can produce.
// ===========================================================================

namespace detail {

template <typename Expr>
using expr_symbols_t = extract_symbols_from_expr_t<std::remove_cvref_t<Expr>>;

template <typename Expr>
inline constexpr std::size_t expr_arity_v =
    mp::mp_size(expr_symbols_t<Expr>{});

// std::array and friends advertise a static size; plain ranges do not.
template <typename R>
concept CStaticSized = requires { std::tuple_size<std::remove_cvref_t<R>>::value; };

} // namespace detail

// symbol_order(expr) — value-taking convenience over symbol_order<Expr>().
template <CExpression Expr>
[[nodiscard]] consteval auto symbol_order(const Expr &) noexcept {
  return symbol_order<std::remove_cvref_t<Expr>>();
}

// Single entry point behind both the free eval(expr, ...) and the member
// expr.eval(...), so the two can never drift apart.
namespace detail {

template <CExpression Expr, typename... Args>
[[nodiscard]] constexpr auto eval_dispatch(const Expr &e,
                                           const Args &...args) {
  using VT = typename std::remove_cvref_t<Expr>::value_type;
  using Syms = expr_symbols_t<Expr>;
  constexpr std::size_t N = expr_arity_v<Expr>;

  if constexpr (sizeof...(Args) == 1 &&
                (CValueMap<Args> && ...)) { // eval(map)
    return bind(e, args...).eval();
  } else if constexpr ((CNamedValue<Args> && ...)) { // eval(named<"x">(..), ..)
    return bind(e, args...).eval();
  } else if constexpr (sizeof...(Args) == 1 &&
                       (std::ranges::input_range<Args> && ...)) { // eval(range)
    return [&](const auto &r) {
      if constexpr (CStaticSized<decltype(r)>) {
        static_assert(
            std::tuple_size_v<std::remove_cvref_t<decltype(r)>> == N,
            "eval: range size must equal the expression's symbol count "
            "(see symbol_order<Expr>())");
      }
      std::array<VT, N> vals{};
      std::size_t i = 0;
      std::ranges::for_each(r | std::views::take(N), [&](const auto &v) {
        vals[i++] = static_cast<VT>(v);
      });
      if (i != N)
        throw std::out_of_range("eval: range supplied fewer values than the "
                                "expression has symbols");
      return e.template eval_seeded<Syms>(vals);
    }(args...);
  } else { // eval(x, y, z) — positional
    static_assert(sizeof...(Args) == N,
                  "eval: supply exactly one value per symbol, in canonical "
                  "order (see symbol_order<Expr>())");
    const std::array<VT, N> vals{static_cast<VT>(args)...};
    return e.template eval_seeded<Syms>(vals);
  }
}

} // namespace detail

// eval(expr, ...) — same dispatcher the member uses.
template <CExpression Expr, typename... Args>
  requires(sizeof...(Args) > 0)
[[nodiscard]] constexpr auto eval(const Expr &e, const Args &...args) {
  return detail::eval_dispatch(e, args...);
}

} // namespace diff
