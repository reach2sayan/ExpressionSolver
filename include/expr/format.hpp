#pragma once

#include "expr/expressions.hpp"
#include "expr/operations.hpp"
#include "expr/traits.hpp" // CVariable
#include "util/fmt.hpp"    // detail::fmt_put

#include <format>
#include <ostream>
#include <string_view>
#include <tuple>
#include <utility>

namespace ddx::impl {

namespace detail {

// Precedence as seen by the parent; leaves and function calls are atomic.
template <CExpression E> consteval int node_precedence() {
  using U = std::remove_cvref_t<E>;
  if constexpr (CExpressionNode<U>) {
    using Op = typename U::op_type;
    if constexpr (Op::notation == Notation::Function) {
      return precedence_atom;
    } else {
      return Op::precedence;
    }
  } else {
    return precedence_atom;
  }
}

template <typename E> inline constexpr bool is_negation_v = false;
template <Numeric T, CExpression C>
inline constexpr bool is_negation_v<Expression<NegateOp<T>, C>> = true;

template <typename E> inline constexpr bool is_sum_v = false;
template <Numeric T, CExpression... C>
inline constexpr bool is_sum_v<Expression<SumOp<T>, C...>> = true;

// Renders as a subtraction: a two-child sum whose right operand is a negation.
template <CExpression E> consteval bool renders_as_subtraction() {
  using U = std::remove_cvref_t<E>;
  if constexpr (is_sum_v<U> && std::tuple_size_v<typename U::children_t> == 2) {
    return is_negation_v<
        std::remove_cvref_t<std::tuple_element_t<1, typename U::children_t>>>;
  } else {
    return false;
  }
}

// What a walk carries that is not per-node: the output context and the value
// formatter.
template <Numeric V> struct printer {
  std::format_context &ctx;
  const std::formatter<V, char> &value_fmt;

  void put(std::string_view s) const { fmt_put(ctx, s); }
  void put(char c) const { put(std::string_view{&c, 1}); }

  // Through the nested formatter, so a caller's "{::.3f}" reaches every leaf.
  // Generic in the value: a leaf may hand over something that converts to V.
  template <Numeric U> void put_value(const U &v) const {
    ctx.advance_to(value_fmt.format(v, ctx));
  }
};

// `min_prec` is the binding strength the parent demands; anything looser is
// parenthesised.  A left operand may be as loose as its own operator, a right
// operand must be one step tighter -- which is what parenthesises x / (y / z)
// and x - (y + z) but leaves x - y * z clean.
template <CExpression E, Numeric V>
void print_infix(const printer<V> &p, const E &e, int min_prec) {
  using U = std::remove_cvref_t<E>;

  if constexpr (CExpressionNode<U>) {
    using Op = typename U::op_type;
    constexpr int prec = node_precedence<U>();
    const bool parens = prec < min_prec;
    if (parens) {
      p.put('(');
    }

    auto &&kids = e.expressions();

    if constexpr (renders_as_subtraction<U>()) {
      print_infix(p, std::get<0>(kids), prec);
      p.put(" - ");
      auto &&inner = std::get<1>(kids).expressions();
      print_infix(p, std::get<0>(inner), prec + 1);
    } else if constexpr (Op::notation == Notation::Infix) {
      static_assert(std::tuple_size_v<typename U::children_t> == 2,
                    "infix notation is only defined for binary nodes");
      print_infix(p, std::get<0>(kids), prec);
      p.put(' ');
      p.put(Op::label);
      p.put(' ');
      print_infix(p, std::get<1>(kids), prec + 1);
    } else if constexpr (Op::notation == Notation::Prefix) {
      static_assert(std::tuple_size_v<typename U::children_t> == 1,
                    "prefix notation is only defined for unary nodes");
      p.put(Op::label);
      print_infix(p, std::get<0>(kids), prec);
    } else { // Function: brings its own parentheses, children start fresh
      p.put(Op::label);
      p.put('(');
      std::apply(
          [&](const auto &...c) {
            bool first = true;
            auto arg = [&](const auto &child) {
              if (!std::exchange(first, false)) {
                p.put(", ");
              }
              print_infix(p, child, 0);
            };
            (arg(c), ...);
          },
          kids);
      p.put(')');
    }

    if (parens) {
      p.put(')');
    }
  } else if constexpr (CVariable<U>) {
    p.put(U::label.view());
    if constexpr (U::frozen) {
      p.put("_c");
    }
  } else if constexpr (CConstant<U>) {
    p.put_value(e.get());
  } else { // EvalResult and anything else that is just a value
    p.put_value(e.eval());
  }
}

} // namespace detail

std::ostream &operator<<(std::ostream &out, const CExpression auto &e) {
  return out << std::format("{}", e);
}

} // namespace ddx::impl

// The whole spec is the value-spec, handed verbatim to the formatter for
// value_type.  The leading ':' is optional, as in std::formatter for ranges.
template <ddx::impl::CExpression E> struct std::formatter<E, char> {
  constexpr auto parse(std::format_parse_context &ctx) {
    auto it = ctx.begin();
    if (it != ctx.end() && *it == ':') {
      ++it;
    }
    ctx.advance_to(it);
    return value_fmt_.parse(ctx);
  }

  auto format(const E &e, std::format_context &ctx) const {
    const ddx::impl::detail::printer p{ctx, value_fmt_};
    ddx::impl::detail::print_infix(p, e, 0);
    return ctx.out();
  }

private:
  std::formatter<typename E::value_type, char> value_fmt_{};
};

// The JIT's IR prints through the same machinery.  Guarded because ddx::jit is
// opt-in while this header is core; ddx_jit defines DDX_HAS_JIT, and
// jit/kernel.hpp names no LLVM type, so nothing third-party arrives with it.
#ifdef DDX_HAS_JIT
#include "jit/kernel.hpp"

template <> struct std::formatter<ddx::jit::Ir, char> {
  constexpr auto parse(std::format_parse_context &ctx) const {
    return ctx.begin();
  }

  // A formatter has nowhere to put an error, so a module that would not build
  // prints why instead of the IR that does not exist.
  auto format(const ddx::jit::Ir &ir, std::format_context &ctx) const {
    const auto text = ir.str();
    ddx::impl::detail::fmt_put(ctx,
                               text ? std::string_view{*text}
                                    : std::string_view{text.error().detail});
    return ctx.out();
  }
};

namespace ddx::jit {
inline std::ostream &operator<<(std::ostream &out, const Ir &ir) {
  return out << std::format("{}", ir);
}
} // namespace ddx::jit
#endif
