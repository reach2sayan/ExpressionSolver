#pragma once

// The whole printing surface of the library, in one place.
//
// An expression is a type, not a data structure, so printing it is a recursive
// walk over `children_t` -- and the walk is identical for every node.  Writing
// that walk once, here, is what removes the per-node `operator<<` and the
// `Op::print` member that used to make every operation depend on <ostream>
// just to be able to spell itself.  Operations now carry only *data* about how
// they are written (label, notation, precedence; see operations.hpp).
//
// The entry point is std::formatter, with operator<< as a thin forward, so
// std::format, std::print and iostreams all share a single implementation and
// cannot drift apart.

#include "expr/expressions.hpp"
#include "expr/operations.hpp"
#include "expr/traits.hpp" // CVariable

#include <algorithm>
#include <cstddef>
#include <format>
#include <ostream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace diff {

// Selected per call by the format spec, not by a global toggle:
//
//   Math   x * y + sin(x)      infix, parenthesised only where needed
//   Debug  x_var * 2_c         same shape, every leaf annotated with its kind
//   Tree   indented node dump, each line tagged with its preorder cache index
enum class PrintStyle : std::uint8_t { Math, Debug, Tree };

namespace detail {

// Precedence of a whole node, as seen by its parent.  Leaves and function-call
// nodes are atomic: neither can be split by a surrounding operator, so neither
// ever needs parentheses.
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

// `a - b` is built as `a + (-b)` (values.hpp), so a sum whose right child is a
// negation is the only shape subtraction ever has.  Recognising it here is what
// keeps `x - y` from printing as `x + -y`.
template <typename E> inline constexpr bool is_negation_v = false;
template <Numeric T, CExpression C>
inline constexpr bool is_negation_v<Expression<NegateOp<T>, C>> = true;

template <typename E> inline constexpr bool is_sum_v = false;
template <Numeric T, CExpression... C>
inline constexpr bool is_sum_v<Expression<SumOp<T>, C...>> = true;

// True when `e` should render as a subtraction: a two-child sum whose right
// operand is a negation.
template <CExpression E> consteval bool renders_as_subtraction() {
  using U = std::remove_cvref_t<E>;
  if constexpr (is_sum_v<U> && std::tuple_size_v<typename U::children_t> == 2) {
    return is_negation_v<std::remove_cvref_t<
        std::tuple_element_t<1, typename U::children_t>>>;
  } else {
    return false;
  }
}

// The state a walk carries that is not per-node: where output goes, how values
// are formatted, and which style was asked for.  Bundled so the two walkers
// take two arguments instead of five.
template <typename Ctx, typename ValueFmt> struct printer {
  Ctx &ctx;
  const ValueFmt &value_fmt;
  PrintStyle style;

  void put(std::string_view s) const {
    ctx.advance_to(std::ranges::copy(s, ctx.out()).out);
  }
  void put(char c) const { put(std::string_view{&c, 1}); }

  // Values go through the nested formatter, so a value-spec supplied by the
  // caller ("{::.3f}") reaches every leaf.
  template <Numeric V> void put_value(const V &v) const {
    ctx.advance_to(value_fmt.format(v, ctx));
  }
};

// --- infix walk (Math and Debug styles) ------------------------------------
//
// `min_prec` is the binding strength the parent demands of this subtree;
// anything looser has to be parenthesised.  A left operand may be as loose as
// its own operator (`a - b - c` is flat), a right operand must be one step
// tighter -- which is exactly what keeps `x / (y / z)` and `x - (y + z)`
// parenthesised while leaving `x - y * z` clean.
template <CExpression E, typename P>
void print_infix(const P &p, const E &e, int min_prec) {
  using U = std::remove_cvref_t<E>;
  const bool debug = p.style == PrintStyle::Debug;

  if constexpr (CExpressionNode<U>) {
    using Op = typename U::op_type;
    constexpr int prec = node_precedence<U>();
    const bool parens = prec < min_prec;
    if (parens) {
      p.put('(');
    }

    // expressions() returns by value for a stateless node and by reference for
    // a storing one; auto&& binds -- and lifetime-extends -- either.
    auto &&kids = e.expressions();

    if constexpr (renders_as_subtraction<U>()) {
      if (debug) { // debug shows the real a + (-b) structure
        print_infix(p, std::get<0>(kids), prec);
        p.put(" + ");
        print_infix(p, std::get<1>(kids), prec + 1);
      } else {
        print_infix(p, std::get<0>(kids), prec);
        p.put(" - ");
        auto &&inner = std::get<1>(kids).expressions();
        print_infix(p, std::get<0>(inner), prec + 1);
      }
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
    if (debug) {
      p.put(U::frozen ? "_frozen" : "_var");
    }
  } else if constexpr (CConstant<U>) {
    p.put_value(e.get());
    if (debug) {
      p.put(is_lit_v<U> ? "_lit" : "_c");
    }
  } else { // EvalResult and anything else that is just a value
    p.put_value(e.eval());
  }
}

// --- tree walk (Tree style) ------------------------------------------------
//
// One line per node, tagged with its preorder index -- the same numbering
// child_base_at() assigns, which is what the value cache and the reverse sweep
// are keyed on.  That correspondence is the reason to print a tree at all.
template <std::size_t Base, CExpression E, typename P>
void print_tree(const P &p, const E &e, std::string_view prefix,
                std::string_view branch, bool last) {
  using U = std::remove_cvref_t<E>;

  // Newline before the line rather than after it, so the rendering has no
  // trailing separator -- only the root carries preorder index 0.
  if constexpr (Base != 0) {
    p.put('\n');
  }
  p.put(prefix);
  p.put(branch);
  p.ctx.advance_to(std::format_to(p.ctx.out(), "[{}] ", Base));

  if constexpr (CExpressionNode<U>) {
    p.put(U::op_type::label);

    std::string child_prefix{prefix};
    if (!branch.empty()) {
      child_prefix += last ? "    " : "│   ";
    }

    using Kids = typename U::children_t;
    constexpr std::size_t n = std::tuple_size_v<Kids>;
    auto &&kids = e.expressions();
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      ((print_tree<child_base_at<Base, Kids, I>()>(
           p, std::get<I>(kids), child_prefix,
           I + 1 == n ? "└── " : "├── ",
           I + 1 == n)),
       ...);
    }(std::make_index_sequence<n>{});
  } else if constexpr (CVariable<U>) {
    p.put(U::label.view());
    p.put(U::frozen ? " (frozen)" : "");
  } else if constexpr (CConstant<U>) {
    p.put_value(e.get());
    p.put(is_lit_v<U> ? " (lit)" : " (const)");
  } else {
    p.put_value(e.eval());
  }
}

} // namespace detail

// One inserter for every expression shape.  Constrained on CExpression, so it
// is found by ADL for Lit, Variable, Expression and EvalResult alike, and beats
// ostream's member operator<<(double) for Lit -- that one is reachable only
// through the operator T() conversion, which is a worse match than an exact
// template.
template <CExpression E>
std::ostream &operator<<(std::ostream &out, const E &e) {
  return out << std::format("{}", e);
}

} // namespace diff

// Spec grammar, following std::formatter for ranges (which likewise uses a
// trailing ':' to introduce the spec for what it contains):
//
//   expr-spec  ::= [ style ] [ ':' value-spec ]
//   style      ::= 'm' math (default) | 'd' debug | 't' tree
//
// value-spec is handed verbatim to the formatter for the expression's
// value_type, so "{::.3f}" fixes the precision of every number in the tree.
template <diff::CExpression E> struct std::formatter<E, char> {
  constexpr auto parse(std::format_parse_context &ctx) {
    auto it = ctx.begin();
    if (it != ctx.end() && *it != '}' && *it != ':') {
      switch (*it) {
      case 'm':
        style_ = diff::PrintStyle::Math;
        break;
      case 'd':
        style_ = diff::PrintStyle::Debug;
        break;
      case 't':
        style_ = diff::PrintStyle::Tree;
        break;
      default:
        throw std::format_error(
            "expression format spec: style must be 'm', 'd' or 't'");
      }
      ++it;
    }
    if (it != ctx.end() && *it == ':') {
      ++it;
    }
    ctx.advance_to(it);
    return value_fmt_.parse(ctx);
  }

  auto format(const E &e, std::format_context &ctx) const {
    const diff::detail::printer p{ctx, value_fmt_, style_};
    if (style_ == diff::PrintStyle::Tree) {
      diff::detail::print_tree<0>(p, e, "", "", true);
    } else {
      diff::detail::print_infix(p, e, 0);
    }
    return ctx.out();
  }

private:
  diff::PrintStyle style_ = diff::PrintStyle::Math;
  std::formatter<typename E::value_type, char> value_fmt_{};
};
