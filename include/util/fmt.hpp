#pragma once

// The pieces every std::formatter in the library is built out of: emit fixed
// punctuation, and emit numbers through a nested formatter handed the caller's
// spec.  Writing those once is what keeps "{:.3f}" meaning the same thing in
// Dual, TaylorDual and the expression tree.
//
// This header knows about none of those types, so it sits below all of them.

#include <algorithm>
#include <cstddef>
#include <format>
#include <ostream>
#include <ranges>
#include <string_view>
#include <type_traits>

namespace diff::detail {

// The infinitesimal unit, U+03B5 GREEK SMALL LETTER EPSILON, spelled as UTF-8
// bytes rather than "\u03b5" so the literal does not depend on the source
// encoding the translation unit happens to be compiled under.
inline constexpr std::string_view eps = "\xce\xb5";

// Append literal text, keeping the context's iterator in step.
inline void fmt_put(std::format_context &ctx, std::string_view s) {
  ctx.advance_to(std::ranges::copy(s, ctx.out()).out);
}

// The half of a dual formatter that does not depend on which dual it is: every
// dual is a value followed by derivative parts, and every one forwards the spec
// whole to the numbers.  The derived specialisation supplies only the layout.
// S is the coefficient type, and stays a bare parameter on purpose: the
// requirement is `std::formattable<S, char>`, but a nested dual names this base
// from inside the very std::formatter specialisation that would answer it, so
// spelling the constraint is a recursive instantiation and does not compile.
template <typename S> struct dual_formatter_base {
  constexpr auto parse(std::format_parse_context &ctx) {
    return part_.parse(ctx);
  }

protected:
  static void put(std::format_context &ctx, std::string_view s) {
    fmt_put(ctx, s);
  }

  void num(std::format_context &ctx, const S &v) const {
    ctx.advance_to(part_.format(v, ctx));
  }

  // A coefficient that is itself a series has to be bracketed, or the levels run
  // together: Dual<Dual<double>> would print 1+2ε+3+4εε, which reads as four
  // terms of one series rather than two of two.
  void term(std::format_context &ctx, const S &v) const {
    if constexpr (std::is_arithmetic_v<S>) {
      num(ctx, v);
    } else {
      put(ctx, "(");
      num(ctx, v);
      put(ctx, ")");
    }
  }

  // c0+c1ε+c2ε^2+... -- the perturbation series that both Dual (two terms) and
  // TaylorDual (N+1 of them) are.  Every coefficient is printed, zeros
  // included, so the order of the series is legible in the text.
  void series(std::format_context &ctx, const auto &coeffs) const {
    for (std::size_t k = 0; k != std::ranges::size(coeffs); ++k) {
      if (k > 0) {
        put(ctx, "+");
      }
      term(ctx, coeffs[k]);
      if (k == 1) {
        put(ctx, eps);
      } else if (k > 1) {
        ctx.advance_to(std::format_to(ctx.out(), "{}^{}", eps, k));
      }
    }
  }

private:
  std::formatter<S, char> part_{};
};

// Opt-in list for the shared operator<< below; each dual header adds itself.
template <typename T> inline constexpr bool is_dual_family_v = false;

} // namespace diff::detail

namespace diff {

// Anything in the dual family.  One inserter serves all of them, so the iostream
// spelling cannot drift from the std::format one.
template <typename T>
concept CDualFamily = detail::is_dual_family_v<std::remove_cvref_t<T>>;

template <CDualFamily D>
std::ostream &operator<<(std::ostream &out, const D &d) {
  return out << std::format("{}", d);
}

} // namespace diff
