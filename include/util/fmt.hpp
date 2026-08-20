#pragma once

// The pieces every std::formatter in the library is built out of.
//
// Three types are formattable here -- Dual, TaylorDual and the
// expression tree -- and all four do the same two things: emit fixed
// punctuation, and emit numbers through a nested formatter that was handed the
// caller's spec.  Writing those two once is what keeps "{:.3f}" meaning the
// same thing in all four, and keeps a fifth from having to reinvent them.
//
// This header knows about none of those types, which is why it can sit below
// all of them.

#include <algorithm>
#include <cstddef>
#include <format>
#include <ostream>
#include <ranges>
#include <string_view>
#include <type_traits>

namespace diff::detail {

// Append literal text, keeping the context's iterator in step.
inline void fmt_put(std::format_context &ctx, std::string_view s) {
  ctx.advance_to(std::ranges::copy(s, ctx.out()).out);
}

// The half of a dual formatter that does not depend on which dual it is.
//
// All three duals are a value followed by derivative parts, and all three want
// the same thing from a spec: forward it whole to the numbers, so one "{:.3f}"
// fixes the precision of every part.  parse(), plus the primitives a format()
// is written in terms of, live here; the derived specialisation supplies only
// the layout.
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

  // A coefficient that is itself a series has to be bracketed, or the two
  // levels run together: Dual<Dual<double>> printed 1+2e+3+4ee, which reads as
  // four terms of one series rather than two of two.
  void term(std::format_context &ctx, const S &v) const {
    if constexpr (std::is_arithmetic_v<S>) {
      num(ctx, v);
    } else {
      put(ctx, "(");
      num(ctx, v);
      put(ctx, ")");
    }
  }

  // c0+c1e+c2e^2+... -- the perturbation series that both Dual (two terms) and
  // TaylorDual (N+1 of them) are.  Every coefficient is printed, zeros
  // included, so the order of the series is legible in the text.
  void series(std::format_context &ctx, const auto &coeffs) const {
    for (std::size_t k = 0; k != std::ranges::size(coeffs); ++k) {
      if (k > 0) {
        put(ctx, "+");
      }
      term(ctx, coeffs[k]);
      if (k == 1) {
        put(ctx, "e");
      } else if (k > 1) {
        ctx.advance_to(std::format_to(ctx.out(), "e^{}", k));
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

// Anything in the dual family: Dual, TaylorDual.  One inserter serves both, so
// the iostream spelling can never drift from the std::format one.
template <typename T>
concept CDualFamily = detail::is_dual_family_v<std::remove_cvref_t<T>>;

template <CDualFamily D>
std::ostream &operator<<(std::ostream &out, const D &d) {
  return out << std::format("{}", d);
}

} // namespace diff
