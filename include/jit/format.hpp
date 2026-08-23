#pragma once
// Printing a compiled module's IR.  This lived at the tail of expr/format.hpp
// behind an #ifdef DDX_HAS_JIT, which made the compile-time layer name a
// header from the JIT -- the one edge pointing the wrong way through the
// stack.  It belongs here, where the type it formats is defined.
//
// The machinery is the same as the expression formatter's: fmt_put writes a
// string_view into whatever context it is handed.

#include "jit/kernel.hpp"
#include "util/fmt.hpp" // fmt_put

#include <format>
#include <ostream>
#include <string_view>

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
