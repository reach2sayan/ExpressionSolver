#pragma once

#include <cstdint>
#include <expected>
#include <format>
#include <ostream>
#include <string_view>
#include <utility>

namespace ddx {

// Everything the library can refuse at run time -- small, because almost every
// misuse is a static_assert instead.
enum class errc : std::uint8_t {
  // A point that does not match the expression's symbols.
  short_point,
  wrong_arity,
  unknown_symbol,
  index_out_of_range,
  wrong_column_count,
  // Runtime graphs.
  no_arena,
  no_graph,
  not_univariate,
  // The JIT.  Carried with a message by jit::error, which is not this type.
  jit_target,
  jit_module,
  jit_verify,
  jit_lookup,
};

// No message string and no source location: an error travels the numeric path,
// where an allocation is as unwelcome as the throw it replaces.  The text sits
// in a static table that the formatter below reads, so printing one costs the
// same as printing a string_view.
struct error {
  errc code;
  [[nodiscard]] friend constexpr bool operator==(error,
                                                 error) noexcept = default;
};

namespace detail {
[[nodiscard]] constexpr std::string_view message(const errc c) noexcept {
  switch (c) {
  case errc::short_point:
    return "the point supplies fewer values than the expression has symbols";
  case errc::wrong_arity:
    return "the point supplies one value per symbol, and this one does not";
  case errc::unknown_symbol:
    return "no symbol of that name";
  case errc::index_out_of_range:
    return "the index does not name a symbol of this expression";
  case errc::wrong_column_count:
    return "wrong number of output columns";
  case errc::no_arena:
    return "no arena is current -- name symbols inside ddx::rt::equation()";
  case errc::no_graph:
    return "the expression has no graph (a bare literal names none)";
  case errc::not_univariate:
    return "needs exactly one symbol";
  case errc::jit_target:
    return "the JIT could not bring up a target for this host";
  case errc::jit_module:
    return "the JIT could not take the emitted module";
  case errc::jit_verify:
    return "the emitted module failed verification";
  case errc::jit_lookup:
    return "the compiled kernel has no such symbol";
  }
  std::unreachable();
}
} // namespace impl

inline std::ostream &operator<<(std::ostream &out, const errc c) {
  return out << detail::message(c);
}

inline std::ostream &operator<<(std::ostream &out, const error e) {
  return out << e.code;
}

template <typename T> using result = std::expected<T, error>;
[[nodiscard]] constexpr std::unexpected<error> fail(const errc c) noexcept {
  return std::unexpected{error{.code = c}};
}

} // namespace ddx

// Deriving from the string_view formatter, so a caller's "{:>16}" reaches the
// text.
template <>
struct std::formatter<ddx::errc, char>
    : std::formatter<std::string_view, char> {
  auto format(const ddx::errc c, std::format_context &ctx) const {
    return std::formatter<std::string_view, char>::format(ddx::detail::message(c),
                                                          ctx);
  }
};

template <>
struct std::formatter<ddx::error, char> : std::formatter<ddx::errc, char> {
  auto format(const ddx::error e, std::format_context &ctx) const {
    return std::formatter<ddx::errc, char>::format(e.code, ctx);
  }
};
