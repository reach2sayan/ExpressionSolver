#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <format>
#include <ostream>
#include <string_view>
#include <utility>

namespace ddx {

// Everything the library can refuse at run time -- small, because almost every
// misuse is a static_assert instead.  One row per error: the enumerator, and
// the text the formatter prints for it.  A binding that mirrors this enum
// expands the same table rather than restating it.
#define DDX_ERRC_TABLE(X)                                                      \
  /* A point that does not match the expression's symbols. */                  \
  X(short_point,                                                               \
    "the point supplies fewer values than the expression has symbols")         \
  X(wrong_arity,                                                               \
    "the point supplies one value per symbol, and this one does not")          \
  X(unknown_symbol, "no symbol of that name")                                  \
  X(index_out_of_range, "the index does not name a symbol of this expression") \
  X(wrong_column_count, "wrong number of output columns")                      \
  /* Runtime graphs. */                                                        \
  X(no_arena,                                                                  \
    "no arena is current -- name symbols inside ddx::rt::equation()")          \
  X(no_graph, "the expression has no graph (a bare literal names none)")       \
  X(not_univariate, "needs exactly one symbol")                                \
  /* The JIT.  Carried with a message by jit::error, which is not this type.   \
   */                                                                          \
  X(jit_target, "the JIT could not bring up a target for this host")           \
  X(jit_module, "the JIT could not take the emitted module")                   \
  X(jit_verify, "the emitted module failed verification")                      \
  X(jit_lookup, "the compiled kernel has no such symbol")

enum class errc : std::uint8_t {
#define DDX_ERRC_ENUMERATOR(name, text) name,
  DDX_ERRC_TABLE(DDX_ERRC_ENUMERATOR)
#undef DDX_ERRC_ENUMERATOR
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

// Generated from the table, so the messages sit in enumerator order and
// message() is an index rather than a switch.  A binding reads it by
// enumerator to carry the library's own text.
inline constexpr std::array kMessages{
#define DDX_ERRC_MESSAGE(name, text) std::string_view{text},
    DDX_ERRC_TABLE(DDX_ERRC_MESSAGE)
#undef DDX_ERRC_MESSAGE
};

[[nodiscard]] constexpr std::string_view message(const errc c) noexcept {
  const auto i = static_cast<std::size_t>(c);
  return i < kMessages.size() ? kMessages[i] : "?";
}
} // namespace detail

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
    return std::formatter<std::string_view, char>::format(
        ddx::detail::message(c), ctx);
  }
};

template <>
struct std::formatter<ddx::error, char> : std::formatter<ddx::errc, char> {
  auto format(const ddx::error e, std::format_context &ctx) const {
    return std::formatter<ddx::errc, char>::format(e.code, ctx);
  }
};
