#pragma once

#include <cstdint>
#include <expected>
#include <string_view>

namespace ddx {

// Everything the library can refuse at run time.  The set is small because
// almost every misuse is a static_assert instead: these are the ones whose
// shape is not known until a point, a span or an LLVM target actually arrives.
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

// No message string and no source location: an error travels through the
// numeric path, where an allocation would be as unwelcome as the throw it
// replaces.  message() recovers the text from a static table.
struct error {
  errc code;

  [[nodiscard]] friend constexpr bool operator==(error,
                                                 error) noexcept = default;
};

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
  return "unknown error";
}

[[nodiscard]] constexpr std::string_view message(const error e) noexcept {
  return message(e.code);
}

// What every fallible spelling returns.  The infallible ones -- positional and
// named points, whose arity is a static_assert -- keep returning T.
template <typename T> using result = std::expected<T, error>;

[[nodiscard]] constexpr std::unexpected<error> fail(const errc c) noexcept {
  return std::unexpected{error{c}};
}

} // namespace ddx
