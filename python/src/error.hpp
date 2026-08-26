#pragma once

#include "util/error.hpp"

#include <format>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <utility>

namespace ddx::py {

// ddx is exception-free and stays that way: libddx is compiled -fno-exceptions
// and every refusal it makes travels as a result<T>.  This module is a consumer
// of it, like compare/, and its own translation unit cannot be:
//
//   1. Raising a Python error from C++ is an unwind back to the interpreter,
//      and py::error_already_set is itself a C++ exception.
//   2. equation() runs a caller's model while holding scoped_arena; only the
//      unwind runs the guard that puts the previous thread-local arena back.
//
// No throw crosses a -fno-exceptions frame: the callback and the guard are both
// in this TU, and nothing in libddx calls back into Python.
//
// Inherits ddx::error rather than restating it -- the code *is* that type and
// the message is what its own formatter says, so a caller reads `.code` as an
// errc and nothing here duplicates the table in util/error.hpp.
struct PyError : error, std::runtime_error {
  explicit PyError(error e)
      : error(e), std::runtime_error(std::format("{}", e)) {}
  explicit PyError(errc c) : PyError(error{.code = c}) {}
};

[[noreturn]] inline void fail_with(errc c) { throw PyError{c}; }

inline void unwrap(const result<void> &r) {
  if (!r) {
    throw PyError{r.error()};
  }
}

// Constrained away from void: a result<void> prvalue would otherwise bind here
// in preference to the overload above, and `return std::move(*r)` is not a
// thing to do with one.
template <typename T>
  requires(!std::is_void_v<T>)
[[nodiscard]] T unwrap(result<T> &&r) {
  if (!r) {
    throw PyError{r.error()};
  }
  return std::move(*r);
}

} // namespace ddx::py
