#pragma once

#include "util/error.hpp"

#include <format>
#include <stdexcept>
#include <string>
#include <utility>

namespace ddx::py {

// ddx is exception-free and stays that way: libddx is compiled -fno-exceptions
// and throws nothing, and every refusal it makes travels as a result<T>.  This
// module is a *consumer* of it -- like compare/, which strips the same flag --
// and its own translation unit cannot be, for two reasons:
//
//   1. Raising a Python error from C++ is an unwind back to the interpreter.
//      py::error_already_set, which is how an exception escaping a Python
//      callback reaches C++, is itself a C++ exception; pybind11 has no mode
//      without them.
//   2. equation() runs a caller's model while holding scoped_arena.  If that
//      model raises, only the unwind runs the guard that puts the previous
//      thread-local arena back -- otherwise a raising model would poison every
//      one after it.
//
// So the throws live in the last inch, and no throw crosses a -fno-exceptions
// frame: the callback and the guard are both in this TU, and nothing in libddx
// calls back into Python.  A result<T> is still a value everywhere until here.
//
// Inherits ddx::error rather than restating it: the code *is* that type, and
// the message is what its own formatter says -- so nothing here duplicates the
// table in util/error.hpp, and a caller reads `.code` as an errc.
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

template <typename T> [[nodiscard]] T unwrap(result<T> &&r) {
  if (!r) {
    throw PyError{r.error()};
  }
  return std::move(*r);
}

} // namespace ddx::py
