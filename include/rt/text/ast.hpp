#pragma once

#include "rt/opcode.hpp"
#include "util/error.hpp"
#include "util/export.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// One expression, read but not yet in anyone's arena.  The grammar is compiled
// once, in src/rt/text/parse.cpp, against a scalar type it never names; this is
// the seam that lets it be, and it is why lowering is a template and parsing is
// not.
namespace ddx::rt::text {

inline constexpr std::uint32_t no_term = ~std::uint32_t{0};

// A node before it has a value type: an opcode, its operands, and -- for the
// two leaves -- an index into whichever table of the Ast names it.
struct Term {
  OpCode op = OpCode::Const;
  std::uint32_t a = no_term;
  std::uint32_t b = no_term;
  std::uint32_t leaf = 0; // Var: into names.  Const: into literals.
};

// Children before parents, so one forward pass lowers all of it and nothing
// here recurses.
struct Ast {
  std::vector<Term> terms;
  std::vector<std::string> names;    // free identifiers, first named first
  std::vector<std::string> literals; // as written: read at T, never at double
  std::uint32_t root = no_term;
};

// The whole of the grammar's surface.  A refusal is an errc and no position:
// `error` is one byte by design, and a column does not fit in one.
[[nodiscard]] DDX_API result<Ast> parse(std::string_view source);

} // namespace ddx::rt::text
