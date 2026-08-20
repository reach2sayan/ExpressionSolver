#pragma once
// The public surface of the library, and the whole of it.  Everything else
// lives in ddx::impl.  Operators and math functions are found by ADL, so they
// are deliberately not re-exported here.
#include "expr/equation.hpp"
#include "expr/named_map.hpp"

namespace ddx {

// The only derivative entry point.
using impl::Equation;

// Symbol value types.  A `double` graph answers evaluate()/gradient()/
// jacobian()/derivative_tensor()/univariate_derivative(); hessian() needs `dual`.
using impl::dual;
using impl::dual2nd;

// var<"x">, or "x"_s.
using impl::var;
using impl::sym;
namespace literals = impl::literals;

// eq[idx<1>()] -- the subscript spelling of Equation::get<N>().
using impl::idx;

// named<"x">(3.0) -- one keyword argument of a point, or one map entry.
// NamedValue is the type behind it: NamedValue{"x"_s, 3.0}, NamedValue{x, 3.0}.
using impl::named;
using impl::NamedValue;

// map(named<"n">(3), named<"x">(1.5)) -- a compile-time map of those entries,
// and Map{...}, the brace spelling of the same thing.
using impl::map;
using impl::Map;

// Symbolic (evaluate the stored partial trees) vs Reverse (one sweep, no trees).
using impl::DiffMode;

} // namespace ddx
