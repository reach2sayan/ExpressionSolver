#pragma once
// The public surface of the library, and the whole of it.  Everything else
// lives in diff::impl.  Operators and math functions are found by ADL, so they
// are deliberately not re-exported here.
#include "expr/equation.hpp"

namespace diff {

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

// named<"x">(3.0) -- one keyword argument of a point.
using impl::named;

// Symbolic (evaluate the stored partial trees) vs Reverse (one sweep, no trees).
using impl::DiffMode;

} // namespace diff
