#pragma once
// The public surface of the library, and the whole of it.  Everything else
// lives in ddx::impl.  Operators and math functions are found by ADL, so they
// are deliberately not re-exported here.
//
// ---------------------------------------------------------------------------
// Literature
//
// The algorithms are standard; these are where they come from.  Citations for
// one specific derivation sit at the code that implements it -- grep "Ref:".
// REFERENCES.md annotates this list, and docs/ works examples by hand.
//
// Forward mode and dual numbers
//   Clifford, "Preliminary Sketch of Biquaternions", Proc. LMS s1-4 (1873) 381
//     -- adjoin ε with ε² = 0; the ε part of a product is the product rule.
//   Wengert, "A Simple Automatic Derivative Evaluation Program", CACM 7(8)
//     (1964) 463 -- forward mode, in two pages.
//   Fike & Alonso, "The Development of Hyper-Dual Numbers for Exact
//     Second-Derivative Calculations", AIAA 2011-886 -- nesting them.
//
// Reverse mode
//   Linnainmaa, "Taylor Expansion of the Accumulated Rounding Error", BIT
//     16(2) (1976) 146 -- the first published reverse sweep.
//   Baur & Strassen, "The Complexity of Partial Derivatives", Theoret. Comput.
//     Sci. 22(3) (1983) 317 -- the whole gradient for a constant times the
//     cost of the function, whatever the input count.
//
// Higher order
//   Berz, "Differential Algebraic Description of Beam Dynamics to Very High
//     Orders", Particle Accelerators 24 (1989) 109 -- truncated-polynomial
//     arithmetic, the view TaylorDual takes.
//   Knuth, TAOCP Vol. 2 (3rd ed.) §4.7 -- power-series product and division.
//   Jorba & Zou, "A Software Package for the Numerical Integration of ODEs by
//     Means of High-Order Taylor Methods", Exp. Math. 14(1) (2005) 99 -- the
//     elementary-function recurrence table.
//   Pearlmutter, "Fast Exact Multiplication by the Hessian", Neural
//     Computation 6(1) (1994) 147 -- forward over reverse.
//   Gebremedhin, Manne & Pothen, "What Color Is Your Jacobian?", SIAM Review
//     47(4) (2005) 629 -- recovering a sparse derivative from few sweeps.
//
// General
//   Griewank & Walther, "Evaluating Derivatives" (2nd ed., SIAM, 2008) -- the
//     standard reference.  Ch. 3 both modes, Ch. 13 Taylor and tensors.
//   Baydin, Pearlmutter, Radul & Siskind, "Automatic Differentiation in
//     Machine Learning: a Survey", JMLR 18(153) (2018).
// ---------------------------------------------------------------------------
#include "expr/equation.hpp"
#include "expr/named_map.hpp"

namespace ddx {

// The only derivative entry point.
using impl::Equation;

// Symbol value types.  A `double` graph answers evaluate()/gradient()/
// jacobian()/derivative_tensor()/univariate_derivative(); hessian() needs
// `dual`.
using impl::dual;
using impl::dual2nd;

// var<"x">, or "x"_s.  var_of<"x">(v) / dual_var_of<"x">(v) take the scalar
// type from an exemplar value instead of naming it.
using impl::dual_var_of;
using impl::sym;
using impl::var;
using impl::var_of;
namespace literals = impl::literals;

// constant(3.0) -- a value stored in the tree.
using impl::constant;

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

// Symbolic (evaluate the stored partial trees) vs Reverse (one sweep, no
// trees).
using impl::DiffMode;

} // namespace ddx
