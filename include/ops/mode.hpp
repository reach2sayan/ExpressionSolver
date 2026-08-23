#pragma once
// Which sweep a derivative is taken with.  It lives down here, with the
// operation vocabulary, because both graphs answer to it: the compile-time
// drivers in drivers/symbolic.hpp and the runtime ones in rt/derivative.hpp.
// Before it was extracted, rt/ included the whole compile-time driver tree --
// duals, tensors and all -- for this one enumeration.
//
// Forward mode is reached through the drivers, which take a callable rather
// than a Mode, so it is not a value here.
//
// Reverse is the default: one sweep yields every partial, at a cost
// bounded by a constant times the function itself however many inputs there
// are, where forward mode needs one sweep per input.
// Ref: Baur & Strassen, Theoret. Comput. Sci. 22(3) (1983) 317.

namespace ddx::impl {

enum class DiffMode { Symbolic, Reverse };

} // namespace ddx::impl
