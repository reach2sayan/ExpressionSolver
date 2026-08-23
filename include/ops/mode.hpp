#pragma once
// Which sweep a derivative is taken with :
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
