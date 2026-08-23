#include "dual/tests_dual_common.hpp"
#include "tests_simplify_fixtures.hpp"

TEST(Simplify, DualLiteralPairsStayEmpty) {
  // For a dual scalar the folded value cannot go back into the type in general,
  // but 0 and 1 can via the int spelling -- and those are the only values
  // differentiation makes.  Without that the pair falls to the stored-Constant
  // branch and the spine above it stops being empty.
  using D = ddx::impl::Dual<double>;
  static_assert(std::is_empty_v<decltype(ddx::impl::Lit<D, 0>{} +
                                         ddx::impl::Lit<D, 1>{})>);
  static_assert(std::is_empty_v<decltype(ddx::impl::Lit<D, 0>{} *
                                         ddx::impl::Lit<D, 1>{})>);
  static_assert(std::is_empty_v<decltype(ddx::impl::Lit<D, 1>{} +
                                         ddx::impl::Lit<D, 0>{})>);
  static_assert(std::is_empty_v<decltype(ddx::impl::Lit<double, 0>{} +
                                         ddx::impl::Lit<double, 1>{})>);
  // A value that is neither 0 nor 1 has nowhere to live but a stored Constant.
  EXPECT_DOUBLE_EQ(
      (ddx::impl::Lit<D, 1>{} + ddx::impl::Lit<D, 1>{}).get().value(), 2.0);
}
TEST(Simplify, DualValuedTreesAreStatelessToo) {
  // The int spelling of Lit works for a T that can never be an NTTP itself, so
  // the dual-valued path never falls back to a stored Constant -- which would
  // drag the whole spine above it into the storing node form.
  using D = ddx::impl::Dual<double>;
  ddx::impl::Variable<D, ddx::impl::FixedString{"x"}> dx;
  ddx::impl::Variable<D, ddx::impl::FixedString{"y"}> dy;
  static_assert(std::is_empty_v<decltype(dx * dy)>);
  static_assert(std::is_empty_v<decltype((dx * dy).derivative())>);
  static_assert(sizeof(dx * dy) == 1);
}
TEST(Concepts, NumericDemandsWhatTheSweepsActuallyUse) {
  // Differentiation manufactures exactly 0 and 1: reverse_sweep seeds its root
  // adjoint with T{1} and Lit<T, V>'s int spelling is T(V).  A type that cannot
  // spell 1 must be rejected by the concept rather than deep inside a sweep.
  static_assert(std::default_initializable<NoIdentity>);
  static_assert(!std::constructible_from<NoIdentity, int>);
  static_assert(!ddx::impl::Numeric<NoIdentity>);

  // Ordering is required only of the three ops that compare their operands:
  // abs branches on the sign, min and max on which side is larger.
  static_assert(ddx::impl::Numeric<Undeclared>);
  static_assert(!std::totally_ordered<Undeclared>);
  static_assert(!OpAccepts<ddx::impl::AbsOp, Undeclared>);
  static_assert(!OpAccepts<ddx::impl::MaxOp, Undeclared>);
  static_assert(!OpAccepts<ddx::impl::MinOp, Undeclared>);
  static_assert(OpAccepts<ddx::impl::MultiplyOp, Undeclared>);
  static_assert(OpAccepts<ddx::impl::SumOp, Undeclared>);
  // Ordered scalars are accepted by all five, so the guard is narrow.
  static_assert(OpAccepts<ddx::impl::AbsOp, double> &&
                OpAccepts<ddx::impl::MinOp, double>);

  // The scalars that ship satisfy both, so neither constraint narrows the
  // library's actual surface.
  static_assert(ddx::impl::Numeric<double> && std::totally_ordered<double>);
  static_assert(std::totally_ordered<ddx::impl::Dual<double>>);
  static_assert(std::totally_ordered<ddx::impl::TaylorDual<double, 3>>);
  static_assert(std::totally_ordered<ddx::impl::TaylorDual<double, 3>>);
}
TEST(Simplify, MultiplicationCommutesOnlyWhenTheScalarSaysSo) {
  // Numeric admits anything whose product depends on operand order, so a type
  // that says nothing does not commute: that costs a CSE share, never a wrong
  // result.
  static_assert(ddx::impl::Numeric<Undeclared> &&
                !ddx::impl::CCommutativeMultiply<Undeclared>);
  static_assert(ddx::impl::Numeric<Declared>);
  static_assert(ddx::impl::CCommutativeMultiply<double>);
  static_assert(ddx::impl::CCommutativeMultiply<ddx::impl::Dual<double>>);
  static_assert(
      ddx::impl::CCommutativeMultiply<ddx::impl::TaylorDual<double, 3>>);
  static_assert(
      ddx::impl::CCommutativeMultiply<ddx::impl::TaylorDual<double, 3>>);
  // The dual wrappers defer to the scalar underneath rather than asserting for
  // themselves, so an undeclared scalar stays undeclared through a Dual.
  static_assert(!ddx::impl::CCommutativeMultiply<ddx::impl::Dual<Undeclared>>);
  // DDX_COMMUTATIVE_MULTIPLY is the spelling for a concrete user type.
  static_assert(ddx::impl::CCommutativeMultiply<Declared>);

  // And the trait is what canonicalisation actually consults: same expression,
  // same shape, reordered for one scalar and left alone for the other.
  ddx::impl::Variable<Undeclared, ddx::impl::FixedString{"x"}> ux;
  ddx::impl::Variable<Undeclared, ddx::impl::FixedString{"y"}> uy;
  static_assert(!std::is_same_v<decltype(canonicalise(ux * uy)),
                                decltype(canonicalise(uy * ux))>);
  // Addition needs no opt-in -- a ring's addition commutes by definition.
  static_assert(std::is_same_v<decltype(canonicalise(ux + uy)),
                               decltype(canonicalise(uy + ux))>);
}
