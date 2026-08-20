#include "tests_common.hpp"


// ===========================================================================
// Algebraic simplification (expr/simplify.hpp)
//
// Identity folding runs in the operator factories, so trees are born folded
// and the garbage is never instantiated; canonical ordering of commutative
// operands runs where Equation is built.  The point of all of it is node
// count -- node_cache_t is one slot per node and every sweep visits every one
// -- so these assertions are on node_count_v, at compile time.
// ===========================================================================

namespace {
using diff::detail::canonicalise;
constexpr auto sx = diff::var<"x">;
constexpr auto sy = diff::var<"y">;
constexpr auto sz = diff::var<"z">;
constexpr auto sw = diff::var<"w">;

template <diff::FixedString S, class E> consteval std::size_t d_nodes() {
  return diff::node_count_v<decltype(diff::make_all_constant_except<S>(E{})
                                         .derivative())>;
}
constexpr auto F4 =
    (sx + sy) * (sz - sw) + exp(sx * sz) + sin(sy * sw) + sx * sy * sz * sw;
} // namespace

TEST(Simplify, IdentitiesCollapseDerivativeTrees) {
  // d(x*y)/dx was `1*y_c + x*0` -- seven nodes and seven cache slots for one
  // symbol lookup.
  static_assert(d_nodes<diff::FixedString{"x"}, decltype(sx * sy)>() == 1);

  // The whole F4 Jacobian was 268 nodes across its four rows.
  constexpr std::size_t rows = d_nodes<diff::FixedString{"w"}, decltype(F4)>() +
                               d_nodes<diff::FixedString{"x"}, decltype(F4)>() +
                               d_nodes<diff::FixedString{"y"}, decltype(F4)>() +
                               d_nodes<diff::FixedString{"z"}, decltype(F4)>();
  static_assert(rows == 68);
  static_assert(diff::node_count_v<decltype(F4)> == 26); // f itself is untouched
}

TEST(Simplify, OnlyCompileTimeLiteralsFold) {
  // Lit carries its value in the type, so it folds; a stored Constant holds a
  // number that only exists at run time, so it must not.
  static_assert(diff::detail::is_zero_v<diff::Lit<double, 0>>);
  static_assert(diff::detail::is_zero_v<diff::Lit<double, 0.0>>);
  static_assert(diff::detail::is_one_v<diff::Lit<double, 1>>);
  static_assert(!diff::detail::is_zero_v<diff::Constant<double>>);
  static_assert(!diff::detail::is_one_v<diff::Constant<double>>);

  using X = std::remove_cvref_t<decltype(sx)>;
  auto folded = sx * diff::Lit<double, 1>{}; // x*1 -> x, no node built
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(folded)>, X>);
  auto unfolded = sx * diff::Constant<double>{1.0}; // value unknown until run time
  static_assert(!std::is_same_v<std::remove_cvref_t<decltype(unfolded)>, X>);
  EXPECT_DOUBLE_EQ(unfolded.eval(3.0), 3.0);
}

TEST(Simplify, DualLiteralPairsStayEmpty) {
  // Two Lits meeting at a node fold.  For a dual scalar the folded value cannot
  // go back into the type in general, but 0 and 1 can via the int spelling --
  // and those are the only values differentiation makes.  Without that check
  // the pair falls to the stored-Constant branch and the spine above it stops
  // being empty, which is the thing the int spelling exists to prevent.
  using D = diff::Dual<double>;
  static_assert(std::is_empty_v<decltype(diff::Lit<D, 0>{} + diff::Lit<D, 1>{})>);
  static_assert(std::is_empty_v<decltype(diff::Lit<D, 0>{} * diff::Lit<D, 1>{})>);
  static_assert(std::is_empty_v<decltype(diff::Lit<D, 1>{} + diff::Lit<D, 0>{})>);
  static_assert(std::is_empty_v<decltype(diff::Lit<double, 0>{} + diff::Lit<double, 1>{})>);
  // A value that is neither 0 nor 1 has nowhere to live but a stored Constant.
  EXPECT_DOUBLE_EQ((diff::Lit<D, 1>{} + diff::Lit<D, 1>{}).get().value(), 2.0);
}

TEST(Simplify, MaxMinDerivativeSizeIsOnTheLedger) {
  // The branch-free expansion is not cheap: (a+b±|a-b|)/2 names a-b twice, once
  // bare and once inside the abs, and that duplication is structural so it
  // stays in the type.  Recorded rather than defended -- the rule it replaced
  // did not compile at all, so this is not a regression from anything that
  // worked, and it is the densest target in the codebase for a later CSE pass.
  constexpr auto dmax = d_nodes<diff::FixedString{"x"}, decltype(max(sx, sy))>();
  constexpr auto dmin = d_nodes<diff::FixedString{"x"}, decltype(min(sx, sy))>();
  static_assert(dmax == 14);
  static_assert(dmin == 15);
}

TEST(Simplify, DualValuedTreesAreStatelessToo) {
  // The int spelling of Lit works for a T that can never be an NTTP itself, so
  // the dual-valued path never falls back to a stored Constant -- which would
  // drag the whole spine above it into the storing node form.
  using D = diff::Dual<double>;
  diff::Variable<D, diff::FixedString{"x"}> dx;
  diff::Variable<D, diff::FixedString{"y"}> dy;
  static_assert(std::is_empty_v<decltype(dx * dy)>);
  static_assert(std::is_empty_v<decltype((dx * dy).derivative())>);
  static_assert(sizeof(dx * dy) == 1);
}

TEST(Simplify, CommutativeOperandsCanonicalise) {
  // x+y and y+x become the same type, which is what makes type identity a
  // usable value numbering for a later CSE/DAG pass.  double declares its
  // multiplication commutative, so x*y and y*x unify for it too.
  static_assert(
      std::is_same_v<decltype(canonicalise(sx + sy)), decltype(canonicalise(sy + sx))>);
  static_assert(
      std::is_same_v<decltype(canonicalise(sx * sy)), decltype(canonicalise(sy * sx))>);
  static_assert(std::is_same_v<decltype(canonicalise((sx * sy) * sz)),
                               decltype(canonicalise(sz * (sy * sx)))>);
  // Reordering is exact, so the values do not move.
  EXPECT_DOUBLE_EQ(canonicalise(sy * sx).eval(3.0, 5.0), 15.0);
  EXPECT_DOUBLE_EQ(canonicalise(sx - sy).eval(5.0, 2.0), 3.0);
  // Literals sort ahead of symbols, so a scalar always lands on the left.
  EXPECT_EQ(std::format("{}", canonicalise(sx * 2.0)), "2 * x");
  EXPECT_EQ(std::format("{}", canonicalise(2.0 * sx)), "2 * x");
}

// Two scalars for the commutativity opt-in below.  Both satisfy CFieldLike --
// closure under + - * / and negation is all Numeric asks -- and they are the
// same type but for the tag, so the only thing that differs between them is
// whether they declare that their multiplication commutes.
template <int Tag> struct Ring {
  double v{};
  friend constexpr Ring operator+(Ring a, Ring b) noexcept { return {a.v + b.v}; }
  friend constexpr Ring operator-(Ring a, Ring b) noexcept { return {a.v - b.v}; }
  friend constexpr Ring operator*(Ring a, Ring b) noexcept { return {a.v * b.v}; }
  friend constexpr Ring operator/(Ring a, Ring b) noexcept { return {a.v / b.v}; }
  constexpr Ring operator-() const noexcept { return {-v}; }
};
using Undeclared = Ring<0>;
using Declared = Ring<1>;
DIFF_COMMUTATIVE_MULTIPLY(Ring<1>)

// 2x2 real matrices: a genuine non-commutative ring, and the only kind of
// scalar that can observe which side an adjoint multiplies on.  Scalars embed
// as s*I, which is what makes Lit<Mat2, 0> and Lit<Mat2, 1> -- and the reverse
// sweep's root adjoint T{1} -- come out as the zero and identity matrices.
struct Mat2 {
  double a{}, b{}, c{}, d{};
  constexpr Mat2() = default;
  constexpr Mat2(double s) noexcept : a{s}, d{s} {}
  constexpr Mat2(double a_, double b_, double c_, double d_) noexcept
      : a{a_}, b{b_}, c{c_}, d{d_} {}
  friend constexpr Mat2 operator+(Mat2 x, Mat2 y) noexcept {
    return {x.a + y.a, x.b + y.b, x.c + y.c, x.d + y.d};
  }
  friend constexpr Mat2 operator-(Mat2 x, Mat2 y) noexcept {
    return {x.a - y.a, x.b - y.b, x.c - y.c, x.d - y.d};
  }
  friend constexpr Mat2 operator*(Mat2 x, Mat2 y) noexcept {
    return {x.a * y.a + x.b * y.c, x.a * y.b + x.b * y.d,
            x.c * y.a + x.d * y.c, x.c * y.b + x.d * y.d};
  }
  friend constexpr Mat2 operator/(Mat2 x, Mat2 y) noexcept { // x * y^-1
    const double det = y.a * y.d - y.b * y.c;
    return x * Mat2{y.d / det, -y.b / det, -y.c / det, y.a / det};
  }
  constexpr Mat2 operator-() const noexcept { return {-a, -b, -c, -d}; }
  // Deliberately NO operator+= -- CFieldLike does not require one, and this
  // type existing without it is what pins that the reverse sweep does not
  // secretly demand one either (values.hpp accumulates with + and assignment).
  friend constexpr bool operator==(Mat2, Mat2) noexcept = default;
};

TEST(ReverseMode, MultiplyAdjointRespectsOperandSide) {
  // MultiplyOp::adjoints must return {adj*b, a*adj}, not {adj*b, adj*a}: for
  // c = a*b the differential is da*b + a*db, so the adjoint reaching `b`
  // multiplies on the RIGHT of a.  Every scalar the library ships commutes, so
  // the two spellings are bit-identical for all of them -- this is the only
  // test that can tell them apart, and without it a revert is invisible.
  static_assert(diff::Numeric<Mat2> && !diff::CCommutativeMultiply<Mat2>);

  constexpr diff::Variable<Mat2, diff::FixedString{"x"}> mx;
  constexpr diff::Variable<Mat2, diff::FixedString{"y"}> my;
  constexpr diff::Variable<Mat2, diff::FixedString{"z"}> mz;

  // X and Z do not commute: X*Z = {2,1,1,1} but Z*X = {1,1,1,2}.
  constexpr Mat2 X{1, 1, 0, 1}, Y{1, 2, 3, 4}, Z{1, 0, 1, 1};

  // f = z*(x*y).  Nothing reorders it -- Mat2 never opted in -- so the adjoint
  // arriving at the inner product is Z rather than the identity, which is what
  // makes the two sidings disagree.
  const auto g = diff::Equation{mz * (mx * my)}.gradient(std::array{X, Y, Z});

  // Gradients come back in canonical symbol order: x, y, z.
  EXPECT_EQ(g[0], Z * Y); // dx = adj*y with adj = z
  EXPECT_EQ(g[1], X * Z); // dy = x*adj  <-- the sided one; the bug gives Z*X
  EXPECT_EQ(g[2], X * Y); // dz = adj*(x*y) with adj = I
  // Spelled out, so a failure reports the wrong matrix rather than a mismatch.
  EXPECT_EQ(g[1], (Mat2{2, 1, 1, 1}));
  EXPECT_NE(g[1], (Mat2{1, 1, 1, 2})); // what {adj*b, adj*a} would produce
}

TEST(ReverseMode, DivideAdjointRespectsOperandSide) {
  // DivideOp reads a/b as a*b^-1, so c = a*b^-1 differentiates as
  // da*b^-1 - a*b^-1*db*b^-1: the b-adjoint threads BETWEEN a/b and b^-1 and
  // does not collapse to the -adj*a/(b*b) of the quotient rule.  As with
  // multiply, the two agree for every scalar that ships, so only a
  // non-commutative one can tell them apart.
  constexpr diff::Variable<Mat2, diff::FixedString{"x"}> mx;
  constexpr diff::Variable<Mat2, diff::FixedString{"y"}> my;
  constexpr diff::Variable<Mat2, diff::FixedString{"z"}> mz;

  constexpr Mat2 X{1, 1, 0, 1}, Y{1, 0, 1, 1}, Z{1, 2, 3, 4}; // det Y == 1

  // f = z*(x/y), so the adjoint arriving at the quotient is Z, not the
  // identity -- which is what makes the two spellings disagree.
  const auto g = diff::Equation{mz * (mx / my)}.gradient(std::array{X, Y, Z});

  const Mat2 sided = -((X / Y) * Z) / Y;   // what the sided rule must produce
  const Mat2 quotient = -(Z * X) / (Y * Y); // what the commutative rule gives
  // Guard the guard: if these ever coincided the test below would be vacuous.
  ASSERT_NE(sided, quotient);

  EXPECT_EQ(g[0], Z / Y);   // dx = adj*b^-1 with adj = z
  EXPECT_EQ(g[1], sided);   // dy -- the whole point of the test
  EXPECT_NE(g[1], quotient);
  EXPECT_EQ(g[2], X / Y);   // dz = adj*(x/y) with adj = I
}

// Whether an op template accepts a scalar, WITHOUT instantiating it -- naming a
// constrained specialisation directly is a hard error rather than a
// substitution failure, so the template template parameter is what puts the
// check in a deduced context where the constraint can fail softly.
template <template <class> class Op, class T>
concept OpAccepts = requires { typename Op<T>; };

// Closed under the five operators CFieldLike names, but with no way to spell 1.
class NoIdentity {
  double v{};

public:
  constexpr NoIdentity() = default;
  friend constexpr NoIdentity operator+(NoIdentity, NoIdentity) noexcept { return {}; }
  friend constexpr NoIdentity operator-(NoIdentity, NoIdentity) noexcept { return {}; }
  friend constexpr NoIdentity operator*(NoIdentity, NoIdentity) noexcept { return {}; }
  friend constexpr NoIdentity operator/(NoIdentity, NoIdentity) noexcept { return {}; }
  constexpr NoIdentity operator-() const noexcept { return {}; }
};

TEST(Concepts, NumericDemandsWhatTheSweepsActuallyUse) {
  // Closure under + - * / and negation is not sufficient to differentiate a
  // scalar.  Differentiation manufactures exactly 0 and 1: reverse_sweep seeds
  // its root adjoint with T{1} and Lit<T, V>'s int spelling is T(V).  A type
  // that cannot spell 1 must be rejected by the concept rather than failing
  // somewhere inside a sweep.
  static_assert(std::default_initializable<NoIdentity>);
  static_assert(!std::constructible_from<NoIdentity, int>);
  static_assert(!diff::Numeric<NoIdentity>);

  // Ordering is likewise a real requirement, but only of the three ops that
  // compare their operands -- abs branches on the sign, min and max on which
  // side is larger.  Ring is Numeric and defines no ordering, so those three
  // reject it while everything else in the library still accepts it.
  static_assert(diff::Numeric<Undeclared>);
  static_assert(!std::totally_ordered<Undeclared>);
  static_assert(!OpAccepts<diff::AbsOp, Undeclared>);
  static_assert(!OpAccepts<diff::MaxOp, Undeclared>);
  static_assert(!OpAccepts<diff::MinOp, Undeclared>);
  static_assert(OpAccepts<diff::MultiplyOp, Undeclared>);
  static_assert(OpAccepts<diff::SumOp, Undeclared>);
  // Ordered scalars are accepted by all five, so the guard is narrow.
  static_assert(OpAccepts<diff::AbsOp, double> && OpAccepts<diff::MinOp, double>);

  // The scalars that ship satisfy both, so neither constraint narrows the
  // library's actual surface.
  static_assert(diff::Numeric<double> && std::totally_ordered<double>);
  static_assert(std::totally_ordered<diff::Dual<double>>);
  static_assert(std::totally_ordered<diff::TaylorDual<double, 3>>);
  static_assert(std::totally_ordered<diff::TaylorDual<double, 3>>);
}

TEST(Simplify, MultiplicationCommutesOnlyWhenTheScalarSaysSo) {
  // Numeric admits matrices, quaternions, anything whose product depends on
  // operand order, so commuting a product is asked rather than assumed.  The
  // default is the conservative answer: a type that says nothing does not
  // commute, which costs a CSE share and never a wrong result.
  static_assert(diff::Numeric<Undeclared> && !diff::CCommutativeMultiply<Undeclared>);
  static_assert(diff::Numeric<Declared>);
  static_assert(diff::CCommutativeMultiply<double>);
  static_assert(diff::CCommutativeMultiply<diff::Dual<double>>);
  static_assert(diff::CCommutativeMultiply<diff::TaylorDual<double, 3>>);
  static_assert(diff::CCommutativeMultiply<diff::TaylorDual<double, 3>>);
  // The dual wrappers defer to the scalar underneath rather than asserting for
  // themselves, so an undeclared scalar stays undeclared through a Dual.
  static_assert(!diff::CCommutativeMultiply<diff::Dual<Undeclared>>);
  // DIFF_COMMUTATIVE_MULTIPLY is the spelling for a concrete user type.
  static_assert(diff::CCommutativeMultiply<Declared>);

  // And the trait is what canonicalisation actually consults: same expression,
  // same shape, reordered for one scalar and left alone for the other.
  diff::Variable<Undeclared, diff::FixedString{"x"}> ux;
  diff::Variable<Undeclared, diff::FixedString{"y"}> uy;
  static_assert(
      !std::is_same_v<decltype(canonicalise(ux * uy)), decltype(canonicalise(uy * ux))>);
  // Addition needs no opt-in -- a ring's addition commutes by definition.
  static_assert(
      std::is_same_v<decltype(canonicalise(ux + uy)), decltype(canonicalise(uy + ux))>);
}

TEST(Simplify, MaxAndMinHaveASymbolicDerivative) {
  // max(x,y).derivative() did not compile at all before: the old rule selected
  // between lhs.derivative() and rhs.derivative() with a runtime conditional,
  // and those are two different types.
  const auto dmax_dx =
      diff::make_all_constant_except<diff::FixedString{"x"}>(max(sx, sy))
          .derivative();
  const auto dmin_dx =
      diff::make_all_constant_except<diff::FixedString{"x"}>(min(sx, sy))
          .derivative();
  for (auto [a, b] : {std::pair{3.0, 1.0}, std::pair{1.0, 3.0}}) {
    const auto rmax = Equation{max(sx, sy)}.gradient(std::array{a, b});
    const auto rmin = Equation{min(sx, sy)}.gradient(std::array{a, b});
    EXPECT_DOUBLE_EQ(dmax_dx.eval(diff::named<"x">(a), diff::named<"y">(b)),
                     rmax[0]);
    EXPECT_DOUBLE_EQ(dmin_dx.eval(diff::named<"x">(a), diff::named<"y">(b)),
                     rmin[0]);
  }
}
