#pragma once

#include <array>
#include <cstdint>

// The algebraic identities, written once.
namespace ddx::impl::algebra {

enum class RuleOp : std::uint8_t { Add, Mul, Div, Pow, Neg };

// Each simplifier answers these in its own terms: a type trait on one side, a
// node-id compare on the other.
enum class Pred : std::uint8_t {
  ZeroA,   // a is the literal 0
  ZeroB,   // b is the literal 0
  OneA,    // a is the literal 1
  OneB,    // b is the literal 1
  Same,    // a and b are the same expression
  TwoB,    // b is the literal 2
  AOverB,  // a is a quotient whose denominator is exactly b
  BOverA,  // b is a quotient whose denominator is exactly a
  NegatedA // a is itself a negation
};

// What the node becomes.
enum class Take : std::uint8_t {
  LitZero,
  LitOne,
  OperandA,
  OperandB,
  NumeratorOfA, // a is n/b; take n
  NumeratorOfB, // b is n/a; take n
  OperandOfA,   // a is -x; take x
  SquareOfA     // a * a
};

struct Rule {
  RuleOp op;
  Pred when;
  Take then;
  // a * (n/a) is a*n*a^-1, which is n only where the factors commute.  The
  // mirrored (n/a) * a needs no such guard: `/` is right division, so the
  // a^-1 already meets the a.
  bool needs_commutative_multiply = false;
};

// Order is significant: both simplifiers take the first match.
inline constexpr std::array kRules{
    // clang-format off
    Rule{.op = RuleOp::Mul, .when = Pred::ZeroA,  .then = Take::LitZero},
    Rule{.op = RuleOp::Mul, .when = Pred::ZeroB,  .then = Take::LitZero},
    Rule{.op = RuleOp::Mul, .when = Pred::OneA,   .then = Take::OperandB},
    Rule{.op = RuleOp::Mul, .when = Pred::OneB,   .then = Take::OperandA},
    Rule{.op = RuleOp::Mul, .when = Pred::AOverB, .then = Take::NumeratorOfA},
    Rule{.op = RuleOp::Mul, .when = Pred::BOverA, .then = Take::NumeratorOfB, .needs_commutative_multiply = true},

    Rule{.op = RuleOp::Add, .when = Pred::ZeroA,  .then = Take::OperandB},
    Rule{.op = RuleOp::Add, .when = Pred::ZeroB,  .then = Take::OperandA},

    Rule{.op = RuleOp::Div, .when = Pred::Same,   .then = Take::LitOne},
    Rule{.op = RuleOp::Div, .when = Pred::ZeroA,  .then = Take::LitZero},
    Rule{.op = RuleOp::Div, .when = Pred::OneB,   .then = Take::OperandA},

    Rule{.op = RuleOp::Pow, .when = Pred::ZeroB,  .then = Take::LitOne},
    Rule{.op = RuleOp::Pow, .when = Pred::OneB,   .then = Take::OperandA},
    // The one integer power that is exact: a square is a single correctly-
    // rounded multiply, so it *is* pow's answer, where x^3 is two roundings
    // against one.  Worth a rule because the reverse rules manufacture
    // pow(u, r-1) for every integer power a model writes.
    Rule{.op = RuleOp::Pow, .when = Pred::TwoB,     .then = Take::SquareOfA},

    Rule{.op = RuleOp::Neg, .when = Pred::NegatedA, .then = Take::OperandOfA},
    // clang-format on
};

// Neg is the only unary rule; everything else reads two operands.
[[nodiscard]] constexpr bool is_unary(const Rule &r) noexcept {
  return r.op == RuleOp::Neg;
}

} // namespace ddx::impl::algebra
