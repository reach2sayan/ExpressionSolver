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
    Rule{RuleOp::Mul, Pred::ZeroA,    Take::LitZero},
    Rule{RuleOp::Mul, Pred::ZeroB,    Take::LitZero},
    Rule{RuleOp::Mul, Pred::OneA,     Take::OperandB},
    Rule{RuleOp::Mul, Pred::OneB,     Take::OperandA},
    Rule{RuleOp::Mul, Pred::AOverB,   Take::NumeratorOfA},
    Rule{RuleOp::Mul, Pred::BOverA,   Take::NumeratorOfB, true},

    Rule{RuleOp::Add, Pred::ZeroA,    Take::OperandB},
    Rule{RuleOp::Add, Pred::ZeroB,    Take::OperandA},

    Rule{RuleOp::Div, Pred::Same,     Take::LitOne},
    Rule{RuleOp::Div, Pred::ZeroA,    Take::LitZero},
    Rule{RuleOp::Div, Pred::OneB,     Take::OperandA},

    Rule{RuleOp::Pow, Pred::ZeroB,    Take::LitOne},
    Rule{RuleOp::Pow, Pred::OneB,     Take::OperandA},
    // x^2 -> x*x is the one integer power that is exact: a square is a single
    // correctly-rounded multiply, so it *is* pow's answer.  x^3 is not -- two
    // roundings against one -- which is why nothing here generalises it, and
    // why LLVM only contracts the square without fast-math.  Worth a rule
    // because the interpreter otherwise calls libm, and because the reverse
    // rules manufacture pow(u, r-1) for every integer power a model writes.
    Rule{RuleOp::Pow, Pred::TwoB,     Take::SquareOfA},

    Rule{RuleOp::Neg, Pred::NegatedA, Take::OperandOfA},
    // clang-format on
};

// Neg is the only unary rule; everything else reads two operands.
[[nodiscard]] constexpr bool is_unary(const Rule &r) noexcept {
  return r.op == RuleOp::Neg;
}

} // namespace ddx::impl::algebra
