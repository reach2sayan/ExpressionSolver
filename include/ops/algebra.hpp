#pragma once

#include <array>
#include <cstdint>

// The algebraic identities, written once.
//
// Two simplifiers consume these: symbolic/simplify.hpp rewrites *types* while
// an expression is built, and rt/builder.hpp rewrites *node ids* while a graph
// is interned.  The rules are the same either way, so they live here as data
// rather than being spelled twice -- the drift that costs is a rule one side
// learns and the other does not.
//
// Data, not an X-macro: the operation tables in ops/unary_math.hpp have to be
// macros because they emit declarations (enumerators, friend functions, case
// labels).  A rewrite rule emits nothing.  It is a value each side matches
// against in its own representation, which an array does better -- greppable,
// debuggable, and assertable over.
//
// Not IEEE-faithful, deliberately: x*0 -> 0, 0/x -> 0, x/x -> 1 and (n/x)*x
// -> n all disagree with IEEE at 0, Inf and NaN.  They cancel arithmetic the
// derivative rules manufactured rather than anything a user wrote, which is
// the trade every AD library makes.
namespace ddx::impl::algebra {

enum class RuleOp : std::uint8_t { Add, Mul, Div, Pow, Neg };

// What has to hold of the operands.  Each simplifier answers these in its own
// terms: a type trait on one side, a node-id compare on the other.
enum class Pred : std::uint8_t {
  ZeroA,   // a is the literal 0
  ZeroB,   // b is the literal 0
  OneA,    // a is the literal 1
  OneB,    // b is the literal 1
  Same,    // a and b are the same expression
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
  OperandOfA    // a is -x; take x
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

// Order is significant: both simplifiers take the first match, so this reads
// top to bottom exactly as the if-chains it replaced did.
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

    Rule{RuleOp::Neg, Pred::NegatedA, Take::OperandOfA},
    // clang-format on
};

// Neg is the only unary rule; everything else reads two operands.
[[nodiscard]] constexpr bool is_unary(const Rule &r) noexcept {
  return r.op == RuleOp::Neg;
}

} // namespace ddx::impl::algebra
