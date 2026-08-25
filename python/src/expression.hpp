#pragma once

#include "rt/builder.hpp"
#include "rt/expressions.hpp"
#include "rt/opcode.hpp"

#include <format>
#include <memory>
#include <string>
#include <utility>

// The handle Python holds: an RTExpression that knows its own arena.
//
// RTExpression stores a *non-owning* Builder*, because in C++ an arena's
// lifetime is lexical -- the equation() lambda's.  A Python closure keeps no
// such discipline: it can capture a symbol and outlive everything, and
// builder() is private, so a bare handle can neither be checked nor recovered
// from.  Carrying the arena alongside is the whole of what this adds; the
// arithmetic, the folding and the interning are all RTExpression's.
//
// It holds one rather than deriving from one, and that is not a preference.
// pybind11 describes a bound type as descr<N, PyExpression>, so PyExpression
// becomes a *template argument* of a type pybind11 then applies operator+ to.
// ADL on that descr pulls in the associated classes of every template
// argument -- and a base class is associated where a member's type is not.  So
// deriving puts RTExpression's hidden friend operator+ into the overload set
// for descr + descr, which asks whether descr converts to RTExpression, which
// instantiates Numeric<descr>, whose own body is `{ a + b }` over descrs, which
// re-enters the same overload set.  The constraint recursion is a hard error in
// the middle of pybind11's type machinery.  RTExpression's converting
// constructor already carries a comment about exactly this class of cycle; a
// member keeps PyExpression out of the associated set and the cycle cannot form.
namespace ddx::py {

class PyExpression {
public:
  using Base = rt::RTExpression<double>;

  PyExpression() = default;

  // Implicit, and the reason `2.0 * x` and `x * 2.0` both work: pybind11 is
  // told about it with implicitly_convertible, and what follows is the base's
  // own folding -- two pending literals never reach a graph.
  PyExpression(double v) : expression_(v) {} // NOLINT(google-explicit-constructor)

  PyExpression(Base e, std::shared_ptr<rt::Builder<double>> a) noexcept
      : expression_(e), arena_(std::move(a)) {}

  [[nodiscard]] const Base &base() const noexcept { return expression_; }

  // Null exactly while this is a pending literal, which names no graph yet.
  [[nodiscard]] const std::shared_ptr<rt::Builder<double>> &arena() const {
    return arena_;
  }

  [[nodiscard]] bool poisoned() const noexcept {
    return expression_.poisoned();
  }
  [[nodiscard]] rt::NodeId id(rt::Builder<double> &b) const {
    return expression_.id(b);
  }

  [[nodiscard]] std::string repr() const {
    if (!arena_) {
      return std::format("Expression({})", expression_.literal());
    }
    const rt::NodeId v = expression_.id(*arena_);
    const rt::Node<double> &n = (*arena_)[v];
    switch (n.op) {
    case rt::OpCode::Const:
      return std::format("Expression({})", n.value);
    case rt::OpCode::Var:
      return std::format("Expression({})", arena_->symbols()[n.slot]);
    default:
      // The op and the identity, not the subtree: an id *is* the identity of a
      // subexpression here, and a whole tree is what to_dot() is for.
      return std::format("Expression({} #{})", rt::label_of(n.op), v);
    }
  }

private:
  Base expression_;
  std::shared_ptr<rt::Builder<double>> arena_;
};

// The arithmetic surface, off the same three tables the library builds its own
// from -- so an opcode added there reaches Python without this file changing.
// Each body calls the hidden friend on the member and puts the arena back on.
#define DDX_PY_UNFN(fn, Op, label, ...)                                        \
  [[nodiscard]] inline PyExpression fn(const PyExpression &u) {                \
    return {fn(u.base()), u.arena()};                                          \
  }
DDX_UNARY_MATH_TABLE(DDX_PY_UNFN)
DDX_RT_UNARY_TABLE(DDX_PY_UNFN)
#undef DDX_PY_UNFN

// Whichever operand names a graph, mirroring RTExpression::form: a pending
// literal folds against the other rather than materialising into it.
#define DDX_PY_BINFN(fn, Op, label, ...)                                       \
  [[nodiscard]] inline PyExpression fn(const PyExpression &l,                  \
                                       const PyExpression &r) {                \
    return {fn(l.base(), r.base()), l.arena() ? l.arena() : r.arena()};        \
  }
DDX_RT_BINARY_TABLE(DDX_PY_BINFN)
#undef DDX_PY_BINFN

[[nodiscard]] inline PyExpression operator+(const PyExpression &l,
                                            const PyExpression &r) {
  return add(l, r);
}
[[nodiscard]] inline PyExpression operator*(const PyExpression &l,
                                            const PyExpression &r) {
  return mul(l, r);
}
[[nodiscard]] inline PyExpression operator/(const PyExpression &l,
                                            const PyExpression &r) {
  return div(l, r);
}
[[nodiscard]] inline PyExpression operator-(const PyExpression &u) {
  return neg(u);
}
[[nodiscard]] inline PyExpression operator-(const PyExpression &l,
                                            const PyExpression &r) {
  return l + (-r);
}

} // namespace ddx::py
