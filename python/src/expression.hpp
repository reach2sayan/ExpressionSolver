#pragma once

#include "error.hpp"
#include "rt/builder.hpp"
#include "rt/expressions.hpp"
#include "rt/opcode.hpp"

#include <format>
#include <memory>
#include <string>
#include <utility>

// The handle Python holds: an RTExpression that owns its arena.  RTExpression
// stores a *non-owning* Builder* because in C++ an arena's lifetime is lexical;
// a Python closure can capture a symbol and outlive everything.  Shared
// ownership is the whole of what this adds.
//
// It holds one rather than deriving from one: pybind11 describes a bound type
// as descr<N, PyExpression>, and a base class is an associated class where a
// member's type is not.  Deriving would put RTExpression's hidden operator+
// into the overload set for descr + descr, instantiating Numeric<descr> whose
// body is `{ a + b }` -- constraint recursion inside pybind11's machinery.
namespace ddx::py {

class PyExpression {
public:
  using Base = rt::RTExpression<double>;
  using Arena = std::shared_ptr<rt::Builder<double>>;

  constexpr PyExpression() = default;

  // Implicit, and the reason `2.0 * x` and `x * 2.0` both work: pybind11 is
  // told about it with implicitly_convertible, and what follows is the base's
  // own folding -- two pending literals never reach a graph.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr PyExpression(double v) : expression_(v) {}
  PyExpression(Base e, Arena a) noexcept
      : expression_(e), arena_(std::move(a)) {}

  [[nodiscard]] constexpr bool poisoned() const noexcept {
    return expression_.poisoned();
  }

  // The id this names in `arena`, or a refusal: an id from another arena would
  // index this one at random.  A pending literal materialises into it.
  [[nodiscard]] rt::NodeId root_in(rt::Builder<double> &arena) const {
    if (arena_ && arena_.get() != &arena) {
      fail_with(errc::no_graph);
    }
    return expression_.id(arena);
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
  // The arithmetic surface below is the only thing that puts an arena back on a
  // new handle, so the two halves of a handle are reachable there and nowhere
  // else.
#define DDX_PY_UNFRIEND(fn, Op, label, ...)                                    \
  friend PyExpression fn(const PyExpression &);
  DDX_UNARY_MATH_TABLE(DDX_PY_UNFRIEND)
  DDX_RT_UNARY_TABLE(DDX_PY_UNFRIEND)
#undef DDX_PY_UNFRIEND
#define DDX_PY_BINFRIEND(fn, Op, label, ...)                                   \
  friend PyExpression fn(const PyExpression &, const PyExpression &);
  DDX_RT_BINARY_TABLE(DDX_PY_BINFRIEND)
  DDX_RT_COMPARE_TABLE(DDX_PY_BINFRIEND)
  DDX_PY_BINFRIEND(gt, , )
  DDX_PY_BINFRIEND(ge, , )
  DDX_PY_BINFRIEND(equal, , )
  DDX_PY_BINFRIEND(unequal, , )
  // The one ternary.
  friend PyExpression select(const PyExpression &, const PyExpression &,
                             const PyExpression &);
#undef DDX_PY_BINFRIEND

  [[nodiscard]] constexpr const Base &base() const noexcept {
    return expression_;
  }

  // Null exactly while this is a pending literal, which names no graph yet.
  [[nodiscard]] constexpr const Arena &arena() const noexcept { return arena_; }

  Base expression_;
  Arena arena_;
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
DDX_RT_COMPARE_TABLE(DDX_PY_BINFN)
DDX_PY_BINFN(gt, , )
DDX_PY_BINFN(ge, , )
DDX_PY_BINFN(equal, , )
DDX_PY_BINFN(unequal, , )
#undef DDX_PY_BINFN_UNUSED

// The ternary: the arena is whichever operand names one, as above.
[[nodiscard]] inline PyExpression select(const PyExpression &c,
                                         const PyExpression &t,
                                         const PyExpression &f) {
  const auto arena = c.arena()   ? c.arena()
                     : t.arena() ? t.arena()
                                 : f.arena();
  return {select(c.base(), t.base(), f.base()), arena};
}
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
