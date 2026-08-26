#pragma once

#include "rt/apply.hpp"
#include "rt/builder.hpp"
#include "rt/opcode.hpp"
#include "util/error.hpp"
#include "util/scope_guard.hpp"

#include <concepts> // std::same_as
#include <optional>
#include <string_view>
#include <type_traits> // std::remove_cvref_t
#include <utility>     // std::move

namespace ddx::impl {
// Befriended below: an Equation is the one thing that takes an expression's
// arena over, and the only reader of the poison an absent arena leaves.
template <typename... Ts> class Equation;
} // namespace ddx::impl

namespace ddx::rt {

template <impl::Numeric T = double> class RTExpression;

// Declared ahead of RTExpression, which befriends it: `var` is where a missing
// arena turns into a poisoned expression.
template <impl::Numeric T = double>
[[nodiscard]] RTExpression<T> var(std::string_view name) noexcept;

// The primitive the one above reaches: it makes poison as well, for the arena
// that has stopped taking symbols.
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> var(Builder<T> &b,
                                            std::string_view name);

// A handle onto one node, plus the arithmetic surface that builds more.  A null
// builder is a literal not yet given to a graph, which is what lets
// RTExpression{1} exist for CFieldLike's constructible_from<int>.
template <impl::Numeric T> class RTExpression {
public:
  using value_type = T;
  using builder_type = Builder<T>;

  constexpr RTExpression() = default;

  // By value: RTExpression is itself Numeric, and a Numeric&& would out-match
  // the copy constructor for a non-const lvalue.  Prevent circular check
  // clang diagnoses the cycle where GCC accepts it.
  template <typename V>
    requires(!std::same_as<std::remove_cvref_t<V>, RTExpression> &&
             impl::Numeric<V>)
  constexpr RTExpression(V v) : lit_(static_cast<T>(std::move(v))) {}

  constexpr RTExpression(Builder<T> &b, NodeId id) noexcept
      : builder_(&b), id_(id) {}

  [[nodiscard]] constexpr bool pending() const noexcept {
    return builder_ == nullptr;
  }

  // Whether the symbol named no slot -- no arena current, or a sealed one --
  // which surfaces as an errc out of equation() rather than as a zero.
  [[nodiscard]] constexpr bool poisoned() const noexcept {
    return why_.has_value();
  }
  [[nodiscard]] constexpr const T &literal() const noexcept { return lit_; }

  // Give the node an identity in `b`, materialising a pending literal.
  [[nodiscard]] constexpr NodeId id(Builder<T> &b) const {
    return builder_ ? id_ : b.constant(lit_);
  }

  // Two pending literals fold without reaching a graph, so the constants the
  // derivative rules produce never become nodes.
  [[nodiscard]] static constexpr RTExpression form(OpCode op,
                                                   const RTExpression &u) {
    if (u.why_) {
      return poison(*u.why_);
    }
    Builder<T> *const b = u.builder();
    return b ? RTExpression{*b, b->make(op, u.id(*b))}
             : RTExpression{apply<T>(op, u.literal())};
  }

  [[nodiscard]] static constexpr RTExpression
  form(OpCode op, const RTExpression &l, const RTExpression &r) {
    if (l.why_ || r.why_) {
      return poison(l.why_ ? *l.why_ : *r.why_);
    }
    Builder<T> *const b = l.builder() ? l.builder() : r.builder();
    return b ? RTExpression{*b, b->make(op, l.id(*b), r.id(*b))}
             : RTExpression{apply<T>(op, l.literal(), r.literal())};
  }

  // Hidden friends: ordinary functions for a given RTExpression<T>, so `x * 2`
  // and `2 * x` both convert.
  friend constexpr RTExpression operator+(const RTExpression &l,
                                          const RTExpression &r) {
    return form(OpCode::Add, l, r);
  }
  friend constexpr RTExpression operator*(const RTExpression &l,
                                          const RTExpression &r) {
    return form(OpCode::Mul, l, r);
  }
  friend constexpr RTExpression operator/(const RTExpression &l,
                                          const RTExpression &r) {
    return form(OpCode::Div, l, r);
  }
  friend constexpr RTExpression operator-(const RTExpression &u) {
    return form(OpCode::Neg, u);
  }
  friend constexpr RTExpression operator-(const RTExpression &l,
                                          const RTExpression &r) {
    return l + (-r);
  }

  // Through the operators above, never into builder_ and id_ directly: form()
  // propagates poison and folds pending literals, where an in-place version
  // would take a poisoned accumulator, adopt the other side's arena and answer
  // instead of refusing.  The parameter is the class type rather than a Numeric
  // auto, so += accepts exactly what + accepts.
  constexpr RTExpression &operator+=(const RTExpression &o) {
    return *this = *this + o;
  }
  constexpr RTExpression &operator-=(const RTExpression &o) {
    return *this = *this - o;
  }
  constexpr RTExpression &operator*=(const RTExpression &o) {
    return *this = *this * o;
  }
  constexpr RTExpression &operator/=(const RTExpression &o) {
    return *this = *this / o;
  }

#define DDX_RT_UNFN(fn, Op, label, ...)                                        \
  friend constexpr RTExpression fn(const RTExpression &u) {                    \
    return form(OpCode::Op, u);                                                \
  }
  DDX_UNARY_MATH_TABLE(DDX_RT_UNFN)
  DDX_RT_UNARY_TABLE(DDX_RT_UNFN)
#undef DDX_RT_UNFN

#define DDX_RT_BINFN(fn, Op, label, ...)                                       \
  friend constexpr RTExpression fn(const RTExpression &l,                      \
                                   const RTExpression &r) {                    \
    return form(OpCode::Op, l, r);                                             \
  }
  DDX_RT_BINARY_TABLE(DDX_RT_BINFN)
#undef DDX_RT_BINFN

private:
  // What an Equation needs of an expression: which arena it names, and --
  // through poison() -- that it names none.
  template <typename... Ts> friend class impl::Equation;
  template <impl::Numeric U>
  friend RTExpression<U> var(std::string_view) noexcept;
  template <impl::Numeric U>
  friend constexpr RTExpression<U> var(Builder<U> &, std::string_view);

  [[nodiscard]] constexpr Builder<T> *builder() const noexcept {
    return builder_;
  }

  // Not a pending literal, which would materialise into the first graph it
  // meets and put a zero where a symbol should be.
  [[nodiscard]] static constexpr RTExpression poison(errc why) noexcept {
    RTExpression e;
    e.why_ = why;
    return e;
  }

  // What an Equation reports rather than building over a symbol that is not
  // there.
  [[nodiscard]] constexpr std::optional<errc> why() const noexcept {
    return why_;
  }

  Builder<T> *builder_ = nullptr;
  NodeId id_ = no_node;
  T lit_{};
  std::optional<errc> why_;
};

template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> var(Builder<T> &b,
                                            std::string_view name) {
  const NodeId id = b.variable(name);
  return id == no_node ? RTExpression<T>::poison(errc::sealed_arena)
                       : RTExpression<T>{b, id};
}
template <impl::Numeric T>
[[nodiscard]] constexpr RTExpression<T> lit(Builder<T> &b, const T &v) {
  return RTExpression<T>{b, b.constant(v)};
}

// The arena var() registers into while an equation() callback runs.  Thread-
// local, and held by the same guard the drivers seed derivatives with.
namespace detail {

// Spell one type N times in a pack expansion.
template <std::size_t, impl::Numeric T> using Repeat = RTExpression<T>;

template <impl::Numeric T>
inline thread_local Builder<T> *current_arena = nullptr;

// The runtime counterpart of scoped_seed; the arena is not a constant, so it
// comes in as an argument.
template <impl::Numeric T>
[[nodiscard]] auto scoped_arena(Builder<T> &b) noexcept {
  return impl::scoped_value<Builder<T> *>{current_arena<T>, &b};
}

} // namespace detail

// The spelling a model uses; the two-argument form above is the primitive.
template <impl::Numeric T>
[[nodiscard]] RTExpression<T> var(std::string_view name) noexcept {
  return (detail::current_arena<T> == nullptr)
             ? RTExpression<T>::poison(errc::no_arena)
             : var(*detail::current_arena<T>, name);
}

} // namespace ddx::rt

namespace ddx::impl {
// A node stands for a value in S, so its product commutes exactly when S's
// does.  Load-bearing: DivideOpFn asks it to pick between the two spellings of
// the quotient rule.
template <Numeric S>
inline constexpr bool is_commutative_multiply_v<rt::RTExpression<S>> =
    is_commutative_multiply_v<S>;
} // namespace ddx::impl
