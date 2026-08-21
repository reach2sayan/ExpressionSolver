#pragma once

#include "expr/expressions.hpp"
#include "expr/operations.hpp"
#include "expr/traits.hpp"
#include "rt/expr.hpp"

#include <array>
#include <tuple>
#include <utility>

// Lowering a compile-time ddx tree into the runtime graph.  The structure lives
// in the type, so the walk is a compile-time recursion that emits nodes; what
// it buys is a differential test, since the same expression can then be
// evaluated by Equation and by the graph and the two must agree.
namespace ddx::rt {

namespace detail {

template <typename Op> struct op_code_of;

#define DDX_RT_MAP_OP(DdxOp, Code)                                             \
  template <impl::Numeric T> struct op_code_of<impl::DdxOp<T>> {               \
    static constexpr OpCode value = OpCode::Code;                              \
  };
DDX_RT_MAP_OP(SumOp, Add)
DDX_RT_MAP_OP(MultiplyOp, Mul)
DDX_RT_MAP_OP(DivideOp, Div)
DDX_RT_MAP_OP(NegateOp, Neg)
DDX_RT_MAP_OP(PowOp, Pow)
DDX_RT_MAP_OP(Atan2Op, Atan2)
DDX_RT_MAP_OP(HypotOp, Hypot)
DDX_RT_MAP_OP(MaxOp, Max)
DDX_RT_MAP_OP(MinOp, Min)
DDX_RT_MAP_OP(AbsOp, Abs)
DDX_RT_MAP_OP(SignOp, Sign)
#undef DDX_RT_MAP_OP

// The eighteen share their spelling with the enumerators, by construction.
#define DDX_RT_MAP_MATH(fn, Op, label)                                         \
  template <impl::Numeric T> struct op_code_of<impl::Op<T>> {                  \
    static constexpr OpCode value = OpCode::Op;                                \
  };
DDX_UNARY_MATH_TABLE(DDX_RT_MAP_MATH)
#undef DDX_RT_MAP_MATH

} // namespace detail

// Sum and Multiply are n-ary in the type but binary in the graph, so a wide
// node folds left into a chain.
template <impl::Numeric T, impl::CExpression E>
[[nodiscard]] constexpr RTExpression<T> to_graph(Builder<T> &b, const E &e) {
  using U = std::remove_cvref_t<E>;
  if constexpr (impl::CVariable<U>) {
    return var(b, U::label.view());
  } else if constexpr (impl::CLit<U>) {
    return lit(b, static_cast<T>(U::value));
  } else if constexpr (impl::CConstant<U>) {
    return lit(b,
               static_cast<double>(e.template eval_seeded<impl::mp::mp_list<>>(
                   std::array<double, 1>{})));
  } else {
    constexpr OpCode code = detail::op_code_of<typename U::op_type>::value;
    return std::apply(
        [&](const auto &...kids) {
          if constexpr (sizeof...(kids) == 1) {
            return make(code, to_graph(b, kids)...);
          } else {
            RTExpression<T> acc{};
            bool first = true;
            const auto step = [&](const auto &kid) {
              const RTExpression k = to_graph(b, kid);
              acc = first ? k : make(code, acc, k);
              first = false;
            };
            (step(kids), ...);
            return acc;
          }
        },
        e.expressions());
  }
}

} // namespace ddx::rt
