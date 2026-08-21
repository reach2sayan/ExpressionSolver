#pragma once

#include "rt/apply.hpp"
#include "rt/builder.hpp"

#include <ranges>
#include <vector>

namespace ddx::rt {

// Evaluate every node once, in id order.  Ids are assigned as subexpressions
// are formed, so a child always precedes its parent and one forward pass
// suffices; shared subexpressions are computed once because they are one node.
// This is the reference the JIT is checked against.
//
// The point's element type chooses the arithmetic, exactly as eval_seeded does
// on the compile-time side: hand it doubles and it computes values, hand it
// Dual<double> and the same walk carries derivatives.  The graph's own scalar
// only has to convert into it.
//
// The loop is indexed rather than a transform because it is not one: v[i] reads
// entries the same pass wrote, so it is a fold over a dependence order.
template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto evaluate_all(const Builder<T> &b, const R &point) {
  using U = std::ranges::range_value_t<R>;
  const auto at = std::ranges::begin(point);

  std::vector<U> v(b.size());
  for (const auto [i, n] : std::views::enumerate(b.nodes())) {
    switch (arity_of(n.op)) {
    case 0:
      v[i] = n.op == OpCode::Const ? static_cast<U>(n.value) : at[n.slot];
      break;
    case 1:
      v[i] = apply(n.op, v[n.a]);
      break;
    default:
      v[i] = apply(n.op, v[n.a], v[n.b]);
      break;
    }
  }
  return v;
}

template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto evaluate(const Builder<T> &b, NodeId root,
                                      const R &point) {
  return evaluate_all(b, point)[root];
}

} // namespace ddx::rt
