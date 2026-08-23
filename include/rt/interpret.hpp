#pragma once

#include "rt/apply.hpp"
#include "rt/builder.hpp"
#include "util/config.hpp"

#include <concepts>
#include <ranges>
#include <span>
#include <vector>

namespace ddx::rt {

// --- block sweep -----------------------------------------------------------
// W points per node instead of one, so the switch is paid once per node per
// block rather than once per node per point, and each operation becomes a
// counted lane loop the vectoriser can take whole -- including the libm call,
// which is most of a Jacobian.  The lane loops are deliberately raw counted
// loops over a constant bound, as in vector_dual.hpp: whole-register stores
// with no peel prologue.
//
// The cases are generated from the same tables as apply(), so the operation
// set cannot drift between the scalar sweep and this one.
namespace detail {

template <std::size_t W, impl::Numeric T>
constexpr void lanes_unary(OpCode op, const T *DDX_RESTRICT u,
                           T *DDX_RESTRICT out) noexcept {
  switch (op) {
#define DDX_RT_LANES(fn, Op, label, functor, ...)                              \
  case OpCode::Op:                                                             \
    for (std::size_t k = 0; k < W; ++k) {                                      \
      out[k] = supported<functor, T, probes_##Op<T>>(u[k]);                    \
    }                                                                          \
    return;
    DDX_RT_UNARY_TABLE(DDX_RT_LANES)
#undef DDX_RT_LANES
#define DDX_RT_LANES(fn, Op, label)                                            \
  case OpCode::Op:                                                             \
    for (std::size_t k = 0; k < W; ++k) {                                      \
      out[k] = supported<impl::detail::Op##Fn<T>, T, probes_##Op<T>>(u[k]);    \
    }                                                                          \
    return;
    DDX_UNARY_MATH_TABLE(DDX_RT_LANES)
#undef DDX_RT_LANES
  default:
    for (std::size_t k = 0; k < W; ++k) {
      out[k] = T{};
    }
    return;
  }
}

template <std::size_t W, impl::Numeric T>
constexpr void lanes_binary(OpCode op, const T *DDX_RESTRICT l,
                            const T *DDX_RESTRICT r,
                            T *DDX_RESTRICT out) noexcept {
  switch (op) {
#define DDX_RT_LANES(fn, Op, label, functor, ...)                              \
  case OpCode::Op:                                                             \
    for (std::size_t k = 0; k < W; ++k) {                                      \
      out[k] = supported<functor, T, probes_##Op<T>>(l[k], r[k]);              \
    }                                                                          \
    return;
    DDX_RT_BINARY_TABLE(DDX_RT_LANES)
#undef DDX_RT_LANES
  default:
    for (std::size_t k = 0; k < W; ++k) {
      out[k] = T{};
    }
    return;
  }
}

} // namespace detail

// The nodes named by `order`, for W points at once.  `point_lanes` is
// symbol-major (symbol s, lane k at s * W + k) and `tape` node-major, so every
// lane loop is contiguous in both.  Same constraints on `order` as
// evaluate_into; the tail of a batch is padded by repeating a point rather
// than falling back to a scalar path, and the padded lanes are simply not read
// back.
template <std::size_t W, impl::Numeric T, std::ranges::random_access_range R,
          std::ranges::input_range Order, impl::Numeric U>
  requires impl::Numeric<std::ranges::range_value_t<R>> &&
           std::convertible_to<std::ranges::range_value_t<Order>, NodeId>
constexpr void evaluate_block(const Builder<T> &b, const R &point_lanes,
                              Order order, std::span<U> tape) {
  const auto at = std::ranges::begin(point_lanes);
  for (const NodeId i : order) {
    const Node<T> &n = b[i];
    U *const out = tape.data() + std::size_t{i} * W;
    switch (arity_of(n.op)) {
    case 0:
      if (n.op == OpCode::Const) {
        for (std::size_t k = 0; k < W; ++k) {
          out[k] = static_cast<U>(n.value);
        }
      } else {
        const auto src =
            at + static_cast<std::ptrdiff_t>(std::size_t{n.slot} * W);
        for (std::size_t k = 0; k < W; ++k) {
          out[k] = src[static_cast<std::ptrdiff_t>(k)];
        }
      }
      break;
    case 1:
      detail::lanes_unary<W>(n.op, tape.data() + std::size_t{n.a} * W, out);
      break;
    default:
      detail::lanes_binary<W>(n.op, tape.data() + std::size_t{n.a} * W,
                              tape.data() + std::size_t{n.b} * W, out);
      break;
    }
  }
}

template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto evaluate(const Builder<T> &b, NodeId root,
                                      const R &point) {
  return evaluate_all(b, point)[root];
}

// The nodes named by `order`, one point at a time, into caller-owned scratch
// indexed by node id.  `order` must be topological and closed under operands,
// so that every entry read has already been written; plain id order and
// Graph::live_order() are both.  Entries outside `order` are left alone.
//
// The width-one block sweep, not a second implementation of it: a lane loop
// over one lane is the scalar walk, and having the two drift apart is the one
// bug neither would show on its own.
template <impl::Numeric T, std::ranges::random_access_range R,
          std::ranges::input_range Order, impl::Numeric U>
  requires impl::Numeric<std::ranges::range_value_t<R>> &&
           std::convertible_to<std::ranges::range_value_t<Order>, NodeId>
constexpr void evaluate_into(const Builder<T> &b, const R &point, Order order,
                             std::span<U> v) {
  evaluate_block<1>(b, point, order, v);
}

// Every node once, in id order: a child always precedes its parent.  The
// reference the JIT is checked against.  The point's element type chooses the
// arithmetic, as eval_seeded does, so Dual<double> carries derivatives through
// the same walk.
template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto evaluate_all(const Builder<T> &b, const R &point) {
  using U = std::ranges::range_value_t<R>;
  std::vector<U> v(b.size());
  evaluate_into(b, point,
                std::views::iota(NodeId{0}, static_cast<NodeId>(b.size())),
                std::span<U>{v});
  return v;
}

} // namespace ddx::rt
