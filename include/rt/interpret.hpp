#pragma once

#include "rt/apply.hpp"
#include "rt/builder.hpp"
#include "util/config.hpp"

#include <cmath> // std::fma
#include <concepts>
#include <functional>
#include <ranges>
#include <span>
#include <type_traits> // std::bool_constant
#include <utility>     // std::unreachable
#include <vector>

namespace ddx::rt {

// --- block sweep -----------------------------------------------------------
// W points per node instead of one, so the switch is paid once per node per
// block and each operation becomes a counted lane loop the vectoriser can take
// whole -- including the libm call, which is most of a Jacobian.  Raw counted
// loops over a constant bound, as in vector_dual.hpp: no peel prologue.
//
// The cases are generated from the same tables as apply(), so the operation set
// cannot drift between the scalar sweep and this one.
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
    std::unreachable(); // every unary row is above; Builder forms no other
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
    std::unreachable();
  }
}

// x * y + z rounded once, which is what the kernel's llvm.fma lowers to.  A
// scalar with no fma of its own rounds twice and has no kernel to disagree
// with.
template <impl::Numeric T>
[[nodiscard]] constexpr T fused_multiply_add(const T &x, const T &y,
                                             const T &z) noexcept {
  if constexpr (std::floating_point<T>) {
    // <cmath> is constexpr only in C++26 and folding a libm call before that is
    // a GCC extension, so clang takes the arithmetic unfused here.  The probe
    // asks the compiler rather than the version macros.
    if consteval {
      if constexpr (!requires {
                      std::bool_constant<(std::fma(T{2}, T{3}, T{4}) ==
                                          T{10})>{};
                    }) {
        return T{x * y + z};
      }
    }
    return std::fma(x, y, z);
  } else {
    return T{x * y + z};
  }
}

template <std::size_t W, impl::Numeric T>
constexpr void lanes_fma(bool negated, const T *DDX_RESTRICT x,
                         const T *DDX_RESTRICT y, const T *DDX_RESTRICT z,
                         T *DDX_RESTRICT out) noexcept {
  // The sign is hoisted out of the lane loop rather than tested in it.
  const auto sweep = [&](auto sign) {
    for (std::size_t k = 0; k < W; ++k) {
      out[k] = fused_multiply_add<T>(sign(x[k]), y[k], z[k]);
    }
  };
  if (negated) {
    sweep(std::negate<>{});
  } else {
    sweep(std::identity{});
  }
}

} // namespace detail

// The nodes named by `order`, for W points at once.  `point_lanes` is
// symbol-major (symbol s, lane k at s * W + k) and `tape` node-major, so every
// lane loop is contiguous in both.  Same constraints on `order` as
// evaluate_into; a batch's tail repeats a point, and those lanes are not read.
//
// `contract` takes a multiply feeding an add as one rounding -- the arithmetic
// the kernel emits, and so the default.  Graph::contracted_order() is the order
// that goes with it; live_order() is also correct and computes one or two
// multiplies for nobody.
template <std::size_t W, impl::Numeric T, std::ranges::random_access_range R,
          std::ranges::input_range Order, impl::Numeric U>
  requires impl::Numeric<std::ranges::range_value_t<R>> &&
           std::convertible_to<std::ranges::range_value_t<Order>, NodeId>
constexpr void evaluate_block(const Builder<T> &b, const R &point_lanes,
                              Order order, std::span<U> tape,
                              bool contract = true) {
  const auto at = std::ranges::begin(point_lanes);
  const auto lane = [&tape](NodeId v) {
    return tape.data() + std::size_t{v} * W;
  };
  for (const NodeId i : order) {
    const Node<T> &n = b[i];
    U *const out = lane(i);
    if (contract) {
      if (const Contraction c = contraction_at(b, i)) {
        detail::lanes_fma<W>(c.negated, lane(c.x), lane(c.y), lane(c.z), out);
        continue;
      }
    }
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
      detail::lanes_unary<W>(n.op, lane(n.a), out);
      break;
    default:
      detail::lanes_binary<W>(n.op, lane(n.a), lane(n.b), out);
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
// indexed by node id.  `order` must be topological and closed under operands;
// plain id order and Graph::live_order() are both.  Entries outside it are left
// alone.  The width-one block sweep, not a second implementation of it.
template <impl::Numeric T, std::ranges::random_access_range R,
          std::ranges::input_range Order, impl::Numeric U>
  requires impl::Numeric<std::ranges::range_value_t<R>> &&
           std::convertible_to<std::ranges::range_value_t<Order>, NodeId>
constexpr void evaluate_into(const Builder<T> &b, const R &point, Order order,
                             std::span<U> v, bool contract = true) {
  evaluate_block<1>(b, point, order, v, contract);
}

// evaluate_all narrowed to what `roots` reach.  Still id-indexed and
// arena-sized, with the entries the roots miss left value-initialised and not
// to be read.  Worth the extra pass: a derivative sweep leaves a whole Hessian
// in the arena that one root's walk has no use for.
template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto evaluate_reachable(const Builder<T> &b,
                                                std::span<const NodeId> roots,
                                                const R &point) {
  using U = std::ranges::range_value_t<R>;
  const auto live =
      detail::reachable(b.size(), roots, [&b](NodeId v, auto &&mark) {
        for (const NodeId u : b.operands(v)) {
          if (u != no_node) {
            mark(u);
          }
        }
      });
  std::vector<U> v(b.size());
  evaluate_into(b, point,
                std::views::iota(NodeId{0}, static_cast<NodeId>(b.size())) |
                    std::views::filter([&live](NodeId i) { return live[i]; }),
                std::span<U>{v});
  return v;
}

// Every node once, in id order: a child always precedes its parent, and the
// reference the JIT is checked against.  The point's element type chooses the
// arithmetic, so Dual<double> carries derivatives through the same walk.
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
