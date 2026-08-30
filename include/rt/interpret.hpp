#pragma once

#include "rt/apply.hpp"
#include "rt/builder.hpp"
#include "util/config.hpp"

#include <array>
#include <cmath> // std::fma
#include <concepts>
#include <functional>
#include <ranges>
#include <span>
#include <type_traits> // std::bool_constant
#include <utility>     // std::unreachable
#include <vector>

namespace ddx::rt {

// Points per block sweep: two AVX2 registers of doubles.  Also where a batch
// stops being a batch for the kernel, through jit::for_batch.
inline constexpr std::size_t block_lanes = 8;

// --- block sweep -----------------------------------------------------------
// W points per node instead of one, so the switch is paid once per node per
// block and each operation becomes a counted lane loop the vectoriser can take
// whole -- including the libm call, which is most of a Jacobian.  Raw counted
// loops over a constant bound, as in vector_dual.hpp: no peel prologue.
//
// The cases are generated from the same tables as apply(), so the operation set
// cannot drift between the scalar sweep and this one.
namespace detail {

// One counted loop for every arity: the operand columns are the pack, and the
// body is apply()'s per element.  `supported` is what keeps the loop compiling
// at a scalar the op has no meaning for.
template <std::size_t W, typename Fn, impl::Numeric T, bool Ok,
          std::same_as<T>... Ts>
constexpr void lanes(T *DDX_RESTRICT out,
                     const Ts *DDX_RESTRICT... in) noexcept {
  for (std::size_t k = 0; k < W; ++k) {
    out[k] = supported<Fn, T, Ok>(in[k]...);
  }
}

// Every op's lane loop, off apply.hpp's one dispatch: the row says how many of
// the three columns it reads, and only those are resolved -- an operand a row
// does not have is no_node, which names no lane.  Builder forms no op outside
// the tables, so a leaf row is unreachable here.
template <std::size_t W, impl::Numeric T, typename U>
DDX_ALWAYS_INLINE constexpr void
lanes_apply(const Node<T> &n, U *DDX_RESTRICT out, auto &&lane) noexcept {
  dispatch<U>(n.op, [&]<typename R>(R) {
    using Fn = typename R::functor;
    if constexpr (R::arity == 1) {
      lanes<W, Fn, U, R::ok>(out, lane(n.a));
    } else if constexpr (R::arity == 2) {
      lanes<W, Fn, U, R::ok>(out, lane(n.a), lane(n.b));
    } else if constexpr (R::arity == 3) {
      lanes<W, Fn, U, R::ok>(out, lane(n.a), lane(n.b), lane(n.c));
    } else {
      std::unreachable();
    }
  });
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

namespace detail {

// The tape row an id names, W points wide.
template <std::size_t W, impl::Numeric U>
[[nodiscard]] constexpr auto lanes_of(std::span<U> tape) {
  return [tape](NodeId v) { return tape.data() + std::size_t{v} * W; };
}

// The only place that turns a Step into numbers.
template <std::size_t W, impl::Numeric T>
DDX_ALWAYS_INLINE constexpr void sweep_step(const Builder<T> &b,
                                            std::size_t symbols, const Step &s,
                                            auto at, auto &&lane) {
  const auto &[i, fma] = s;
  const Node<T> &n = b[i];
  auto *const out = lane(i);
  using U = std::remove_cvref_t<decltype(*out)>;
  if (fma) {
    detail::lanes_fma<W>(fma.negated, lane(fma.x), lane(fma.y), lane(fma.z),
                         out);
    return;
  }
  if (n.op == OpCode::Const) {
    for (std::size_t k = 0; k < W; ++k) {
      out[k] = static_cast<U>(n.value);
    }
  } else if (is_leaf(n.op)) {
    const auto src = at + static_cast<std::ptrdiff_t>(
                              input_column(symbols, n.op, n.slot) * W);
    for (std::size_t k = 0; k < W; ++k) {
      out[k] = src[static_cast<std::ptrdiff_t>(k)];
    }
  } else {
    detail::lanes_apply<W>(n, out, lane);
  }
}

} // namespace detail

// The nodes of `schedule`, for W points at once.  `point_lanes` is
// symbol-major (symbol s, lane k at s * W + k) and `tape` node-major, so every
// lane loop is contiguous in both.  Same constraints on the schedule as
// evaluate_into; a batch's tail repeats a point, and those lanes are not read.
template <std::size_t W, impl::Numeric T, std::ranges::random_access_range R,
          impl::Numeric U>
  requires impl::Numeric<std::ranges::range_value_t<R>>
constexpr void evaluate_block(const Builder<T> &b, const R &point_lanes,
                              std::span<const Step> schedule,
                              std::span<U> tape) {
  const auto lane = detail::lanes_of<W>(tape);
  const std::size_t symbols = b.symbols().size();
  for (const Step &s : schedule) {
    detail::sweep_step<W>(b, symbols, s, std::ranges::begin(point_lanes), lane);
  }
}

// Whether every op the roots reach computes at U.  What a sweep at a scalar
// other than the graph's asks first: apply() answers U{} for an op the scalar
// lacks, and a zero node is a wrong number with nothing to say so.
template <impl::Numeric U, impl::Numeric T>
[[nodiscard]] constexpr bool computes_at(const Builder<T> &b,
                                         std::span<const NodeId> roots) {
  const auto live = detail::reachable(b, roots);
  return std::ranges::all_of(
      std::views::iota(NodeId{0}, static_cast<NodeId>(b.size())),
      [&](NodeId v) { return !live[v] || supports<U>(b[v].op); });
}

template <impl::Numeric T, std::ranges::random_access_range R>
  requires impl::Numeric<std::ranges::range_value_t<R>>
[[nodiscard]] constexpr auto evaluate(const Builder<T> &b, NodeId root,
                                      const R &point) {
  return evaluate_all(b, point)[root];
}

// The nodes of `schedule`, one point at a time, into caller-owned scratch
// indexed by node id.  The schedule must be topological and closed under
// operands; detail::schedule_of over id order and Graph::schedule() are both.
// Entries outside it are left alone.  The width-one block sweep, not a second
// implementation of it.
template <impl::Numeric T, std::ranges::random_access_range R, impl::Numeric U>
  requires impl::Numeric<std::ranges::range_value_t<R>>
constexpr void evaluate_into(const Builder<T> &b, const R &point,
                             std::span<const Step> schedule, std::span<U> v) {
  evaluate_block<1>(b, point, schedule, v);
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
  const auto live = detail::reachable(b, roots);
  std::vector<U> v(b.size());
  evaluate_into(b, point, detail::schedule_of(b, detail::live_ids(live)),
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
  evaluate_into(
      b, point,
      detail::schedule_of(
          b, std::views::iota(NodeId{0}, static_cast<NodeId>(b.size()))),
      std::span<U>{v});
  return v;
}

} // namespace ddx::rt
