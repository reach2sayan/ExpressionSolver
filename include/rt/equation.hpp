#pragma once

#include "ops/numeric.hpp" // compile_time_factorial
// The univariate sweep runs on the truncated-polynomial scalar forward mode
// supplies, and goes with it under DDX_BUILD_DUAL=OFF.
#if DDX_HAS_DUAL
#include "dual/taylor_dual.hpp"
#endif
#include "md/md.hpp"
#include "rt/coupling.hpp"
#include "rt/derivative.hpp"
#include "rt/expressions.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "symbolic/equation.hpp"

// Optional: without it the batch calls run the interpreter, under the same
// signatures.
#ifdef DDX_HAS_JIT
#include "jit/kernel.hpp"
#endif

#include <algorithm>
#include <concepts>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <vector>

// Keyed on the RTExpression<T> *pattern* rather than a constraint: a
// `requires` over <TFirst, TRest...> would be ambiguous with the compile-time
// specialisation, since the two concepts do not subsume one another.
namespace ddx::impl {

namespace rt_detail {

template <typename A, typename T>
concept CPointArg =
    Numeric<std::remove_cvref_t<A>> || CNamedValue<A> ||
    (std::ranges::input_range<A> &&
     Numeric<std::remove_cvref_t<std::ranges::range_value_t<A>>>);

// CEvalArg admits any input_range, so without this a span of columns would try
// to become a scalar.
template <typename R, typename Ptr>
concept CColumns = std::ranges::contiguous_range<R> &&
                   std::convertible_to<std::ranges::range_value_t<R>, Ptr>;

} // namespace rt_detail

template <Numeric T, typename... Rest>
  requires(std::same_as<Rest, rt::RTExpression<T>> && ...)
class Equation<rt::RTExpression<T>, Rest...> {
public:
  using value_type = T;
  static constexpr std::size_t output_dim = 1 + sizeof...(Rest);

  // A constructor cannot answer with an error, and an expression can name no
  // graph two ways -- a bare literal, or a symbol named with no arena current.
  // The constructors below take the graph as a precondition.
  [[nodiscard]] static constexpr result<Equation>
  create(rt::RTExpression<T> first, Rest... rest) {
    if (const auto bad = why_not(first, rest...)) {
      return std::unexpected{*bad};
    }
    return Equation{first, rest...};
  }

  // Owning: the Builder is heap-allocated, so the nodes survive the move.
  [[nodiscard]] static result<Equation>
  create(std::unique_ptr<rt::Builder<T>> owned, rt::RTExpression<T> first,
         Rest... rest) {
    if (const auto bad = why_not(first, rest...)) {
      return std::unexpected{*bad};
    }
    return Equation{std::move(owned), first, rest...};
  }

  [[nodiscard]] constexpr std::size_t arity() const {
    return arena_->symbols().size();
  }
  [[nodiscard]] constexpr std::span<const std::string> symbols() const {
    return arena_->symbols();
  }

  // Every spelling make_point accepts.  The symbol list exists only at run
  // time, so every one of them, positional included, answers with result<T>.
  template <rt_detail::CPointArg<T>... Args>
  [[nodiscard]] constexpr result<std::vector<T>>
  point(const Args &...args) const {
    std::vector<T> at(arity(), T{});
    result<void> ok{};
    if constexpr ((CNamedValue<Args> && ...) && sizeof...(Args) > 0) {
      ((ok = ok ? assign_named(at, args) : ok), ...);
    } else if constexpr (sizeof...(Args) == 1 &&
                         (std::ranges::input_range<Args> && ...)) {
      ok = assign_range(at, args...);
    } else {
      const std::array<T, sizeof...(Args)> positional{static_cast<T>(args)...};
      ok = assign_range(at, positional);
    }
    if (!ok) {
      return std::unexpected{ok.error()};
    }
    return at;
  }

  [[nodiscard]] constexpr auto
  evaluate(const rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      const auto values = rt::evaluate_all(*arena_, at);
      if constexpr (output_dim == 1) {
        return values[roots_[0]];
      } else {
        return roots_ |
               std::views::transform([&](rt::NodeId r) { return values[r]; }) |
               std::ranges::to<std::vector<T>>();
      }
    });
  }

  [[nodiscard]] constexpr result<std::vector<T>>
  gradient(const rt_detail::CPointArg<T> auto &...args) const
    requires(output_dim == 1)
  {
    return point(args...).transform(
        [this](const auto &at) { return harvest(derivative_.partial, at); });
  }

  // Row-major m x n, as Equation::jacobian's extents<output_dim, input_dim>.
  [[nodiscard]] constexpr result<std::vector<T>>
  jacobian(const rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform(
        [this](const auto &at) { return harvest(derivative_.partial, at); });
  }

  // Dense row-major m x n x n, as Equation::hessian returns; the graph holds
  // it compressed by colour and the scatter hides that from callers.
  [[nodiscard]] result<std::vector<T>>
  hessian(const rt_detail::CPointArg<T> auto &...args) const {
    return point(args...).transform([this](const auto &at) {
      const auto &blocks = this->cached_hessians();
      const auto values = rt::evaluate_all(*arena_, at);
      const std::size_t n = arity();

      std::vector<T> out(output_dim * n * n);
      const impl::md::mdspan dense{
          out.data(), impl::md::dextents<std::size_t, 3>{output_dim, n, n}};
      for (const auto [k, block] : std::views::enumerate(blocks)) {
        for (const auto [i, j] : std::views::cartesian_product(
                 std::views::iota(0uz, n), std::views::iota(0uz, n))) {
          dense[static_cast<std::size_t>(k), i, j] = values[block.at(i, j)];
        }
      }
      return out;
    });
  }

  // One sweep per colour, not per symbol.  A mixing rule couples everything
  // and colours in n, so it is not always a win.
  [[nodiscard]] std::size_t hessian_colors() const
    requires(output_dim == 1)
  {
    return this->cached_hessians().front().colors();
  }

#if DDX_HAS_DUAL
  // One Taylor sweep rather than K nested duals: seed c[0] = x0, c[1] = 1,
  // then un-normalise c[Order], since TaylorDual stores f^(k)/k!.
  template <std::size_t Order>
  [[nodiscard]] result<T> univariate_derivative(T x0) const
    requires(output_dim == 1 && Order > 0)
  {
    if (arity() != 1) {
      return fail(errc::not_univariate);
    }
    using Taylor = impl::TaylorDual<T, Order>;
    Taylor seed;
    seed.c[0] = x0;
    seed.c[1] = T{1};

    const std::array<Taylor, 1> at{seed};
    const auto values = rt::evaluate_all(*arena_, at);
    return values[roots_[0]].c[Order] *
           static_cast<T>(impl::detail::compile_time_factorial(Order));
  }
#endif // DDX_HAS_DUAL

  // --- batch ---------------------------------------------------------------
  //
  // `xs[j]` is the column for symbol j; each output span holds one pointer per
  // output column, every column n long.  A column-count mismatch that went
  // unchecked would be silent memory corruption.

  [[nodiscard]] result<void>
  gradient(const rt_detail::CColumns<const T *> auto &xs,
           const rt_detail::CColumns<T *> auto &f,
           const rt_detail::CColumns<T *> auto &g, std::size_t n) const
    requires(output_dim == 1)
  {
    return dispatch(compiled(false), as_columns(xs), as_columns(f),
                    as_columns(g), {}, n);
  }

  [[nodiscard]] result<void>
  jacobian(const rt_detail::CColumns<const T *> auto &xs,
           const rt_detail::CColumns<T *> auto &f,
           const rt_detail::CColumns<T *> auto &g, std::size_t n) const {
    return dispatch(compiled(false), as_columns(xs), as_columns(f),
                    as_columns(g), {}, n);
  }

  [[nodiscard]] result<void>
  hessian(const rt_detail::CColumns<const T *> auto &xs,
          const rt_detail::CColumns<T *> auto &f,
          const rt_detail::CColumns<T *> auto &g,
          const rt_detail::CColumns<T *> auto &h, std::size_t n) const
    requires(output_dim == 1)
  {
    return dispatch(compiled(true), as_columns(xs), as_columns(f),
                    as_columns(g), as_columns(h), n);
  }

  // What a caller sizes its buffers by.
  [[nodiscard]] std::size_t value_columns() const {
    return compiled(false).graph.layout().values;
  }
  [[nodiscard]] std::size_t jacobian_columns() const {
    return compiled(false).graph.layout().jacobian;
  }
  [[nodiscard]] std::size_t hessian_columns() const
    requires(output_dim == 1)
  {
    return compiled(true).graph.layout().hessian;
  }

private:
  // Both take a graph as a precondition; create() above is what establishes it.
  constexpr explicit Equation(rt::RTExpression<T> first, Rest... rest)
      : arena_(first.builder(), borrow) {
    roots_.reserve(output_dim);
    roots_.push_back(first.id(*arena_));
    (roots_.push_back(rest.id(*arena_)), ...);
    derivative_ = rt::jacobian(*arena_, roots_);
  }

  Equation(std::unique_ptr<rt::Builder<T>> owned, rt::RTExpression<T> first,
           Rest... rest)
      : Equation(first, rest...) {
    arena_ = ArenaPtr{owned.release(), reclaim};
  }

  // The graph an expression names, or why it names none.  Poison first: a
  // symbol named with no arena current is a different mistake from a literal
  // that never reached a graph.
  [[nodiscard]] static constexpr std::optional<error>
  why_not(const rt::RTExpression<T> &first, const Rest &...rest) noexcept {
    if (first.poisoned() || (rest.poisoned() || ...)) {
      return error{errc::no_arena};
    }
    if (first.builder() == nullptr) {
      return error{errc::no_graph};
    }
    return std::nullopt;
  }

  template <CNamedValue V>
  [[nodiscard]] constexpr result<void> assign_named(std::vector<T> &at,
                                                    const V &nv) const {
    const auto name = std::remove_cvref_t<V>::symbol.view();
    const auto it = std::ranges::find(symbols(), name);
    if (it == symbols().end()) {
      return fail(errc::unknown_symbol);
    }
    at[static_cast<std::size_t>(it - symbols().begin())] =
        static_cast<T>(nv.value);
    return {};
  }

  [[nodiscard]] constexpr result<void>
  assign_range(std::vector<T> &at,
               const std::ranges::input_range auto &r) const {
    if (std::ranges::size(r) != at.size()) {
      return fail(errc::wrong_arity);
    }
    std::ranges::transform(r, at.begin(),
                           [](const auto &v) { return static_cast<T>(v); });
    return {};
  }

  const std::vector<rt::Hessian> &cached_hessians() const {
    if (hessians_.empty()) {
      hessians_ = roots_ | std::views::transform([&](rt::NodeId r) {
                    return rt::hessian(*arena_, r);
                  }) |
                  std::ranges::to<std::vector<rt::Hessian>>();
    }
    return hessians_;
  }

  [[nodiscard]] constexpr std::vector<T>
  harvest(const std::vector<rt::NodeId> &nodes,
          const std::vector<T> &at) const {
    const auto values = rt::evaluate_all(*arena_, at);
    return nodes |
           std::views::transform([&](rt::NodeId n) { return values[n]; }) |
           std::ranges::to<std::vector<T>>();
  }

  struct Compiled {
    rt::Graph<T> graph;
    // Not readable back off the graph: a zero-arity equation compresses to no
    // Hessian columns and would re-freeze on every call.
    bool has_hessian = false;
#ifdef DDX_HAS_JIT
    // Empty until freeze(), and for good where T is not the JIT's scalar.
    jit::Kernel kernel{};
#endif
  };

  // The with-Hessian graph is a superset of the without, so one cache serves
  // both.
  const Compiled &compiled(bool want_hessian) const {
    if (!compiled_ || (want_hessian && !compiled_->has_hessian)) {
      compiled_ = freeze(want_hessian);
    }
    return *compiled_;
  }

  Compiled freeze(bool want_hessian) const {
    rt::GraphBuilder<T> gb{*arena_};
    gb.values_from(roots_);
    gb.jacobian_from(derivative_.partial);
    if (want_hessian) {
      gb.hessian();
    }
    Compiled out{.graph = gb.build(), .has_hessian = want_hessian};
#ifdef DDX_HAS_JIT
    if constexpr (std::same_as<T, double>) {
      // Not an error a caller has to handle: run() falls back to interpret(),
      // which is the reference the JIT is tested against anyway.
      if (auto *const c = compiler()) {
        if (auto k = c->compile(out.graph)) {
          out.kernel = std::move(*k);
        }
      }
    }
#endif
    return out;
  }

#ifdef DDX_HAS_JIT
  // A Kernel does not own its code, so the compiler outlives every kernel it
  // produced.  Null on a host with no JIT, asked once.
  static jit::Compiler *compiler() {
    static jit::result<jit::Compiler> instance = jit::Compiler::create();
    return instance ? &*instance : nullptr;
  }
#endif

  void interpret(const Compiled &c, std::span<const T *const> xs,
                 std::span<T *const> f, std::span<T *const> g,
                 std::span<T *const> h, std::size_t n) const {
    const auto blocks = c.graph.output_blocks();
    std::vector<T> at(arity());

    for (const std::size_t i : std::views::iota(0uz, n)) {
      for (const auto [j, column] : std::views::enumerate(xs)) {
        at[static_cast<std::size_t>(j)] = column[i];
      }
      const auto values = rt::evaluate_all(*arena_, at);

      const auto fill = [&](std::span<T *const> columns,
                            std::span<const rt::NodeId> block) {
        for (const auto [column, o] : std::views::zip(columns, block)) {
          column[i] = values[o];
        }
      };
      fill(f, blocks.values);
      fill(g, blocks.jacobian);
      fill(h, blocks.hessian);
    }
  }

  template <std::ranges::contiguous_range R>
  static auto as_columns(const R &r) {
    using Ptr = std::ranges::range_value_t<R>;
    return std::span<Ptr const>{std::ranges::data(r), std::ranges::size(r)};
  }

  [[nodiscard]] result<void>
  dispatch(const Compiled &c, std::span<const T *const> xs,
           std::span<T *const> f, std::span<T *const> g, std::span<T *const> h,
           std::size_t n) const {
    const auto blocks = c.graph.output_blocks();
    if (xs.size() != arity() || f.size() != blocks.values.size() ||
        g.size() != blocks.jacobian.size() ||
        h.size() != blocks.hessian.size()) {
      return fail(errc::wrong_column_count);
    }
    run(c, xs, f, g, h, n);
    return {};
  }

  void run(const Compiled &c, std::span<const T *const> xs,
           std::span<T *const> f, std::span<T *const> g, std::span<T *const> h,
           std::size_t n) const {
#ifdef DDX_HAS_JIT
    if constexpr (std::same_as<T, double>) {
      if (c.kernel) {
        c.kernel(xs, f, g, h, n);
        return;
      }
    }
#endif
    interpret(c, xs, f, g, h, n);
  }

  // The deleter is the only difference between an arena the caller keeps and
  // one this Equation was handed.
  using ArenaPtr = std::unique_ptr<rt::Builder<T>, void (*)(rt::Builder<T> *)>;
  static constexpr void borrow(rt::Builder<T> *) noexcept {}
  static constexpr void reclaim(rt::Builder<T> *b) noexcept { delete b; }

  ArenaPtr arena_{nullptr, borrow};
  std::vector<rt::NodeId> roots_;

  // Eager: one reverse sweep costs microseconds, and building it here keeps
  // every per-point accessor const and constexpr.
  rt::Jacobian derivative_;
  // Lazy, being ~1.2M nodes and 350 ms for a 50-species mixture.
  mutable std::vector<rt::Hessian> hessians_;
  mutable std::optional<Compiled> compiled_;
};

} // namespace ddx::impl

namespace ddx::rt {

// Over expressions already built in a caller's own arena.  Partial
// specialisations contribute no deduction guides and create() is a static
// member, so this is the whole of the CTAD surface.
template <impl::Numeric T, typename... Ts>
  requires(std::same_as<Ts, RTExpression<T>> && ...)
[[nodiscard]] constexpr auto equation(RTExpression<T> first, Ts... rest) {
  return impl::Equation<RTExpression<T>, Ts...>::create(first, rest...);
}

// Build an Equation without naming an arena:
//
//   const auto eq = ddx::rt::equation([] {
//     const auto x = var("x");
//     const auto y = var("y");
//     return exp(x) * sin(y);
//   });
//
// The arena is moved into the Equation, so it can be returned and stored where
// one over a caller's Builder cannot.  The callback takes nothing: equation()
// makes an arena current for its duration.  A range of expressions is a system.
template <impl::Numeric T = double, std::invocable Assemble>
[[nodiscard]] auto equation(Assemble &&assemble) {
  auto arena = std::make_unique<Builder<T>>();
  auto built = [&] {
    const auto scope = detail::scoped_arena(*arena);
    return std::forward<Assemble>(assemble)();
  }();

  using Built = std::remove_cvref_t<decltype(built)>;
  if constexpr (std::same_as<Built, RTExpression<T>>) {
    return impl::Equation<RTExpression<T>>::create(std::move(arena), built);
  } else {
    // output_dim lives in the type, so the size must too -- array, not vector.
    constexpr std::size_t outputs = std::tuple_size_v<Built>;
    static_assert(outputs > 0,
                  "equation: a system needs at least one function");
    return impl::index_apply<outputs - 1>([&]<std::size_t... Rest>() {
      return impl::Equation<RTExpression<T>, detail::Repeat<Rest, T>...>::
          create(std::move(arena), built[0], built[Rest + 1]...);
    });
  }
}

} // namespace ddx::rt
