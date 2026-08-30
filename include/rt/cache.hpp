#pragma once

#include "rt/cone.hpp"
#include "rt/graph.hpp"
#include "rt/interpret.hpp"
#include "util/config.hpp" // DDX_FWD

#include <boost/container/vector.hpp> // a vector<bool> that holds bools
#include <boost/dynamic_bitset.hpp>

#include <algorithm>
#include <concepts>
#include <cstring> // std::memcmp, the only key comparison there is
#include <functional>
#include <memory>
#include <mutex> // unique_lock and try_to_lock, which <shared_mutex> lacks
#include <ranges>
#include <shared_mutex>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// A call the equation need not make again.  Nothing here is reached unless an
// equation was given a cache, and no answer differs because one was: a served
// call gives back the bits the sweep gave, and an amended one re-evaluates
// exactly the nodes the point moved through, so what it does not touch keeps
// the bits a cold call would have left there.
namespace ddx::rt {

// Whether a lane's answer comes from a sweep, which leaves a tape behind to
// amend, or from a kernel, which computes its whole graph and leaves nothing.
enum class Answered : std::uint8_t { BySweep, ByKernel };

// One remembered call.  `point` is the input columns as they were passed --
// symbols then seeds, input_column()'s order.  `values` is every output column
// end to end, in the order the ABI writes them.  `tape` is node-indexed, and
// empty on a lane a kernel answers.
template <typename T> struct Recorded {
  std::span<T> point;
  std::span<T> values;
  std::span<T> tape;
};

// What a slot has to hold, and what it holds it for: two calls match only where
// their whole Extent does, so a lane frozen again, an arena that grew, or a
// kernel arriving over a sweep all part an entry from the call that reads it.
// Only extent_of() makes one, so no site can size a slot its own way.
class Extent {
public:
  [[nodiscard]] friend constexpr bool operator==(const Extent &,
                                                 const Extent &) = default;

  [[nodiscard]] constexpr std::size_t point() const noexcept { return point_; }
  [[nodiscard]] constexpr std::size_t values() const noexcept {
    return values_;
  }
  [[nodiscard]] constexpr std::size_t tape() const noexcept { return tape_; }

  // The only way to make one, so no site can size a slot its own way.  `nodes`
  // is the arena's size and not the graph's: a borrowed arena may have grown
  // since the freeze, and live ids index below that.
  template <impl::Numeric T>
  [[nodiscard]] static Extent of(const Graph<T> &graph, Answered how,
                                 std::size_t nodes) {
    const Layout &layout = graph.layout();
    return Extent{graph.arity(),
                  layout.values + layout.jacobian + layout.hessian,
                  how == Answered::ByKernel ? 0uz : nodes, &graph};
  }

private:
  constexpr Extent(std::size_t point, std::size_t values, std::size_t tape,
                   const void *of) noexcept
      : point_(point), values_(values), tape_(tape), of_(of) {}

  std::size_t point_ = 0;
  std::size_t values_ = 0;
  std::size_t tape_ = 0;
  const void *of_ = nullptr; // which frozen graph, by identity

public:
  constexpr Extent() noexcept = default;
};

// A lease on one lane's slot, held for as long as it is read or written.
// Disengaged where the cache declines -- and, on the write side, where another
// thread holds it, which is a call that runs uncached rather than one that
// waits.
template <typename L, typename T>
concept CReadLease = requires(const L &l) {
  { static_cast<bool>(l) } -> std::same_as<bool>;
  { l.filled() } -> std::same_as<bool>; // holds a previous call
  { l.entry() } -> std::same_as<Recorded<const T>>;
};

template <typename L, typename T>
concept CWriteLease = requires(L &l, const L &cl) {
  { static_cast<bool>(cl) } -> std::same_as<bool>;
  { cl.filled() } -> std::same_as<bool>;
  { l.entry() } -> std::same_as<Recorded<T>>;
  { l.commit() } -> std::same_as<void>; // what was written is the entry now
};

// Somewhere to put one remembered call per Want, and nothing else: the equation
// decides what is stale, a cache only decides what it keeps.  Both spellings
// are const because every accessor on an Equation is, and const calls on one
// are safe to make concurrently -- so a cache carries its own lock.
//
// Copied in by value and never referred to afterwards, so a caller's object may
// die the moment the equation is built.
template <typename C, typename T>
concept CValueCache =
    std::copy_constructible<C> && std::movable<C> &&
    requires(const C &c, Want w, Extent e) {
      { c.read(w, e) } -> CReadLease<T>;
      { c.write(w, e) } -> CWriteLease<T>;
      { c.clear() }; // a re-freeze invalidates every lane
    };

// The built-in: the last call per lane, behind one shared_mutex per lane.  A
// reader takes a shared lock, so any number of threads can be served the same
// point at once; a writer takes a unique one and does not wait for it, because
// a thread that would have waited can sweep instead and be no slower than it
// was before there was a cache.
//
// The state sits behind a pointer so an Equation holding one is still movable,
// and a copy is a copy of the configuration: entries belong to the object that
// filled them.
template <impl::Numeric T = double> class LastValue {
public:
  LastValue() : lanes_(std::make_unique<Lanes>()) {}
  explicit LastValue(bool on)
      : lanes_(on ? std::make_unique<Lanes>() : nullptr) {}

  LastValue(const LastValue &other)
      : lanes_(other.lanes_ ? std::make_unique<Lanes>() : nullptr) {}
  LastValue &operator=(const LastValue &other) {
    lanes_ = other.lanes_ ? std::make_unique<Lanes>() : nullptr;
    return *this;
  }
  LastValue(LastValue &&) noexcept = default;
  LastValue &operator=(LastValue &&) noexcept = default;
  ~LastValue() = default;

private:
  // Filled once per point and read until the next one.  `extent` is what it was
  // filled for; a call whose own differs is a call about some other graph.
  struct Slot {
    mutable std::shared_mutex mutex;
    Extent extent{};
    std::vector<T> point, values, tape;
    bool filled = false;
  };
  using Lanes = std::array<Slot, want_count>;

public:
  class ReadLease {
  public:
    ReadLease() = default;
    ReadLease(const Slot &slot, std::shared_lock<std::shared_mutex> lock)
        : slot_(&slot), lock_(std::move(lock)) {}

    [[nodiscard]] explicit operator bool() const noexcept {
      return slot_ != nullptr;
    }
    [[nodiscard]] bool filled() const noexcept { return slot_->filled; }
    [[nodiscard]] Recorded<const T> entry() const noexcept {
      return {slot_->point, slot_->values, slot_->tape};
    }

  private:
    const Slot *slot_ = nullptr;
    std::shared_lock<std::shared_mutex> lock_;
  };

  class WriteLease {
  public:
    WriteLease() = default;
    WriteLease(Slot &slot, std::unique_lock<std::shared_mutex> lock)
        : slot_(&slot), lock_(std::move(lock)) {}

    [[nodiscard]] explicit operator bool() const noexcept {
      return slot_ != nullptr;
    }
    [[nodiscard]] bool filled() const noexcept { return slot_->filled; }
    [[nodiscard]] Recorded<T> entry() noexcept {
      return {slot_->point, slot_->values, slot_->tape};
    }
    void commit() noexcept { slot_->filled = true; }

  private:
    Slot *slot_ = nullptr;
    std::unique_lock<std::shared_mutex> lock_;
  };

  // Engaged only where the slot is already the shape the caller asked for:
  // resizing is a write, and a reader holds nothing but a shared lock.
  [[nodiscard]] ReadLease read(Want want, const Extent &extent) const {
    if (!lanes_) {
      return {};
    }
    const Slot &slot = (*lanes_)[std::to_underlying(want)];
    std::shared_lock lock{slot.mutex};
    return slot.extent == extent ? ReadLease{slot, std::move(lock)}
                                 : ReadLease{};
  }

  // Never waits: a lane another thread is filling is a lane this call does
  // without.
  [[nodiscard]] WriteLease write(Want want, const Extent &extent) const {
    if (!lanes_) {
      return {};
    }
    Slot &slot = (*lanes_)[std::to_underlying(want)];
    std::unique_lock lock{slot.mutex, std::try_to_lock};
    if (!lock) {
      return {};
    }
    if (slot.extent != extent) {
      slot.extent = extent;
      slot.point.assign(extent.point(), T{});
      slot.values.assign(extent.values(), T{});
      slot.tape.assign(extent.tape(), T{});
      slot.filled = false;
    }
    return {slot, std::move(lock)};
  }

  void clear() const {
    if (!lanes_) {
      return;
    }
    for (Slot &slot : *lanes_) {
      const std::unique_lock lock{slot.mutex};
      slot.extent = {};
      slot.filled = false;
    }
  }

private:
  // mutable: the lanes are what a const call writes, which is the whole point
  // of a cache, and the locks above are what make that safe.
  mutable std::unique_ptr<Lanes> lanes_;
};

// What a caller never writes: the default cache, and the mixin that decides
// what to do with one.  The contract above is the whole of what a cache of
// their own has to model.
namespace detail {

// The default: no slot, so every call runs and nothing is remembered.  Empty,
// and Caching skips it outright rather than trusting the optimiser to.
template <impl::Numeric T> struct NoCache {
  struct Lease {
    [[nodiscard]] explicit operator bool() const noexcept { return false; }
    [[nodiscard]] bool filled() const noexcept { return false; }
    [[nodiscard]] Recorded<const T> entry() const noexcept { return {}; }
    [[nodiscard]] Recorded<T> entry() noexcept { return {}; }
    void commit() noexcept {}
  };
  [[nodiscard]] constexpr Lease read(Want, const Extent &) const noexcept {
    return {};
  }
  [[nodiscard]] constexpr Lease write(Want, const Extent &) const noexcept {
    return {};
  }
  constexpr void clear() const noexcept {}
};

// Every single-point call comes through here, in both equations, so there is
// one implementation of "has this been asked before" and one of what to do
// about it.  Derived supplies the arena the ids index and whatever it wants
// given up around a sweep -- the GIL, on the Python side.
template <typename Derived, impl::Numeric T, CValueCache<T> Cache>
class Caching {
protected:
  constexpr Caching() = default;

  // Copied in: the caller's object is not referred to again.
  constexpr void take_cache(Cache c) { cache_ = std::move(c); }
  [[nodiscard]] constexpr const Cache &cache() const noexcept { return cache_; }

  // `run` is what the call would have done with no cache at all, and is what
  // every path that cannot be served falls back to.
  void through(Want want, const Graph<T> &graph, Answered how,
               std::span<const T *const> xs, const Columns<T> &out,
               std::size_t n, auto &&run) const {
    if constexpr (std::same_as<Cache, NoCache<T>>) {
      run();
    } else {
      // A batch is amortised already, and comparing points would cost what it
      // saves.
      if (n != 1) {
        run();
      } else {
        serve(want, graph, how, xs, out, DDX_FWD(run));
      }
    }
  }

private:
  [[nodiscard]] const Derived &self() const noexcept {
    return static_cast<const Derived &>(*this);
  }

  // What an entry is worth to the point in hand.  Four outcomes, so four types:
  // nothing here is a bool a reader has to remember the meaning of, and which
  // sweep runs is chosen by overload rather than by a flag.
  struct Served {                    // asked before; the answer is to hand
    std::span<const T> values;
  };
  struct Fill {};                    // nothing to work from: every step
  struct Amend {                     // only the cone the move reached
    std::span<const bool> changed;
  };
  struct Relay {};                   // a kernel answers this lane: all or none
  using Plan = std::variant<Served, Fill, Amend, Relay>;

  void serve(Want want, const Graph<T> &graph, Answered how,
             std::span<const T *const> xs, const Columns<T> &out,
             auto &&run) const {
    const Extent extent = Extent::of(graph, how, self().arena().size());

    // The point, once, in the order a sweep reads it.
    std::vector<T> &at = point_scratch();
    at.resize(xs.size());
    std::ranges::transform(xs, at.begin(),
                           [](const T *column) { return *column; });

    // Under the shared lock there is one question, and it is the common one.
    if (const auto lease = cache().read(want, extent);
        lease && lease.filled() && unmoved(at, lease.entry().point)) {
      spread(lease.entry().values, out);
      return;
    }

    auto lease = cache().write(want, extent);
    if (!lease) { // another thread is filling this lane; do without
      run();
      return;
    }
    const Recorded<T> slot = lease.entry();
    std::visit(
        [&]<typename P>(const P &plan) {
          if constexpr (std::same_as<P, Served>) {
            spread(plan.values, out);
            return;
          } else if constexpr (std::same_as<P, Relay>) {
            run();
            collect(out, slot.values);
          } else { // Fill or Amend: whichever sweep the overload names
            sweep(want, graph, at, plan, slot);
            keep(graph.output_blocks(), slot.tape, slot.values);
            spread(slot.values, out);
          }
          remember(at, slot, lease);
        },
        decide(lease.filled(), at, slot));
  }

  [[nodiscard]] Plan decide(bool filled, const std::vector<T> &at,
                            const Recorded<T> &slot) const {
    // A tape is what an amendment amends, and a kernel leaves none.
    const bool swept = !slot.tape.empty();
    if (!filled) {
      return swept ? Plan{Fill{}} : Plan{Relay{}};
    }
    boost::container::vector<bool> &changed = column_scratch();
    changed.assign(at.size(), false);
    std::ranges::transform(
        at, slot.point, changed.begin(),
        [](const T &now, const T &was) { return !same_bits(now, was); });
    // The thread that won the upgrade may have filled the entry with this very
    // point while this one waited for it.
    if (std::ranges::none_of(changed, std::identity{})) {
      return Served{slot.values};
    }
    return swept ? Plan{Amend{{changed.data(), changed.size()}}}
                 : Plan{Relay{}};
  }

  // Every step, into a tape holding nothing worth keeping.  A first call is not
  // an amendment with everything moved: a literal is dirty for no point at all,
  // and its slot still has to be written once.
  void sweep(Want, const Graph<T> &graph, const std::vector<T> &at, Fill,
             const Recorded<T> &slot) const {
    [[maybe_unused]] const auto held = self().unlocked();
    evaluate_into(self().arena(), at, graph.schedule(), slot.tape);
  }

  // Only the steps the move reached, over the tape the last point left.
  void sweep(Want want, const Graph<T> &graph, const std::vector<T> &at,
             const Amend &plan, const Recorded<T> &slot) const {
    const Cone &cone = cone_for(want, graph);
    // Given up around the sweep and nothing else: the GIL, on the Python side.
    [[maybe_unused]] const auto held = self().unlocked();
    evaluate_reached(self().arena(), at, graph.schedule(), cone, plan.changed,
                     reach_scratch(), stack_scratch(), slot.tape);
  }

  // One cone per lane, read upward the first time an amendment wants it and
  // dropped when the lane is frozen again.  Written only while this lane's
  // write lease is held, which is what a mutable member is safe under here.
  [[nodiscard]] const Cone &cone_for(Want want, const Graph<T> &graph) const {
    Reached &kept = cones_[std::to_underlying(want)];
    if (kept.of != &graph) {
      kept = Reached{&graph,
                     Cone{self().arena(), graph.schedule(), graph.arity()}};
    }
    return kept.cone;
  }

  static void remember(const std::vector<T> &at, const Recorded<T> &slot,
                       auto &lease) {
    std::ranges::copy(at, slot.point.begin());
    lease.commit();
  }

  // Bitwise, never ==: 0.0 and -0.0 are the same number and not the same point
  // -- a derivative at either sign of zero differs -- and a repeated NaN is a
  // hit, which == would refuse.
  [[nodiscard]] static bool same_bits(const T &a, const T &b) noexcept {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
  }

  [[nodiscard]] static bool unmoved(const std::vector<T> &at,
                                    std::span<const T> was) noexcept {
    return std::ranges::equal(at, was, same_bits);
  }

  // A remembered call is one run of values, block after block: joining the
  // three is the same ordering every other reader of a tape pairs them by, so
  // reading one back and writing one down are each a single transform.
  static void spread(std::span<const T> values, const Columns<T> &out) {
    for (const auto [column, value] :
         std::views::zip(in_order(out) | std::views::join, values)) {
      *column = value;
    }
  }

  static void collect(const Columns<T> &out, std::span<T> values) {
    std::ranges::transform(in_order(out) | std::views::join, values.begin(),
                           [](const T *column) { return *column; });
  }

  static void keep(const OutputSpans &blocks, std::span<const T> tape,
                   std::span<T> values) {
    std::ranges::transform(in_order(blocks) | std::views::join, values.begin(),
                           [tape](NodeId o) { return tape[o]; });
  }

  // thread_local rather than members: a const call may be made from two threads
  // at once, and none of this is remembered -- only reused.  Grown and never
  // shrunk, so a loop allocates once.  boost::container::vector<bool> because
  // std::vector<bool> is a bitset and cannot lend a span.
  [[nodiscard]] static std::vector<T> &point_scratch() {
    static thread_local std::vector<T> at;
    return at;
  }
  [[nodiscard]] static boost::container::vector<bool> &column_scratch() {
    static thread_local boost::container::vector<bool> changed;
    return changed;
  }
  [[nodiscard]] static boost::dynamic_bitset<> &reach_scratch() {
    static thread_local boost::dynamic_bitset<> steps;
    return steps;
  }
  [[nodiscard]] static std::vector<std::uint32_t> &stack_scratch() {
    static thread_local std::vector<std::uint32_t> stack;
    return stack;
  }

  struct Reached {
    const void *of = nullptr; // which frozen graph it was read from
    Cone cone;
  };
  // Nothing at all without a cache: a Cone holds a graph and a constant
  // evaluation can hold neither.
  using Cones =
      std::conditional_t<std::same_as<Cache, NoCache<T>>, std::tuple<>,
                         std::array<Reached, want_count>>;

  [[no_unique_address]] Cache cache_{};
  [[no_unique_address]] mutable Cones cones_{};
};

} // namespace detail

} // namespace ddx::rt
