#pragma once

#include "rt/graph.hpp"
#include "rt/interpret.hpp"

#include <boost/dynamic_bitset.hpp>
#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/range/iterator_range_core.hpp>

#include <cstdint>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace ddx::rt {

// Which steps a moved input column reaches.  A schedule is read upward once,
// here, so that a call moving one symbol walks its own cone rather than an
// order most of which it is about to skip -- which on a wide model is the
// difference between amending and simply sweeping again.
//
// Vertices are positions in the schedule, not node ids: the walk hands its
// answer straight to the loop that runs the steps, and neither has to translate
// for the other.
class Cone {
public:
  Cone() = default;

  // Every reader of a step, and every step a column is read by.  The operands
  // are the contraction's where a step has one -- an fma reads the multiply's
  // operands and not the multiply, which nothing computes.
  template <impl::Numeric T>
  Cone(const Builder<T> &b, std::span<const Step> schedule,
       std::size_t columns) {
    const std::size_t symbols = b.symbols().size();
    std::vector<std::uint32_t> at(b.size(), unscheduled);
    for (const auto [i, s] : schedule | std::views::enumerate) {
      at[s.node] = static_cast<std::uint32_t>(i);
    }

    std::vector<std::pair<std::size_t, std::size_t>> edges;
    edges.reserve(schedule.size() * 3);
    std::vector<std::vector<std::uint32_t>> reads(columns);
    for (const auto [i, s] : schedule | std::views::enumerate) {
      const auto step = static_cast<std::uint32_t>(i);
      const Node<T> &n = b[s.node];
      if (!s.fma && is_leaf(n.op)) {
        if (n.op != OpCode::Const) {
          reads[input_column(symbols, n.op, n.slot)].push_back(step);
        }
        continue;
      }
      // An absent operand is no_node, which names nothing; a contraction's
      // three are all real.
      for (const NodeId u :
           operands_of(n, s) |
               std::views::filter([](NodeId v) { return v != no_node; })) {
        if (const std::uint32_t from = at[u]; from != unscheduled) {
          edges.emplace_back(from, step);
        }
      }
    }
    readers_ = Readers{boost::edges_are_unsorted_multi_pass, edges.begin(),
                       edges.end(), schedule.size()};
    // One flat run per column, so a walk's seeds are a subrange and not a
    // vector of vectors chased per call.
    seeds_.reserve(columns + 1);
    seeds_.push_back(0);
    for (const auto &column : reads) {
      impl::append(seeded_, column);
      seeds_.push_back(static_cast<std::uint32_t>(seeded_.size()));
    }
    steps_ = schedule.size();
  }

  [[nodiscard]] std::size_t steps() const noexcept { return steps_; }

  // The steps `changed` reaches, set in `dirty`.  `stack` is the walk's, and
  // both are the caller's so that a call allocates nothing.
  //
  // A plain stack rather than one of BGL's searches: what the walk has to leave
  // behind is a bitset indexed by schedule position, which the loop below reads
  // in order and skips a word of clean steps at a time.  A colour map would be
  // a second marking of the same thing.
  void reach(std::span<const bool> changed, boost::dynamic_bitset<> &dirty,
             std::vector<std::uint32_t> &stack) const {
    dirty.resize(steps_);
    dirty.reset();
    stack.clear();
    for (const auto [column, moved] : changed | std::views::enumerate) {
      if (!moved) {
        continue;
      }
      for (const std::uint32_t step :
           seeds_of(static_cast<std::size_t>(column))) {
        if (!dirty.test_set(step)) {
          stack.push_back(step);
        }
      }
    }
    while (!stack.empty()) {
      const std::uint32_t step = stack.back();
      stack.pop_back();
      for (const auto &edge : boost::make_iterator_range(
               boost::out_edges(static_cast<Reader>(step), readers_))) {
        const auto reader =
            static_cast<std::uint32_t>(boost::target(edge, readers_));
        if (!dirty.test_set(reader)) {
          stack.push_back(reader);
        }
      }
    }
  }

  [[nodiscard]] std::span<const std::uint32_t>
  seeds_of(std::size_t column) const {
    return std::span{seeded_}.subspan(seeds_[column],
                                      seeds_[column + 1] - seeds_[column]);
  }

private:
  // Vertices are schedule positions; an edge runs from a step to a step that
  // reads it.
  using Readers = boost::compressed_sparse_row_graph<boost::directedS>;
  using Reader = boost::graph_traits<Readers>::vertex_descriptor;

  static constexpr std::uint32_t unscheduled =
      std::numeric_limits<std::uint32_t>::max();

  template <impl::Numeric T>
  [[nodiscard]] static std::array<NodeId, 3> operands_of(const Node<T> &n,
                                                         const Step &s) {
    return s.fma ? std::array{s.fma.x, s.fma.y, s.fma.z}
                 : std::array{n.a, n.b, n.c};
  }

  Readers readers_;
  std::vector<std::uint32_t> seeded_; // steps that read a column, by column
  std::vector<std::uint32_t> seeds_;  // where each column's run starts
  std::size_t steps_ = 0;
};

// The steps a moved point reaches, one point, into a tape that still holds the
// last point's values: everything the move does not reach keeps the bits it
// had, which is what makes this equal a cold sweep exactly rather than nearly.
// `changed` is one flag per input column, in input_column()'s order.
template <impl::Numeric T, std::ranges::random_access_range R, impl::Numeric U>
  requires impl::Numeric<std::ranges::range_value_t<R>>
void evaluate_reached(const Builder<T> &b, const R &point,
                      std::span<const Step> schedule, const Cone &cone,
                      std::span<const bool> changed,
                      boost::dynamic_bitset<> &dirty,
                      std::vector<std::uint32_t> &stack, std::span<U> tape) {
  cone.reach(changed, dirty, stack);
  const auto lane = detail::lanes_of<1>(tape);
  const std::size_t symbols = b.symbols().size();
  for (auto step = dirty.find_first(); step != boost::dynamic_bitset<>::npos;
       step = dirty.find_next(step)) {
    detail::sweep_step<1>(b, symbols, schedule[step], std::ranges::begin(point),
                          lane);
  }
}

} // namespace ddx::rt
