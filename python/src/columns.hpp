#pragma once

#include "error.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <span>
#include <string>
#include <vector>

// NumPy on one side, the kernel ABI's column pointers on the other.  They are
// already the same layout: a C-contiguous (symbols, points) array holds each
// symbol's column contiguously, which is exactly what `xs[j]` has to be, so
// nothing here copies unless the caller handed over something that was not
// already doubles.
namespace ddx::py {

namespace pyb = pybind11;

using Array = pyb::array_t<double, pyb::array::c_style | pyb::array::forcecast>;

[[nodiscard]] constexpr pyb::ssize_t ssize(std::size_t n) {
  return static_cast<pyb::ssize_t>(n);
}

// A point, or a batch of them: one column per symbol, each `n` long.  Holds
// what it points into for as long as it lives, which is the whole of a call.
class Point {
public:
  Point(const pyb::handle &x, std::span<const std::string> symbols) {
    const std::size_t arity = symbols.size();
    columns_.assign(arity, nullptr);
    held_.reserve(arity);
    filled_.reserve(arity);

    if (pyb::isinstance<pyb::dict>(x)) {
      from_dict(pyb::reinterpret_borrow<pyb::dict>(x), symbols);
    } else {
      from_array(x, arity);
    }
  }

  [[nodiscard]] std::span<const double *const> columns() const {
    return columns_;
  }
  [[nodiscard]] std::size_t size() const noexcept { return n_; }
  // Whether the caller supplied a points axis, which is what decides whether
  // the answer carries one back.
  [[nodiscard]] bool batched() const noexcept { return batched_; }

private:
  // The keyword spelling.  C++ writes it named<"x">(v), whose FixedString is a
  // template parameter and cannot cross; a dict is the same thing with the name
  // as data, resolved the way Equation::assign_named resolves it.
  void from_dict(const pyb::dict &d, std::span<const std::string> symbols) {
    for (const auto item : d) {
      const auto name = pyb::cast<std::string>(item.first);
      if (std::ranges::find(symbols, name) == symbols.end()) {
        fail_with(errc::unknown_symbol);
      }
    }
    if (d.size() != symbols.size()) {
      fail_with(errc::wrong_arity); // a name is missing, which is not a typo
    }

    // Two passes: the batch length is whatever the 1-D values agree on, and a
    // scalar among them is held at every point of it.
    std::vector<Array> values;
    values.reserve(symbols.size());
    for (const auto &name : symbols) {
      auto a = pyb::cast<Array>(d[pyb::str(name)]);
      if (a.ndim() > 1) {
        fail_with(errc::wrong_arity);
      }
      if (a.ndim() == 1) {
        const auto len = static_cast<std::size_t>(a.shape(0));
        if (batched_ && len != n_) {
          fail_with(errc::wrong_arity);
        }
        n_ = len;
        batched_ = true;
      }
      values.push_back(std::move(a));
    }

    for (const auto [j, a] : std::views::enumerate(values)) {
      const auto slot = static_cast<std::size_t>(j);
      if (a.ndim() == 1 || n_ == 1) {
        columns_[slot] = a.data();
        held_.push_back(a);
      } else {
        columns_[slot] = filled_.emplace_back(n_, *a.data()).data();
      }
    }
  }

  // Positional: (arity,) is one point, (arity, n) a batch of n.  Rows are in
  // symbol order, which is what `symbols` reports.
  void from_array(const pyb::handle &x, std::size_t arity) {
    auto a = pyb::cast<Array>(x);
    if (a.ndim() != 1 && a.ndim() != 2) {
      fail_with(errc::wrong_arity);
    }
    if (static_cast<std::size_t>(a.shape(0)) != arity) {
      fail_with(errc::wrong_arity);
    }
    n_ = a.ndim() == 2 ? static_cast<std::size_t>(a.shape(1)) : 1;
    batched_ = a.ndim() == 2;

    const double *const base = a.data();
    for (const std::size_t j : std::views::iota(0uz, arity)) {
      columns_[j] = base + j * n_;
    }
    held_.push_back(std::move(a));
  }

  std::vector<const double *> columns_;
  std::vector<pyb::object> held_;           // what columns_ points into
  std::vector<std::vector<double>> filled_; // a scalar spread over a batch
  std::size_t n_ = 1;
  bool batched_ = false;
};

// One block of output columns: (columns, n), allocated flat because that is
// what the ABI writes into, and reshaped on the way out.
class Block {
public:
  Block(std::size_t columns, std::size_t n)
      : array_(std::vector<pyb::ssize_t>{ssize(columns), ssize(n)}),
        rows_(columns, nullptr) {
    double *const base = array_.mutable_data();
    for (const std::size_t j : std::views::iota(0uz, columns)) {
      rows_[j] = base + j * n;
    }
  }

  [[nodiscard]] std::span<double *const> rows() { return rows_; }
  [[nodiscard]] double *at(std::size_t column) { return rows_[column]; }

  // The trailing axis goes when the caller gave one point, the leading one when
  // there is a single function -- so a scalar model at a point answers with a
  // float, a gradient and an n x n Hessian rather than three arrays with 1s in
  // them.
  [[nodiscard]] pyb::object reshaped(std::vector<pyb::ssize_t> shape) && {
    return array_.attr("reshape")(pyb::cast(shape));
  }

private:
  Array array_;
  std::vector<double *> rows_;
};

} // namespace ddx::py
