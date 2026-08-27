#pragma once

#include "error.hpp"

#include "util/ranges.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <span>
#include <string>
#include <vector>

// NumPy and the kernel ABI already agree: a C-contiguous (symbols, points)
namespace ddx::py {

namespace pyb = pybind11;

class PyEquation;

using Array = pyb::array_t<double, pyb::array::c_style | pyb::array::forcecast>;

[[nodiscard]] constexpr pyb::ssize_t ssize(std::size_t n) noexcept {
  return static_cast<pyb::ssize_t>(n);
}

class Point {
public:
  Point(const pyb::handle &x, std::span<const std::string> symbols)
      : columns_(symbols.size(), nullptr) {
    held_.reserve(symbols.size());
    filled_.reserve(symbols.size());
    if (pyb::isinstance<pyb::dict>(x)) {
      from_dict(pyb::reinterpret_borrow<pyb::dict>(x), symbols);
    } else {
      from_array(x, symbols.size());
    }
  }

private:
  friend class PyEquation;

  [[nodiscard]] constexpr std::span<const double *const> columns() const {
    return columns_;
  }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return n_; }
  [[nodiscard]] constexpr bool batched() const noexcept { return batched_; }

  void from_dict(const pyb::dict &d, std::span<const std::string> symbols) {
    const auto named = [symbols](const auto item) {
      return std::ranges::contains(symbols, pyb::cast<std::string>(item.first));
    };
    if (!std::ranges::all_of(d, named)) {
      fail_with(errc::unknown_symbol);
    }
    if (d.size() != symbols.size()) {
      fail_with(errc::wrong_arity); // a name is missing, which is not a typo
    }

    const auto values = symbols |
                        std::views::transform([&d](const std::string &name) {
                          auto a = pyb::cast<Array>(d[pyb::str(name)]);
                          if (a.ndim() > 1) {
                            fail_with(errc::wrong_arity);
                          }
                          return a;
                        }) |
                        impl::to<std::vector<Array>>();

    for (const Array &a : values | std::views::filter([](const Array &a) {
                            return a.ndim() == 1;
                          })) {
      const auto len = static_cast<std::size_t>(a.shape(0));
      if (batched_ && len != n_) {
        fail_with(errc::wrong_arity);
      }
      n_ = len;
      batched_ = true;
    }

    for (auto [column, a] : std::views::zip(columns_, values)) {
      if (a.ndim() == 1 || n_ == 1) {
        column = a.data();
        held_.push_back(a);
      } else {
        column = filled_.emplace_back(n_, *a.data()).data();
      }
    }
  }

  // Rows are in symbol order
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

    std::ranges::transform(
        std::views::iota(0uz, arity), columns_.begin(),
        [base = a.data(), this](std::size_t j) { return base + j * n_; });
    held_.push_back(std::move(a));
  }

  std::vector<const double *> columns_;
  std::vector<pyb::object> held_;           // what columns_ points into
  std::vector<std::vector<double>> filled_; // a scalar spread over a batch
  std::size_t n_ = 1;
  bool batched_ = false;
};

// One block of output columns: (columns, n), flat because that is what the ABI
// writes into, and reshaped on the way out.
class Block {
public:
  Block(std::size_t columns, std::size_t n)
      : array_(std::vector<pyb::ssize_t>{ssize(columns), ssize(n)}),
        rows_(columns, nullptr) {
    std::ranges::transform(std::views::iota(0uz, columns), rows_.begin(),
                           [base = array_.mutable_data(), n](std::size_t j) {
                             return base + j * n;
                           });
  }

private:
  friend class PyEquation;

  [[nodiscard]] constexpr std::span<double *const> rows() { return rows_; }
  [[nodiscard]] constexpr double *at(std::size_t column) {
    return rows_[column];
  }

  [[nodiscard]] pyb::object reshaped(std::vector<pyb::ssize_t> shape) && {
    return array_.attr("reshape")(pyb::cast(shape));
  }

  Array array_;
  std::vector<double *> rows_;
};

} // namespace ddx::py
