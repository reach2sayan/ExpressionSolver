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
  // `names` is the symbol list as interned Python strings, held by the equation:
  // a dict lookup needs a key object, and building one per symbol per call was
  // most of what a dict point cost.
  Point(const pyb::handle &x, std::span<const std::string> symbols,
        std::span<const pyb::object> names)
      : columns_(symbols.size(), nullptr) {
    if (pyb::isinstance<pyb::dict>(x)) {
      const auto d = pyb::reinterpret_borrow<pyb::dict>(x);
      if (!scalars_from_dict(d, names)) {
        from_dict(d, symbols);
      }
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

  // The common case: a dict of plain floats at one point.  One lookup per
  // symbol against an interned key, the value read as a double, and no numpy
  // object anywhere -- where the general path below builds an Array per symbol
  // and prescans every key against every symbol.
  //
  // False means "not this shape", not "bad input": anything that is not a float
  // falls through to from_dict, which is what reports the errors.  A size
  // mismatch is settled there too, so the two paths cannot disagree.
  [[nodiscard]] bool scalars_from_dict(const pyb::dict &d,
                                       std::span<const pyb::object> names) {
    if (d.size() != names.size()) {
      return false;
    }
    scalars_.resize(names.size());
    for (const auto [i, name] : std::views::enumerate(names)) {
      PyObject *const v = PyDict_GetItemWithError(d.ptr(), name.ptr());
      if (v == nullptr || PyFloat_CheckExact(v) == 0) {
        return PyErr_Occurred() != nullptr ? throw pyb::error_already_set()
                                           : false;
      }
      scalars_[static_cast<std::size_t>(i)] = PyFloat_AS_DOUBLE(v);
    }
    std::ranges::transform(scalars_, columns_.begin(),
                           [](const double &v) { return &v; });
    n_ = 1;
    batched_ = false;
    return true;
  }

  void from_dict(const pyb::dict &d, std::span<const std::string> symbols) {
    held_.reserve(symbols.size());
    filled_.reserve(symbols.size());
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
  std::vector<double> scalars_;             // a dict of floats, read in place
  std::vector<pyb::object> held_;           // what columns_ points into
  std::vector<std::vector<double>> filled_; // a scalar spread over a batch
  std::size_t n_ = 1;
  bool batched_ = false;
};

// One block of output columns.  The ABI writes (columns, n) row-major, which is
// the same memory as the shape the caller wants -- so the array is built at that
// shape once and never reshaped.  Reshaping cost a Python attribute lookup and
// call per output, and PyObject_GetAttrString does not intern, so it built the
// name string afresh on every call.
class Block {
public:
  Block(const std::vector<pyb::ssize_t> &shape, std::size_t columns,
        std::size_t n)
      : array_(shape), rows_(columns, nullptr) {
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
  [[nodiscard]] pyb::object array() && { return std::move(array_); }

  Array array_;
  std::vector<double *> rows_;
};

// Columns the caller never sees: the compressed Hessian on its way to a dense
// one.  A plain buffer, since an ndarray for a value that is discarded is a
// Python object and a second heap block bought for nothing.
class Scratch {
public:
  Scratch(std::size_t columns, std::size_t n)
      : data_(columns * n), rows_(columns, nullptr) {
    std::ranges::transform(
        std::views::iota(0uz, columns), rows_.begin(),
        [base = data_.data(), n](std::size_t j) { return base + j * n; });
  }

private:
  friend class PyEquation;

  [[nodiscard]] constexpr std::span<double *const> rows() { return rows_; }

  std::vector<double> data_;
  std::vector<double *> rows_;
};

} // namespace ddx::py
