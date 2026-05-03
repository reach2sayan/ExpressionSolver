#pragma once

#include "expressions.hpp"
#include <cmath>
#include <ostream>
#include <tuple>
#include <utility>

namespace diff {

template <typename T> class Dual {
private:
  T val{};
  T deriv{};

public:
  constexpr Dual() = default;
  constexpr explicit Dual(T v, T d = T{}) : val(v), deriv(d) {}

  constexpr Dual operator+(const Dual &o) const {
    return Dual{val + o.val, deriv + o.deriv};
  }
  constexpr Dual operator-(const Dual &o) const {
    return Dual{val - o.val, deriv - o.deriv};
  }
  constexpr Dual operator*(const Dual &o) const {
    return Dual{val * o.val, deriv * o.val + val * o.deriv};
  }
  constexpr Dual operator/(const Dual &o) const {
    return Dual{val / o.val, (deriv * o.val - val * o.deriv) / (o.val * o.val)};
  }
  constexpr Dual &operator+=(const Dual &o) {
    val += o.val;
    deriv += o.deriv;
    return *this;
  }
  constexpr Dual &operator-=(const Dual &o) {
    val -= o.val;
    deriv -= o.deriv;
    return *this;
  }
  constexpr Dual &operator*=(const Dual &o) {
    *this = *this * o;
    return *this;
  }
  constexpr Dual &operator/=(const Dual &o) {
    *this = *this / o;
    return *this;
  }
  constexpr Dual operator-() const { return Dual{-val, -deriv}; }
  constexpr Dual &operator++() {
    ++val;
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &out, const Dual &d) {
    return out << d.val << "+" << d.deriv << "e";
  }

  template <std::size_t Index> [[nodiscard]] constexpr auto get() const {
    static_assert(Index < 2, "Dual index out of bounds");
    if constexpr (Index == 0) {
      return val;
    }
    else {
      return deriv;
    }
  }

  [[nodiscard]] friend constexpr Dual sin(const Dual &d) {
    using std::sin, std::cos;
    return Dual{sin(d.val), cos(d.val) * d.deriv};
  }
  [[nodiscard]] friend constexpr Dual cos(const Dual &d) {
    using std::sin, std::cos;
    return Dual{cos(d.val), -sin(d.val) * d.deriv};
  }
  [[nodiscard]] friend constexpr Dual exp(const Dual &d) {
    using std::exp;
    const T e = exp(d.val);
    return Dual{e, e * d.deriv};
  }
  [[nodiscard]] friend constexpr Dual tan(const Dual &d) {
    using std::tan, std::cos;
    const T c = cos(d.val);
    return Dual{tan(d.val), d.deriv / (c * c)};
  }
  [[nodiscard]] friend constexpr Dual log(const Dual &d) {
    using std::log;
    return Dual{log(d.val), d.deriv / d.val};
  }
  [[nodiscard]] friend constexpr Dual sqrt(const Dual &d) {
    using std::sqrt;
    const T s = sqrt(d.val);
    return Dual{s, d.deriv / (T{2} * s)};
  }
  [[nodiscard]] friend constexpr Dual abs(const Dual &d) {
    using std::abs;
    const T sign = d.val > T{} ? T{1} : d.val < T{} ? T{-1} : T{};
    return Dual{abs(d.val), sign * d.deriv};
  }
  [[nodiscard]] friend constexpr Dual asin(const Dual &d) {
    using std::asin, std::sqrt;
    return Dual{asin(d.val), d.deriv / sqrt(T{1} - d.val * d.val)};
  }
  [[nodiscard]] friend constexpr Dual acos(const Dual &d) {
    using std::acos, std::sqrt;
    return Dual{acos(d.val), -d.deriv / sqrt(T{1} - d.val * d.val)};
  }
  [[nodiscard]] friend constexpr Dual atan(const Dual &d) {
    using std::atan;
    return Dual{atan(d.val), d.deriv / (T{1} + d.val * d.val)};
  }
  [[nodiscard]] friend constexpr Dual sinh(const Dual &d) {
    using std::sinh, std::cosh;
    return Dual{sinh(d.val), cosh(d.val) * d.deriv};
  }
  [[nodiscard]] friend constexpr Dual cosh(const Dual &d) {
    using std::sinh, std::cosh;
    return Dual{cosh(d.val), sinh(d.val) * d.deriv};
  }
  [[nodiscard]] friend constexpr Dual tanh(const Dual &d) {
    using std::tanh, std::cosh;
    const T c = cosh(d.val);
    return Dual{tanh(d.val), d.deriv / (c * c)};
  }
};

static_assert(Numeric<Dual<double>>);
static_assert(Numeric<Dual<float>>);

namespace {
template <typename T> T dual_scalar_impl(T &&);
template <typename T> T dual_scalar_impl(Dual<T> &&);
template <typename T> consteval bool is_dual_impl(std::type_identity<T>) {
  return false;
}
template <typename T> consteval bool is_dual_impl(std::type_identity<Dual<T>>) {
  return true;
}
} // anonymous namespace

template <typename T>
inline constexpr bool is_dual_v = is_dual_impl(std::type_identity<T>{});
template <typename T>
using dual_scalar_t = decltype(dual_scalar_impl(std::declval<T>()));

} // namespace diff

namespace std {
template <typename T>
struct tuple_size<diff::Dual<T>> : integral_constant<std::size_t, 2> {};
template <typename T, std::size_t N> struct tuple_element<N, diff::Dual<T>> {
  using type = T;
};
} // namespace std
