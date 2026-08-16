#include "bound.hpp"
#include "equation.hpp"
#include "values.hpp"
#include <array>
#include <iostream>

int main() {
  using namespace diff;

  // Variables are symbols and carry no value, so an expression is an empty
  // type; the point is supplied at evaluation time.
  constexpr auto x = var<"x", int>;
  constexpr auto y = var<"y", int>;
  auto expr = x * y + PC(3) * x * y * y; // f(x, y) = x*y + 3*x*y²

  // A tree of pure symbols is an empty type however often a symbol repeats.
  auto sym_only = x * y + x * y * y;
  static_assert(std::is_empty_v<decltype(sym_only)>,
                "an expression over stateless leaves stores nothing");

  std::cout << "f(x,y) = " << expr << "\n";
  std::cout << "sizeof(symbols only) = " << sizeof(sym_only) << " byte\n";
  // A user-written Constant genuinely holds data, so the nodes above it have
  // to store their operands and stop being empty.
  std::cout << "sizeof(f) = " << sizeof(expr)
            << " bytes (one stored constant, plus the spine above it)\n";

  // Values are positional in CANONICAL (alphabetical) symbol order ...
  std::cout << "f(4,2) = " << expr.eval(4, 2) << "\n";
  // ... or bound by name, which is immune to that ordering.
  auto bf = bind(expr, named<"y">(2), named<"x">(4));
  std::cout << "f(4,2) = " << bf.eval() << "  (bound by name)\n";

  auto eq = Equation(expr);
  auto [dx, dy] = eq.eval_derivatives(std::array{4, 2});
  std::cout << "df/dx at (4,2) = " << dx << "\n";
  std::cout << "df/dy at (4,2) = " << dy << "\n";

  // trig: g(x) = sin(x) * cos(x)
  constexpr auto vx = var<"x">;
  auto trig = sin(vx) * cos(vx);
  std::cout << "\ng(1.0) = sin(x)*cos(x) = " << trig.eval(1.0) << "\n";
  std::cout << "g'(1.0) = " << trig.derivative().eval(1.0) << "\n";

  // --- Equation: f: R^2 -> R^2 ---
  constexpr auto vy = var<"y">;
  auto ve = Equation(vx + vy, vx * 3 * vy * 2 + 1);
  const std::array pt2{3.0, 4.0};
  std::cout << "\n--- Equation f(x,y) = (x+y, 6xy+1) at (3, 4) ---\n";
  std::cout << ve;

  auto fval = ve.evaluate(pt2);
  std::cout << "f(3,4) = (" << fval[0] << ", " << fval[1] << ")\n";

  auto J = ve.symbolic_mode_jac(pt2);
  std::cout << "Jacobian:\n";
  std::cout << "  [df0/dx, df0/dy] = [" << J[0][0] << ", " << J[0][1] << "]\n";
  std::cout << "  [df1/dx, df1/dy] = [" << J[1][0] << ", " << J[1][1] << "]\n";

  // --- Equation: f: R^2 -> R^3 ---
  auto ve3 = Equation(vx * vx, sin(vx) * vy, vx + vy * vy);
  const std::array pt3{1.0, 2.0};
  std::cout << "\n--- Equation f(x,y) = (x^2, sin(x)*y, x+y^2) at (1, 2) ---\n";
  std::cout << ve3;

  auto fval3 = ve3.evaluate(pt3);
  std::cout << "f(1,2) = (" << fval3[0] << ", " << fval3[1] << ", " << fval3[2]
            << ")\n";

  auto J3 = ve3.symbolic_mode_jac(pt3);
  std::cout << "Jacobian:\n";
  for (const auto &row : J3.rows()) {
    std::cout << "  [";
    for (std::size_t j = 0; j < row.size(); ++j)
      std::cout << (j ? ", " : "") << row[j];
    std::cout << "]\n";
  }
}
