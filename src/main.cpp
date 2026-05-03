#include "equation.hpp"
#include "values.hpp"
#include <iostream>

int main() {
  // f(x, y) = x * y + 3 * x * y^2
  auto x = PV(4, 'x');
  auto y = PV(2, 'y');
  auto expr = x * y + PC(3) * x * y * y;

  std::cout << "f(x,y) = " << expr << "\n";
  std::cout << "f(4,2) = " << expr.eval() << "\n";

  auto eq = diff::Equation(expr);
  auto [dx, dy] = eq.eval_derivatives();
  std::cout << "df/dx at (4,2) = " << dx << "\n";
  std::cout << "df/dy at (4,2) = " << dy << "\n";

  // trig: g(x) = sin(x) * cos(x)
  auto vx = PV(1.0, 'x');
  auto trig = sin(vx) * cos(vx);
  std::cout << "\ng(1.0) = sin(x)*cos(x) = " << trig.eval() << "\n";
  std::cout << "g'(1.0) = " << trig.derivative().eval() << "\n";

  // --- Equation: f: R^2 -> R^2 ---
  // f(x, y) = (x + y,  x * y)
  auto vx2 = PV(3.0, 'x');
  auto vy2 = PV(4.0, 'y');
  auto ve = diff::Equation(vx2 + vy2, vx2 * 3 * vy2 * 2 + 1);
  std::cout << "\n--- Equation f(x,y) = (x+y, x*y) at (3, 4) ---\n";
  std::cout << ve;

  auto fval = ve.evaluate();
  std::cout << "f(3,4) = (" << fval[0] << ", " << fval[1] << ")\n";

  auto J = ve.symbolic_mode_jac();
  std::cout << "Jacobian:\n";
  std::cout << "  [df0/dx, df0/dy] = [" << J.row(0) << "]\n";
  std::cout << "  [df1/dx, df1/dy] = [" << J.row(1) << "]\n";

  // --- Equation: f: R^2 -> R^3 ---
  // f(x, y) = (x*x,  sin(x)*y,  x + y*y)
  auto vx3 = PV(1.0, 'x');
  auto vy3 = PV(2.0, 'y');
  auto ve3 = diff::Equation(vx3 * vx3, sin(vx3) * vy3, vx3 + vy3 * vy3);
  std::cout << "\n--- Equation f(x,y) = (x^2, sin(x)*y, x+y^2) at (1, 2) ---\n";
  std::cout << ve3;

  auto fval3 = ve3.evaluate();
  std::cout << "f(1,2) = (" << fval3[0] << ", " << fval3[1] << ", " << fval3[2]
            << ")\n";

  auto J3 = ve3.symbolic_mode_jac();
  std::cout << "Jacobian:\n";
  std::cout << J3;
}
