"""
Example Problems Solved Using Numerical Methods Lab
---------------------------------------------------
Each problem demonstrates the use of one or more functions from the package.
Run this script to see the numerical solutions and compare with known exact values.
"""

import numpy as np
from numerical_method_lab.root_finding import bisection_method, Newton_raphson_method
from numerical_method_lab.integration import trapezoidal, simpson
from numerical_method_lab.linear_algebra import gaussian_elimination

def problem1_sqrt3():
    """Find sqrt(3) using bisection and Newton-Raphson."""
    print("Problem 1: Approximate sqrt(3)")
    f = lambda x: x**2 - 3
    true_val = np.sqrt(3)

    root_b = bisection_method(f, 1, 2, tol=1e-8)
    root_n = Newton_raphson_method(f, 1.5, tol=1e-8)

    print(f"  Bisection result : {root_b:.10f} (error = {abs(root_b - true_val):.2e})")
    print(f"  Newton result    : {root_n:.10f} (error = {abs(root_n - true_val):.2e})")
    print(f"  Exact value      : {true_val:.10f}\n")

def problem2_cosx_minus_x():
    """Find the root of cos(x) - x = 0."""
    print("Problem 2: Solve cos(x) = x")
    f = lambda x: np.cos(x) - x
    # initial guess for Newton
    root_n = Newton_raphson_method(f, 0.5, tol=1e-10)
    print(f"  Newton result    : {root_n:.10f}")
    print(f"  Verification f(x): {f(root_n):.2e}\n")

def problem3_integrate_1_over_1plusx2():
    """Integrate 1/(1+x^2) from 0 to 1, exact = pi/4."""
    print("Problem 3: Compute ∫₀¹ 1/(1+x²) dx")
    f = lambda x: 1/(1 + x**2)
    a, b = 0, 1
    exact = np.pi / 4

    trap = trapezoidal(f, a, b, n=1000)
    simp = simpson(f, a, b, n=1000)

    print(f"  Trapezoidal (n=1000) : {trap:.10f} (error = {abs(trap - exact):.2e})")
    print(f"  Simpson (n=1000)     : {simp:.10f} (error = {abs(simp - exact):.2e})")
    print(f"  Exact value          : {exact:.10f}\n")

def problem4_integrate_exp_minus_x2():
    """Integrate exp(-x^2) from 0 to 1 (no elementary antiderivative)."""
    print("Problem 4: Compute ∫₀¹ e^(-x²) dx")
    f = lambda x: np.exp(-x**2)
    a, b = 0, 1

    trap = trapezoidal(f, a, b, n=2000)
    simp = simpson(f, a, b, n=2000)

    # reference value from scipy (or known approximation)
    # scipy is not imported to keep dependencies minimal; using high-accuracy approximation
    reference = 0.746824132812427
    print(f"  Trapezoidal (n=2000) : {trap:.12f}")
    print(f"  Simpson (n=2000)     : {simp:.12f}")
    print(f"  Reference value      : {reference:.12f}\n")

def problem5_solve_3x3_system():
    """Solve a 3x3 linear system."""
    print("Problem 5: Solve 3x3 linear system")
    A = np.array([[3.0, 1.0, -1.0],
                  [2.0, -2.0, 4.0],
                  [-1.0, 0.5, -1.0]])
    b = np.array([1.0, -2.0, 0.0])

    x = gaussian_elimination(A, b)
    x_np = np.linalg.solve(A, b)

    print("  Our solution      :", x)
    print("  NumPy solution    :", x_np)
    print("  Difference norm   :", np.linalg.norm(x - x_np), "\n")

def problem6_solve_4x4_pivoting():
    """Solve a 4x4 system that requires pivoting."""
    print("Problem 6: Solve 4x4 system (requires partial pivoting)")
    A = np.array([[0.0, 2.0, 1.0, 1.0],
                  [2.0, 1.0, 0.0, 1.0],
                  [1.0, 0.0, 3.0, 2.0],
                  [2.0, 1.0, 1.0, 4.0]])
    b = np.array([5.0, 3.0, 6.0, 8.0])

    x = gaussian_elimination(A, b)
    x_np = np.linalg.solve(A, b)

    print("  Our solution      :", x)
    print("  NumPy solution    :", x_np)
    print("  Difference norm   :", np.linalg.norm(x - x_np), "\n")

if __name__ == "__main__":
    problem1_sqrt3()
    problem2_cosx_minus_x()
    problem3_integrate_1_over_1plusx2()
    problem4_integrate_exp_minus_x2()
    problem5_solve_3x3_system()
    problem6_solve_4x4_pivoting()