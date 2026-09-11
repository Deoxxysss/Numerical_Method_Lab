```markdown
# Numerical Methods Lab

Hey! Welcome to my Numerical Methods Lab — a Python package I built for my high school project. It's a collection of classic numerical algorithms that solve problems you can't always crack with pen and paper: finding roots of equations, approximating integrals, and solving systems of linear equations.

I wrote this because I wanted to actually understand how these methods work — not just plug numbers into a calculator. So I implemented everything from scratch, tested it against known answers, and packaged it up so anyone can install and use it.

---

## Overview

This package gives you clean, readable implementations of three big topics in numerical analysis:

- **Root Finding** — bisection and Newton-Raphson methods
- **Numerical Integration** — left/right rectangle rules, trapezoidal rule, and Simpson's rule
- **Linear Algebra** — Gaussian elimination with partial pivoting, and back substitution

Everything is written using NumPy so it's fast, vectorized, and easy to read. And yes, there are tests for everything.

---

## Mathematical Background

If you're curious about the theory behind each method, I've written a full walkthrough in `notebooks/exploration.ipynb`. It covers:

- What a root is, why we need numerical methods, and how bisection and Newton-Raphson actually work (with the tangent-line picture for Newton)
- How rectangle rules, the trapezoidal rule, and Simpson's rule approximate the area under a curve — and why Simpson's is so much more accurate
- How Gaussian elimination turns a messy system into an upper-triangular one you can solve from the bottom up

There's also a section on why I wrote a separate `derivative` function for Newton-Raphson, and why partial pivoting matters when you're dividing by numbers close to zero.

---

## Algorithms Implemented

Here's a quick breakdown of what's inside:

| Method | Function | File |
|---|---|---|
| Bisection | `bisection_method` | `root_finding.py` |
| Newton-Raphson | `Newton_raphson_method` | `root_finding.py` |
| Numerical derivative | `derivative` | `root_finding.py` |
| Left rectangle rule | `rectanglel` | `integration.py` |
| Right rectangle rule | `rectangler` | `integration.py` |
| Trapezoidal rule | `trapzoidal` | `integration.py` |
| Simpson's rule | `simpson` | `integration.py` |
| Gaussian elimination | `gaussian_elimination` | `linear_algebra.py` |
| Back substitution | `back_substitution` | `linear_algebra.py` |

---

## Results

Some highlights from testing:

- **Bisection** reliably finds √2 to within 1e-10 in about 35 iterations — slow but steady.
- **Newton-Raphson** converges to the same root in under 6 iterations when starting from 1.5. That's the quadratic convergence in action.
- **Simpson's rule** is exact for cubic polynomials and gets `e^x` integrals right to machine precision with just a few hundred intervals.
- **Gaussian elimination** matches `numpy.linalg.solve` to within floating-point tolerance on every system I tested — even ones that need pivoting.

---

## Example Usage

Once installed, you can start using it in three lines:

```python
from numerical_method_lab.root_finding import bisection_method

root = bisection_method(lambda x: x**2 - 2, 1, 2, tol=1e-8)
print(root)  # ≈ 1.41421356
```

A more complete example with all three modules:

```python
import numpy as np
from numerical_method_lab.root_finding import bisection_method, Newton_raphson_method
from numerical_method_lab.integration import simpson
from numerical_method_lab.linear_algebra import gaussian_elimination

# Find √2
f = lambda x: x**2 - 2
print("Bisection:", bisection_method(f, 1, 2))
print("Newton:   ", Newton_raphson_method(f, 1.5))

# Integrate e^x from 0 to 1
print("Integral:", simpson(np.exp, 0, 1, n=100))

# Solve a linear system
A = np.array([[3.0, 1.0, -1.0],
              [2.0, -2.0, 4.0],
              [-1.0, 0.5, -1.0]])
b = np.array([1.0, -2.0, 0.0])
print("Solution:", gaussian_elimination(A, b))
```

There are more examples in `examples/examples.py`, and the notebook has even more.

---

## Project Structure

Here's how everything is organized:

```
Numerical_Method_Lab/
├── README.md
├── pyproject.toml
├── LICENSE
├── src/
│   └── numerical_method_lab/
│       ├── __init__.py
│       ├── root_finding.py
│       ├── integration.py
│       └── linear_algebra.py
├── tests/
│   ├── test_root_finding.py
│   ├── test_integration.py
│   └── test_linear_algebra.py
├── notebooks/
│   └── exploration.ipynb
├── examples/
│   └── examples.py
└── figures/
```

If you want to see the source code, it's all in `src/numerical_method_lab/`. Everything else is supporting material.

---

## Installation

You can install it straight from PyPI (or TestPyPI if you want the pre-release):

```bash
pip install numerical-method-lab
```

If you only have it on TestPyPI:

```bash
pip install --extra-index-url https://test.pypi.org/simple/ numerical-method-lab
```

Or if you've cloned the repo and want to develop on it:

```bash
git clone https://github.com/yourusername/numerical-methods-lab.git
cd numerical-methods-lab
pip install -e .
```

The `-e` means "editable" — any changes you make to the source code show up immediately without reinstalling.

---

## Testing

I wrote a full test suite using `pytest`. To run it:

```bash
pytest -v
```

That will run all 35+ tests across the three modules. They check everything from basic correctness (e.g., does bisection find √2?) to edge cases (e.g., what happens when the interval has the same sign at both ends?) to performance (e.g., does Simpson's rule actually beat the trapezoidal rule?).

You can also run just one file:

```bash
pytest tests/test_root_finding.py -v
```

---

## Limitations

I'll be honest — this is a high school project, not a production library. Some things to keep in mind:

- **Root finding**: Newton-Raphson can fail if you start too far from the root or if the derivative is zero. Bisection needs a bracketing interval — it won't work if you don't have one.
- **Integration**: All methods assume the function is reasonably smooth. For functions with sharp peaks or discontinuities, accuracy drops fast.
- **Linear algebra**: Only handles square systems with unique solutions. Singular or rectangular systems will raise errors.
- **Performance**: Gaussian elimination is O(n³), so it gets slow for very large matrices. Iterative methods would be better there.
- **No adaptive step size**: The integration functions use a fixed number of intervals. Adaptive quadrature would be smarter but also more complex.

---

## Future Improvements

Things I'd love to add if I keep working on this:

- **Secant method** and **fixed-point iteration** for root finding
- **Adaptive Simpson's rule** that automatically refines the mesh where it matters
- **LU decomposition** and **iterative solvers** (Jacobi, Gauss-Seidel) for linear systems
- **Convergence visualizations** as part of the package, not just the notebook
- **More thorough error analysis** — comparing convergence rates experimentally
- **Type hints** and **docstrings** for every function
- **Continuous integration** with GitHub Actions

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Thanks for checking out my project! If you have any questions or spot a bug, feel free to open an issue or reach out. Happy computing!
```