import math
import pytest
from src.numerical_method_lab.root_finding import (
    bisection_method,
    Newton_raphson_method,
    derivative
)


# ----------------------------------------------------------------------
# Tests for the bisection method
# ----------------------------------------------------------------------
def test_bisection_finds_sqrt2():
    """Bisection should find sqrt(2) in [1, 2]."""
    f = lambda x: x**2 - 2
    root = bisection_method(f, 1, 2, tol=1e-8)
    assert math.isclose(root, math.sqrt(2), rel_tol=1e-8)


def test_bisection_finds_root_of_linear():
    """Bisection should find the root of 2x - 6 = 0."""
    f = lambda x: 2 * x - 6
    root = bisection_method(f, 0, 5)
    assert math.isclose(root, 3.0, rel_tol=1e-5)


def test_bisection_handles_tight_tolerance():
    """Bisection should get very close to the true root with small tol."""
    f = lambda x: x**3 - x - 2
    root = bisection_method(f, 1, 2, tol=1e-12, max_iter=200)
    # true root is approximately 1.5213797068045676
    assert math.isclose(root, 1.5213797068, rel_tol=1e-9)


def test_bisection_invalid_interval_raises():
    """If f(a) and f(b) have the same sign, a ValueError must be raised."""
    f = lambda x: x**2 + 1  # always positive
    with pytest.raises(ValueError):
        bisection_method(f, -1, 1)


def test_bisection_returns_number_even_if_max_iter_reached():
    """If max_iter is too small, the method should still return a number."""
    f = lambda x: x**2 - 2
    root = bisection_method(f, 1, 2, tol=1e-12, max_iter=2)
    assert isinstance(root, float)
    assert 1.0 <= root <= 2.0


# ----------------------------------------------------------------------
# Tests for the numerical derivative function
# ----------------------------------------------------------------------
def test_derivative_of_linear():
    """The derivative of f(x) = 3x + 2 should be 3."""
    f = lambda x: 3 * x + 2
    d = derivative(f, 5.0)
    assert math.isclose(d, 3.0, rel_tol=1e-5)


def test_derivative_of_quadratic():
    """The derivative of f(x) = x^2 at x=2 should be 4."""
    f = lambda x: x**2
    d = derivative(f, 2.0)
    assert math.isclose(d, 4.0, rel_tol=1e-4)


# ----------------------------------------------------------------------
# Tests for Newton Raphson method
# ----------------------------------------------------------------------
def test_newton_finds_sqrt2():
    """Newton Raphson should find sqrt(2) starting from 1.5."""
    f = lambda x: x**2 - 2
    root = Newton_raphson_method(f, 1.5, tol=1e-8)
    assert math.isclose(root, math.sqrt(2), rel_tol=1e-8)


def test_newton_with_analytic_derivative():
    """
    Newton Raphson should work when an analytic derivative is supplied.
    The derivative function passed must accept two arguments (f, x) even if
    the first is unused.
    """
    f = lambda x: x**3 - x - 2
    df = lambda f, x: 3 * x**2 - 1  # ignore f
    root = Newton_raphson_method(f, 1.5, df=df, tol=1e-8)
    assert math.isclose(root, 1.5213797068, rel_tol=1e-6)


def test_newton_uses_numerical_derivative_by_default():
    """Without a custom derivative, Newton Raphson should still converge."""
    f = lambda x: x**2 - 2
    root = Newton_raphson_method(f, 1.4, tol=1e-8)
    assert math.isclose(root, math.sqrt(2), rel_tol=1e-6)


def test_newton_raises_when_derivative_zero():
    """If the derivative is (nearly) zero, a ValueError should be raised."""
    f = lambda x: x**2 - 2
    # Starting at x=0, derivative = 0, so the method should fail early.
    with pytest.raises(ValueError):
        Newton_raphson_method(f, 0.0, max_iter=10)


def test_newton_raises_on_non_convergence():
    """If the method does not converge within max_iter, it should raise."""
    f = lambda x: x**3 - 2 * x + 2  # chosen to cause oscillation from 0
    with pytest.raises(ValueError):
        Newton_raphson_method(f, 0.0, max_iter=5)