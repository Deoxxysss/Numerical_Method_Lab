import math
import numpy as np
import pytest
from src.numerical_method_lab.integration import trapzoidal, rectanglel, rectangler, simpson


# ----------------------------------------------------------------------
# Helper: exact integrals of test functions
# ----------------------------------------------------------------------
def f_linear(x):
    return 2 * x + 3          # integral from a to b: x^2 + 3x

def f_quadratic(x):
    return x**2 - 4*x + 5     # integral: x^3/3 - 2x^2 + 5x

def f_exp(x):
    return np.exp(x)          # integral: e^x

def f_sin(x):
    return np.sin(x)          # integral: -cos(x)

# Exact values for specific intervals
def exact_integral(f_name, a, b):
    if f_name == 'linear':
        return (b**2 - a**2) + 3*(b - a)
    elif f_name == 'quadratic':
        return (b**3 - a**3)/3 - 2*(b**2 - a**2) + 5*(b - a)
    elif f_name == 'exp':
        return np.exp(b) - np.exp(a)
    elif f_name == 'sin':
        return -np.cos(b) + np.cos(a)
    else:
        raise ValueError("Unknown function name")


# ----------------------------------------------------------------------
# Tests for trapezoidal rule
# ----------------------------------------------------------------------
def test_trapezoidal_linear_exact():
    """Trapezoidal rule should be exact for linear functions."""
    a, b = 0.0, 3.0
    result = trapzoidal(f_linear, a, b, n=10)   # any n works
    expected = exact_integral('linear', a, b)
    assert math.isclose(result, expected, rel_tol=1e-10)

def test_trapezoidal_quadratic_accuracy():
    """Trapezoidal rule should approximate quadratic with small error for large n."""
    a, b = 0.0, 2.0
    result = trapzoidal(f_quadratic, a, b, n=10000)
    expected = exact_integral('quadratic', a, b)
    assert math.isclose(result, expected, rel_tol=1e-8)

def test_trapezoidal_exp_accuracy():
    """Trapezoidal rule for e^x with n=10000 should be very close."""
    a, b = 0.0, 1.0
    result = trapzoidal(f_exp, a, b, n=10000)
    expected = exact_integral('exp', a, b)
    assert math.isclose(result, expected, rel_tol=1e-8)

def test_trapezoidal_convergence():
    """Error should decrease as n increases."""
    a, b = 0.0, 1.0
    expected = exact_integral('sin', a, b)
    err1 = abs(trapzoidal(f_sin, a, b, n=100) - expected)
    err2 = abs(trapzoidal(f_sin, a, b, n=1000) - expected)
    assert err2 < err1   # larger n should give smaller error


# ----------------------------------------------------------------------
# Tests for left rectangle rule
# ----------------------------------------------------------------------
def test_rectanglel_linear_overestimates_or_underestimates():
    """For increasing function, left rectangle underestimates; for decreasing, overestimates."""
    a, b = 0.0, 1.0
    # f_linear = 2x+3 is increasing, so left rectangles underestimate
    result = rectanglel(f_linear, a, b, n=1000)
    expected = exact_integral('linear', a, b)
    assert result < expected

def test_rectanglel_accuracy():
    """With large n, left rectangle should approximate integral."""
    a, b = 0.0, 2.0
    result = rectanglel(f_quadratic, a, b, n=10000)
    expected = exact_integral('quadratic', a, b)
    assert math.isclose(result, expected, rel_tol=1e-4)

def test_rectanglel_convergence():
    """Error decreases with increasing n."""
    a, b = 0.0, 1.0
    expected = exact_integral('exp', a, b)
    err1 = abs(rectanglel(f_exp, a, b, n=100) - expected)
    err2 = abs(rectanglel(f_exp, a, b, n=1000) - expected)
    assert err2 < err1


# ----------------------------------------------------------------------
# Tests for right rectangle rule
# ----------------------------------------------------------------------
def test_rectangler_linear_overestimates_or_underestimates():
    """For increasing function, right rectangle overestimates."""
    a, b = 0.0, 1.0
    result = rectangler(f_linear, a, b, n=1000)
    expected = exact_integral('linear', a, b)
    assert result > expected

def test_rectangler_accuracy():
    """With large n, right rectangle should approximate integral."""
    a, b = 0.0, 2.0
    result = rectangler(f_quadratic, a, b, n=10000)
    expected = exact_integral('quadratic', a, b)
    assert math.isclose(result, expected, rel_tol=1e-4)

def test_rectangler_convergence():
    """Error decreases with increasing n."""
    a, b = 0.0, 1.0
    expected = exact_integral('exp', a, b)
    err1 = abs(rectangler(f_exp, a, b, n=100) - expected)
    err2 = abs(rectangler(f_exp, a, b, n=1000) - expected)
    assert err2 < err1


# ----------------------------------------------------------------------
# Tests for Simpson's rule
# ----------------------------------------------------------------------
def test_simpson_quadratic_exact():
    """Simpson's rule should be exact for quadratic functions (degree <= 3)."""
    a, b = 0.0, 3.0
    result = simpson(f_quadratic, a, b, n=2)   # minimal even n
    expected = exact_integral('quadratic', a, b)
    assert math.isclose(result, expected, rel_tol=1e-10)

def test_simpson_cubic_exact():
    """Simpson's rule is also exact for cubic polynomials."""
    f_cubic = lambda x: x**3 - 2*x + 1
    a, b = 0.0, 2.0
    # Exact integral: x^4/4 - x^2 + x evaluated from 0 to 2
    expected = (2**4)/4 - 2**2 + 2
    result = simpson(f_cubic, a, b, n=2)
    assert math.isclose(result, expected, rel_tol=1e-10)

def test_simpson_exp_accuracy():
    """Simpson's rule for e^x with modest n is very accurate."""
    a, b = 0.0, 1.0
    result = simpson(f_exp, a, b, n=100)
    expected = exact_integral('exp', a, b)
    assert math.isclose(result, expected, rel_tol=1e-10)

def test_simpson_odd_n_adjusts_to_even():
    """If n is odd, the function should internally make it even and still work."""
    a, b = 0.0, 1.0
    result_odd = simpson(f_sin, a, b, n=101)   # odd n
    result_even = simpson(f_sin, a, b, n=102)  # even n
    expected = exact_integral('sin', a, b)
    assert math.isclose(result_odd, expected, rel_tol=1e-8)
    assert math.isclose(result_even, expected, rel_tol=1e-8)

def test_simpson_convergence():
    """Error decreases with n."""
    a, b = 0.0, 2.0
    expected = exact_integral('sin', a, b)
    err1 = abs(simpson(f_sin, a, b, n=20) - expected)
    err2 = abs(simpson(f_sin, a, b, n=40) - expected)
    assert err2 < err1


# ----------------------------------------------------------------------
# Test that all methods handle n=1 without crashing
# ----------------------------------------------------------------------
def test_n1_no_crash():
    """Each method should at least return a finite number for n=1."""
    a, b = 0.0, 1.0
    f = lambda x: x**2
    assert np.isfinite(trapzoidal(f, a, b, n=1))
    assert np.isfinite(rectanglel(f, a, b, n=1))
    assert np.isfinite(rectangler(f, a, b, n=1))
    # Simpson will adjust n to 2 internally, so should be finite
    assert np.isfinite(simpson(f, a, b, n=1))