import numpy as np
import pytest
from src.numerical_method_lab.linear_algebra import back_substitution, gaussian_elimination


# ----------------------------------------------------------------------
# Tests for back_substitution
# ----------------------------------------------------------------------
def test_back_substitution_2x2():
    A = np.array([[2.0, 1.0],
                  [0.0, 3.0]])
    b = np.array([5.0, 6.0])   # 2x + y = 5, 3y = 6 → y=2, x=1.5
    x = back_substitution(A, b)
    expected = np.array([1.5, 2.0])
    assert np.allclose(x, expected, rtol=1e-12)

def test_back_substitution_3x3():
    A = np.array([[1.0, 2.0, 3.0],
                  [0.0, 4.0, 5.0],
                  [0.0, 0.0, 6.0]])
    b = np.array([14.0, 23.0, 18.0])
    # 6z = 18 → z=3; 4y + 5*3 = 23 → y=2; x + 2*2 + 3*3 = 14 → x=1
    x = back_substitution(A, b)
    expected = np.array([1.0, 2.0, 3.0])
    assert np.allclose(x, expected, rtol=1e-12)

def test_back_substitution_zero_pivot_raises():
    A = np.array([[1.0, 2.0],
                  [0.0, 0.0]])   # zero on diagonal
    b = np.array([1.0, 1.0])
    with pytest.raises(ValueError):
        back_substitution(A, b)


# ----------------------------------------------------------------------
# Tests for gaussian_elimination
# ----------------------------------------------------------------------
def test_gaussian_elimination_2x2():
    A = np.array([[2.0, 1.0],
                  [1.0, 3.0]])
    b = np.array([5.0, 6.0])
    x = gaussian_elimination(A, b)
    expected = np.linalg.solve(A, b)
    assert np.allclose(x, expected, rtol=1e-12)

def test_gaussian_elimination_3x3():
    A = np.array([[3.0, 1.0, -1.0],
                  [2.0, -2.0, 4.0],
                  [-1.0, 0.5, -1.0]])
    b = np.array([1.0, -2.0, 0.0])
    x = gaussian_elimination(A, b)
    expected = np.linalg.solve(A, b)
    assert np.allclose(x, expected, rtol=1e-10)

def test_gaussian_elimination_4x4():
    A = np.array([[4.0, 1.0, 2.0, 0.5],
                  [3.0, 4.0, 0.0, 1.0],
                  [1.0, 0.0, 5.0, 2.0],
                  [0.0, 2.0, 1.0, 3.0]])
    b = np.array([8.5, 10.0, 13.0, 9.0])
    x = gaussian_elimination(A, b)
    expected = np.linalg.solve(A, b)
    assert np.allclose(x, expected, rtol=1e-10)

def test_gaussian_elimination_requires_pivoting():
    # Matrix where naive Gaussian elimination (without pivoting) would fail
    A = np.array([[0.0, 1.0],
                  [1.0, 0.0]])
    b = np.array([2.0, 1.0])   # solution: x=1, y=2
    x = gaussian_elimination(A, b)
    expected = np.array([1.0, 2.0])
    assert np.allclose(x, expected, rtol=1e-12)

def test_gaussian_elimination_ill_conditioned_pivoting():
    # A small pivot without pivoting would cause large errors, but with pivoting it's fine
    A = np.array([[1e-12, 1.0],
                  [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    x = gaussian_elimination(A, b)
    expected = np.linalg.solve(A, b)
    assert np.allclose(x, expected, rtol=1e-8)

def test_gaussian_elimination_singular_raises():
    A = np.array([[1.0, 2.0],
                  [2.0, 4.0]])   # linearly dependent rows
    b = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        gaussian_elimination(A, b)

def test_gaussian_elimination_mismatched_dimensions_raises():
    A = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    b = np.array([1.0])   # too short
    with pytest.raises(ValueError):
        gaussian_elimination(A, b)

def test_gaussian_elimination_non_square_raises():
    A = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])   # 2x3
    b = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        gaussian_elimination(A, b)

def test_gaussian_elimination_does_not_modify_original_input():
    A = np.array([[2.0, 1.0],
                  [1.0, 3.0]])
    b = np.array([5.0, 6.0])
    A_copy = A.copy()
    b_copy = b.copy()
    gaussian_elimination(A, b)
    # The function converts to arrays and modifies them internally, but should not affect originals
    assert np.array_equal(A, A_copy)
    assert np.array_equal(b, b_copy)