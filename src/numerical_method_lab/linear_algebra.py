import numpy as np


def back_substitution(A, b):
    """
    Solve an upper-triangular system Ax = b
    using back substitution.
    """

    n = len(b)

    # Create an array to store the solution
    x = np.zeros(n)

    # Start from the last equation
    for i in range(n - 1, -1, -1):

        # Start with the right-hand side
        total = b[i]

        # Subtract the terms we already know
        for j in range(i + 1, n):
            total -= A[i, j] * x[j]

        # Check for zero diagonal element
        if abs(A[i, i]) < 1e-12:
            raise ValueError("Zero pivot encountered during back substitution.")

        # Solve for x[i]
        x[i] = total / A[i, i]

    return x


def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination
    with partial pivoting and back substitution.
    """

    # Convert inputs to NumPy arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Make sure A is a square matrix
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    # Make sure b has the correct size
    if len(b) != A.shape[0]:
        raise ValueError("The dimensions of A and b do not match.")

    n = len(b)

    # -----------------------------------------
    # Gaussian Elimination
    # -----------------------------------------

    for i in range(n):

        # -------------------------------------
        # PARTIAL PIVOTING
        # -------------------------------------

        # Look at the current column from row i downward
        pivot_row = i + np.argmax(np.abs(A[i:, i]))

        # Check whether the largest value is effectively zero
        if abs(A[pivot_row, i]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular.")

        # Swap the current row with the pivot row
        if pivot_row != i:
            A[[i, pivot_row]] = A[[pivot_row, i]]
            b[[i, pivot_row]] = b[[pivot_row, i]]

        # -------------------------------------
        # ELIMINATION
        # -------------------------------------

        # Eliminate values below the pivot
        for j in range(i + 1, n):

            # Calculate elimination factor
            factor = A[j, i] / A[i, i]

            # Eliminate the value
            for k in range(i, n):
                A[j, k] -= factor * A[i, k]

            # Apply the same operation to b
            b[j] -= factor * b[i]

    # -----------------------------------------
    # BACK SUBSTITUTION
    # -----------------------------------------

    solution = back_substitution(A, b)

    return solution