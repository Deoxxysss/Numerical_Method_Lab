


def bisection_method(f, a, b, tol=1e-5, max_iter=100):
    """
    Bisection method for finding roots of a function.

    Parameters:
    f : function
        The function for which we are trying to find a root.
    a : float
        The start of the interval.
    b : float
        The end of the interval.
    tol : float
        The tolerance for stopping the algorithm.
    max_iter : int
        The maximum number of iterations.

    Returns:
    float
        The approximate root of the function.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        raise ValueError(
            "The function must have different signs at the endpoints."
        )

    for i in range(max_iter):

        c = (a + b) / 2
        fc = f(c)

        if abs(fc) < tol or (b - a) / 2 < tol:
            return c

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return (a + b) / 2

def derivative(f, x, h=1e-5):

    """

    Numerical derivative of a function using central difference.

    Parameters:

    f : function

        The function for which we are calculating the derivative.

    x : float

        The point at which to evaluate the derivative.

    h : float

        The step size for the finite difference.

    Returns:

    float

        The approximate derivative of the function at point x.

    """

    return (f(x + h) - f(x - h)) / (2 * h)


def Newton_raphson_method(f, x0, df=derivative, tol=1e-5, max_iter=100):

    """

    Newton-Raphson Method

    =====================

    What it is:

        The Newton-Raphson method is an iterative numerical technique used to find

        roots of a real-valued function f(x) = 0.

        If x_n is the current estimate, then the next estimate is:

            x_{n+1} = x_n - f(x_n) / f'(x_n)

        where f'(x) is the derivative of f(x).

    How it works:

        - Start with an initial guess x0.

        - Compute the tangent line to the curve at x0.

        - Find where that tangent line crosses the x-axis.

        - Repeat until the change is very small or the function value is close to zero.

    Why use it:

        - Very fast when the initial guess is close to the actual root.

        - It usually converges quadratically near simple roots.

        - Works well for smooth functions with known derivatives.

    Using NumPy:

        NumPy helps with array operations, mathematical calculations, and plotting

        when needed. This example uses only standard NumPy math functions.

    Parameters:

    f : function

        The function for which we are trying to find a root.

    x0 : float

        The initial guess for the root.

    df : function, optional

        The derivative of the function f. If not provided, it will be calculated numerically.

    tol : float, optional

        The tolerance for stopping the algorithm.

    max_iter : int, optional

        The maximum number of iterations.

    """

    x_n = x0

    for i in range(max_iter):

        f_xn = f(x_n)

        df_xn = df(f, x_n)

        if abs(df_xn) < 1e-12:

            raise ValueError("Derivative is too small; no convergence.")

        x_n1 = x_n - f_xn / df_xn

        if abs(x_n1 - x_n) < tol:
            return x_n1

        if abs(f(x_n1)) < tol:
            return x_n1

        x_n = x_n1

    raise ValueError("Maximum iterations reached without convergence.")

