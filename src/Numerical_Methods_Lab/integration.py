import numpy as np
def trapzoidal(f,a,b,n=10000):
    """
    This function implements the trapezoidal rule for numerical integration.
    :param f: The function to integrate
    :param a: The lower limit of integration
    :param b: The upper limit of integration
    :param n: The number of trapezoids (default is 10000)
    :return: The approximate value of the integral
    """
    x = np.linspace(a, b, n+1)  # x values for the trapezoids
    dx = x[1]-x[0]  # width of each trapezoid
    total = np.sum(f(x[:-1])*dx) + np.sum(f(x[1:])*dx)  # sum of areas of trapezoids
    return total/2  # return the average of the two sums

def rectanglel(f,a,b,n=10000):
    """
    This function implements the left rectangle rule for numerical integration.
    :param f: The function to integrate
    :param a: The lower limit of integration
    :param b: The upper limit of integration
    :param n: The number of rectangles (default is 10000)
    :return: The approximate value of the integral
    """
    x = np.linspace(a, b, n+1)  # x values for the rectangles
    dx = x[1]-x[0]  # width of each rectangle
    total = np.sum(f(x[:-1])*dx)  # sum of areas of rectangles
    return total  # return the total area

def rectangler(f,a,b,n=10000):
    """
    This function implements the right rectangle rule for numerical integration.
    :param f: The function to integrate
    :param a: The lower limit of integration
    :param b: The upper limit of integration
    :param n: The number of rectangles (default is 10000)
    :return: The approximate value of the integral
    """
    x = np.linspace(a, b, n+1)  # x values for the rectangles
    dx = x[1]-x[0]  # width of each rectangle
    total = np.sum(f(x[1:])*dx)  # sum of areas of rectangles
    return total  # return the total area

def simpson(f,a,b,n=10000):
    """
    This function implements Simpson's rule for numerical integration.
    :param f: The function to integrate
    :param a: The lower limit of integration
    :param b: The upper limit of integration
    :param n: The number of intervals (default is 10000, must be even)
    :return: The approximate value of the integral
    """
    if n % 2 == 1:
        n += 1  # make n even if it is odd
    x = np.linspace(a, b, n+1)  # x values for the intervals
    dx = x[1]-x[0] # width of each interval
    total = f(x[0]) + f(x[-1]) + 4*np.sum(f(x[1:-1:2])) + 2*np.sum(f(x[2:-1:2]))  # Simpson's rule formula
    return total * dx / 3  # return the approximate integral value

