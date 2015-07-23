"""Trivial example: minimize x**2 from any start value"""

import lbfgs
import sys


from scipy.optimize import minimize, rosen, rosen_der
import numpy as np

x0 = np.array([1.3, 0.7])

def f(x, g):
    g[:] = rosen_der(x)
    print "one call"
    return rosen(x)


def progress(x, g, f_x, xnorm, gnorm, step, k, ls):
    """Report optimization progress."""
    #print("x = %8.2g     f(x) = %8.2g     f'(x) = %8.2g" % (x, f_x, g))
    pass


print("Minimum found", lbfgs.fmin_lbfgs(f, x0, progress))
