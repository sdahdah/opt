import numpy as np


class Problem:
    """Optimization problem"""

    def __init__(self, cost, grad):
        self._cost = cost
        self._grad = grad

    def cost(self, x=None):
        if x is not None:
            return self._cost(x)
        else:
            return self._cost

    def grad(self, x=None):
        if x is not None:
            return self._grad(x)
        else:
            return self._grad


def steepest_descent(p, x, tolerance=1e-6):
    """Steepest descent optimization algorithm"""

    while np.linalg.norm(p.grad(x)) > tolerance:
        s = -p.grad(x).T
        w = _step_size(p, x, s)
        x = x + w * s

    return x


def conjugate_gradient(p, x, tolerance=1e-6):
    """Conjugate gradient optimization algorithm"""

    s = -p.grad(x).T
    while np.linalg.norm(p.grad(x)) > tolerance:
        w = _step_size(p, x, s)
        x_prv = x
        x = x_prv + w * s
        beta = (p.grad(x) - p.grad(x_prv)) @ p.grad(x).T \
             / p.grad(x_prv) @ p.grad(x_prv).T
        s = -p.grad(x).T + beta * s

    return x


def _step_size(p, x, s, gamma=1.5, mu=0.8):
    """Armijo algorithm for computing step size"""

    k_g = 0 # Power of gamma
    k_m = 0 # Power of mu

    def v_bar(w):
        return p.cost(x) + 0.5 * w * p.grad(x) @ s

    while p.cost(x + gamma**k_g * s) < v_bar(gamma**k_g):
        k_g += 1
        w = gamma**k_g

    while p.cost(x + mu**k_m * gamma**k_g * s) > v_bar(mu**k_m * gamma**k_g):
        k_m += 1
        w = mu**k_m * gamma**k_g

    return w
