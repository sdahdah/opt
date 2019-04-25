import numpy as np
from functools import partial


class Problem:
    """Optimization problem"""

    def __init__(self, cost, grad=None, grad_step=None):
        self._cost = cost
        if grad is None and grad_step is None:
            self._grad_step = 1e-8
            self._grad = partial(_fd_grad, self, h=self._grad_step)
        elif grad is None and grad_step is not None:
            self._grad_step = grad_step
            self._grad = partial(_fd_grad, self, h=self._grad_step)
        elif grad is not None:
            self._grad_step = None
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
        beta = ((p.grad(x) - p.grad(x_prv)) @ p.grad(x).T) \
            / (p.grad(x_prv) @ p.grad(x_prv).T)
        s = -p.grad(x).T + beta * s

    return x


def secant(p, x, tolerance=1e-6, H=None):
    """Secant optimization algorithm"""

    if H is None:
        H = np.eye(np.max(np.shape(x)))

    while np.linalg.norm(p.grad(x)) > tolerance:
        s = -H @ p.grad(x).T
        w = _step_size(p, x, s)
        x_prv = x
        x = x_prv + w * s
        # Davidon-Fletcher-Powell (DFP) Algorithm
        dx = x - x_prv
        dg = p.grad(x) - p.grad(x_prv)
        H = H + (dx @ dx.T) / (dx.T @ dg.T) \
            - ((H @ dg.T) @ (H @ dg.T).T) / (dg @ H @ dg.T)

    return x


def _step_size(p, x, s, gamma=1.5, mu=0.8):
    """Armijo algorithm for computing step size"""

    w = 1  # Default step size

    k_g = 0  # Power of gamma
    k_m = 0  # Power of mu

    # Precompute cost and gradient to save time
    vx = p.cost(x)
    gx = p.grad(x)

    def v_bar(w):
        return vx + 0.5 * w * gx @ s

    while p.cost(x + gamma**k_g * s) < v_bar(gamma**k_g):
        k_g += 1
        w = gamma**k_g

    while p.cost(x + mu**k_m * gamma**k_g * s) > v_bar(mu**k_m * gamma**k_g):
        k_m += 1
        w = mu**k_m * gamma**k_g

    return w


def _fd_grad(p, x, h=1e-6):
    """Finite difference approximation of the gradient"""

    dim = np.max(np.shape(x))
    grad_gen = ((p.cost(x + h * np.eye(dim)[:, [i]]) - p.cost(x)) / h
               for i in range(0, dim))
    grad = np.expand_dims(np.fromiter(grad_gen, np.float), axis=0)
    return grad
