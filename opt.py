import numpy as np
from functools import partial


class Problem:
    """Optimization problem"""

    def __init__(self, cost, grad=None, grad_step=None,
                 eq_const=None, ineq_const=None):
        self._cost = cost
        # Check presence of gradient function
        if grad is None and grad_step is None:
            self._grad_step = 1e-8
            self._grad = partial(_fd_grad, self, h=self._grad_step)
        elif grad is None and grad_step is not None:
            self._grad_step = grad_step
            self._grad = partial(_fd_grad, self, h=self._grad_step)
        elif grad is not None:
            self._grad_step = None
            self._grad = grad
        # Check presence of constraints
        if eq_const is not None:
            self._eq_const = eq_const
        else:
            self._eq_const = None
        if ineq_const is not None:
            self._ineq_const = ineq_const
        else:
            self._ineq_const = None

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

    def eq_const(self, x=None):
        if self._eq_const is not None:
            if x is not None:
                return np.array([[eq(x)] for eq in self._eq_const])
            else:
                return np.array([[eq] for eq in self._eq_const])
        else:
            return None

    def ineq_const(self, x=None):
        if self._ineq_const is not None:
            if x is not None:
                return np.array([[ineq(x)] for ineq in self._ineq_const])
            else:
                return np.array([[ineq] for ineq in self._ineq_const])
        else:
            return None

    def num_eq_const(self):
        if self._eq_const is not None:
            return np.max(np.shape(self._eq_const))
        else:
            return 0

    def num_ineq_const(self):
        if self._ineq_const is not None:
            return np.max(np.shape(self._ineq_const))
        else:
            return 0


def steepest_descent(p, x, tol=1e-6, max_iter=999):
    """Steepest descent optimization algorithm"""

    i = 0
    while np.linalg.norm(p.grad(x)) > tol:
        if i > max_iter:
            break
        s = -p.grad(x).T
        w = _step_size(p, x, s)
        x = x + w * s
        i += 1

    return x


def conjugate_gradient(p, x, tol=1e-6, rst_iter=99, max_iter=999):
    """Conjugate gradient optimization algorithm"""

    i = 0
    s = -p.grad(x).T
    while np.linalg.norm(p.grad(x)) > tol:
        if i > rst_iter:
            i = 0
            s = -p.grad(x).T
        elif not _cone_condition(p, x, s):
            i = 0
            s = -p.grad(x).T
        elif i > max_iter:
            break
        w = _step_size(p, x, s)
        x_prv = x
        x = x_prv + w * s
        beta = ((p.grad(x) - p.grad(x_prv)) @ p.grad(x).T) \
            / (p.grad(x_prv) @ p.grad(x_prv).T)
        s = -p.grad(x).T + beta * s
        i += 1

    return x


def secant(p, x, tol=1e-6, H=None, rst_iter=99, max_iter=999):
    """Secant optimization algorithm"""

    if H is None:
        H = np.eye(np.max(np.shape(x)))

    i = 0
    while np.linalg.norm(p.grad(x)) > tol:
        s = -H @ p.grad(x).T
        if i > rst_iter:
            i = 0
            s = -p.grad(x).T
        elif not _cone_condition(p, x, s):
            i = 0
            s = -p.grad(x).T
        elif i > max_iter:
            break
        w = _step_size(p, x, s)
        x_prv = x
        x = x_prv + w * s
        # Davidon-Fletcher-Powell (DFP) Algorithm
        dx = x - x_prv
        dg = p.grad(x) - p.grad(x_prv)
        H = H + (dx @ dx.T) / (dx.T @ dg.T) \
            - ((H @ dg.T) @ (H @ dg.T).T) / (dg @ H @ dg.T)
        i += 1

    return x


def penalty_function(p, x0, tol=1e-6, tol_const=1e-4, sigma_max=1e6):
    """Constrained optimization algorithm using penalty function"""

    def phi(p, sigma, x):
        cost = p.cost(x)
        if p.eq_const() is not None:
            cost = cost + 0.5 * sigma * np.linalg.norm(p.eq_const(x))**2
        if p.ineq_const() is not None:
            ineq_x = p.ineq_const(x)
            c = np.minimum(np.zeros(np.shape(ineq_x)), ineq_x)
            cost = cost + 0.5 * sigma * np.linalg.norm(c)**2
        return cost

    def cost_norm(x):
        cost = 0
        if p.eq_const() is not None:
            cost = cost + np.linalg.norm(p.eq_const(x))**2
        if p.ineq_const() is not None:
            ineq_x = p.ineq_const(x)
            c = np.minimum(np.zeros(np.shape(ineq_x)), ineq_x)
            cost = cost + np.linalg.norm(c)**2
        return np.sqrt(cost)

    sigma = 1
    x = x0

    while cost_norm(x) > tol_const:
        up = Problem(partial(phi, p, sigma))
        x = steepest_descent(up, x0, tol=tol)
        if sigma >= sigma_max:
            break
        sigma *= 10

    return x


def barrier_function(p, x0, tol=1e-6, tol_const=1e-4, sigma_max=1e6, r_min=1e-6, mode='inv'):
    """Constrained optimization algorithm using barrier function"""

    def phi(p, sigma, r, x):
        cost = p.cost(x)
        if p.eq_const() is not None:
            cost = cost + 0.5 * sigma * np.linalg.norm(p.eq_const(x))**2
        if p.ineq_const() is not None:
            ineq_x = p.ineq_const(x)
            if mode == 'log':
                cost = cost - r * np.sum(np.log(ineq_x))
            else:
                cost = cost + r * np.sum(np.reciprocal(ineq_x))
        return cost

    def cost_norm(x):
        cost = 0
        if p.eq_const() is not None:
            cost = cost + np.linalg.norm(p.eq_const(x))**2
        if p.ineq_const() is not None:
            ineq_x = p.ineq_const(x)
            c = np.minimum(np.zeros(np.shape(ineq_x)), ineq_x)
            cost = cost + np.linalg.norm(c)**2
        return np.sqrt(cost)

    sigma = 1
    r = 1
    x = x0

    while cost_norm(x) > tol_const:
        up = Problem(partial(phi, p, sigma, r))
        x = steepest_descent(up, x0, tol=tol)
        if sigma >= sigma_max or r <= r_min:
            break
        sigma *= 10
        r *= 0.1

    return x


# def augmented_lagrange(p, x0, tol=1e-6):
#     """Constrained optimization algorithm using augmented Lagrange method"""

#     def phi(p, lmb, sgm, x):
#         S = np.diagflat(sgm)
#         cost = p.cost(x)
#         if p.eq_const() is not None:
#             eq_x = p.eq_const(x)
#             n_eq = np.max(np.shape(eq_x))
#             cost = cost - lmb.T @ eq_x + 0.5 * (eq_x.T @ S @ eq_x)
#         if p.ineq_const() is not None:
#             ineq_x = p.ineq_const(x)
#             n_ineq = np.max(np.shape(ineq_x))
#             cost = cost + 

#         return cost


def _step_size(p, x, s, gamma=1.5, mu=0.8):
    """Armijo algorithm for computing step size"""

    w = 1  # Default step size

    k_g = 0  # Power of gamma
    k_m = 0  # Power of mu

    # Precompute cost and gradient to save time
    vx = p.cost(x)
    gx_s = p.grad(x) @ s

    def v_bar(w):
        return vx + 0.5 * w * gx_s

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
    grad = np.expand_dims(np.fromiter(grad_gen, np.float64), axis=0)
    return grad


def _cone_condition(p, x, s, theta=89):
    """Check the cone condition at a point"""

    gx = p.grad(x)
    cos_phi = (-gx @ s) / (np.linalg.norm(s) * np.linalg.norm(gx))
    cos_theta = np.cos(theta * 2 * np.pi / 360)

    return (cos_phi > cos_theta)
