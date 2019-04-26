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
            self._grad = partial(_fd_grad, self.cost, h=self._grad_step)
        elif grad is None and grad_step is not None:
            self._grad_step = grad_step
            self._grad = partial(_fd_grad, self.cost, h=self._grad_step)
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
                return np.array([eq(x) for eq in self._eq_const])
            else:
                return np.array([eq for eq in self._eq_const])
        else:
            return None

    def ineq_const(self, x=None):
        if self._ineq_const is not None:
            if x is not None:
                return np.array([ineq(x) for ineq in self._ineq_const])
            else:
                return np.array([ineq for ineq in self._ineq_const])
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


def augmented_lagrange(p, x0, tol=1e-6, tol_const=1e-6, sigma_max=1e12):
    """Constrained optimization algorithm using augmented Lagrange method"""

    def phi(p, lmb, sgm, x):
        cost = p.cost(x)

        n_e = p.num_eq_const()
        n_i = p.num_ineq_const()
        n_c = n_e + n_i

        lmb_e = lmb[0:n_e, :]
        lmb_i = lmb[n_e:n_c, :]
        sgm_e = sgm[0:n_e, :]
        sgm_i = sgm[n_e:n_c, :]

        if p.eq_const() is not None:
            c_e = p.eq_const(x)
            cost = cost - sum(lmb_e * c_e) + 0.5 * sum(sgm_e * c_e**2)

        if p.ineq_const() is not None:
            c_i = p.ineq_const(x)
            p_i = np.array([-lmb_i[i] * c_i[i] + 0.5 * sgm_i[i] * c_i[i]**2 \
                            if c_i[i] <= lmb_i[i] / sgm_i[i] \
                            else -0.5 * lmb_i[i]**2 / sgm_i[i] \
                            for i in range(0, n_i)])
            cost = cost + sum(p_i)

        return cost

    n_e = p.num_eq_const()
    n_i = p.num_ineq_const()
    n_c = n_e + n_i

    lmb = np.zeros((n_c, 1))
    sgm = np.ones((n_c, 1))

    x = x0
    c = 1e12 * np.ones((n_c, 1))

    while np.linalg.norm(c) > tol_const:
        up = Problem(partial(phi, p, lmb, sgm))
        x = steepest_descent(up, x0, tol=tol)

        c_prv = c
        c_e = p.eq_const(x)
        c_i = p.ineq_const(x)

        if c_e is not None and c_i is not None:
            c = np.concatenate((c_e, c_i), axis=0)
        elif c_e is not None:
            c = c_e
        elif c_i is not None:
            c = c_i

        if any(sgm >= sigma_max):
            break

        if np.linalg.norm(c, np.inf) > 0.25 * np.linalg.norm(c_prv, np.inf):
            for i in range(0, n_c):
                if np.abs(c[i]) > 0.25 * np.linalg.norm(c_prv, np.inf):
                    sgm[i] *= 10
            continue

        lmb = lmb - (sgm * c)

    return x


def lagrange_newton(p, x0, tol=1e-6):
    """Constrained optimization algorithm using Lagrange-Newton method"""
    return 0


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


def _fd_grad(f, x, h=1e-8):
    """Finite difference approximation of the gradient"""

    dim = np.max(np.shape(x))
    grad_gen = ((f(x + h * np.eye(dim)[:, [i]]) - f(x)) / h
               for i in range(0, dim))
    grad = np.expand_dims(np.fromiter(grad_gen, np.float64), axis=0)
    return grad


def _fd_hessian(f, x, h=1e-8):
    """Finite different approximation of the Hessian"""

    dim = np.max(np.shape(x))
    I = np.eye(dim)
    H = np.zeros((dim, dim))

    for i in range(0, dim):
        for j in range(0, dim):
            H[i, j] =  (f(x + h * I[:, [i]] + h * I[:, [j]]) \
                      - f(x + h * I[:, [i]]) - f(x + h * I[:, [j]]) \
                      + f(x)) / h**2

    return 0.5 * (H + H.T)


def _cone_condition(p, x, s, theta=89):
    """Check the cone condition at a point"""

    gx = p.grad(x)
    cos_phi = (-gx @ s) / (np.linalg.norm(s) * np.linalg.norm(gx))
    cos_theta = np.cos(theta * 2 * np.pi / 360)

    return (cos_phi > cos_theta)
