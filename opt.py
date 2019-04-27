import numpy as np
from functools import partial


class Problem:
    """Optimization problem"""

    def __init__(self, cost, grad=None, grad_step=None,
                 eq_const=None, ineq_const=None):
        """Constructor for Problem object

        Creates a Problem object for use with optimization algorithms.

        Parameters
        ----------
        cost : function whose input is 2D numpy column array
            Objective function of problem
        grad : function whose input is 2D numpy column array
            Gradient function of problem. If not specified, finite difference
            is used
        grad_step : float
            Step size for finite difference gradient. If not specified, (and
            no analytic gradient is given) 1e-8 is used
        eq_const : list of functions
            List of functions that form equality constraints for problem
        ineq_const : list of functions
            List of functions that form inequality constraints for problem

        """
        self._cost = cost
        # Check presence of gradient function
        if grad is None and grad_step is None:
            # No gradient function, default step size
            self._grad_step = 1e-8
            self._grad = partial(_fd_grad, self.cost, h=self._grad_step)
        elif grad is None and grad_step is not None:
            # No gradient function, specified step size
            self._grad_step = grad_step
            self._grad = partial(_fd_grad, self.cost, h=self._grad_step)
        elif grad is not None:
            # Gradient given, no need for step size
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
        """Cost of Problem

        If x is given, returns cost at x. Otherwise returns cost function.

        Parameters
        ----------
        x : 2D numpy column array
            Point at which to evaluate cost

        Returns
        -------
        float or function
            If x was given, returns cost at x. Otherwise returns cost function.
        """
        if x is not None:
            return self._cost(x)
        else:
            return self._cost

    def grad(self, x=None):
        """Gradient of Problem

        If x is given, returns gradient at x. Otherwise returns gradient 
        function.

        Parameters
        ----------
        x : 2D numpy column array
            Point at which to evaluate gradient

        Returns
        -------
        2D numpy row array or function
            If x was given, returns gradient at x. Otherwise returns gradient
            function.
        """
        if x is not None:
            return self._grad(x)
        else:
            return self._grad

    def eq_const(self, x=None):
        """ Equality constraints of Problem

        If x is given, returns column array of constraints evaluated
        at x. Otherwise returns column array of functions.

        Parameters
        ----------
        x : 2D numpy column array
            Point at which to evaluate constraints

        Returns
        -------
        2D numpy column array of floats or functions
            If x was given, returns column array of costs at x. Otherwise
            returns column array of functions
        """
        if self._eq_const is not None:
            if x is not None:
                return np.array([[eq(x)] for eq in self._eq_const])
            else:
                return np.array([eq for eq in self._eq_const])
        else:
            return None

    def ineq_const(self, x=None):
        """ Inequality constraints of Problem

        If x is given, returns column array of constraints evaluated
        at x. Otherwise returns column array of functions.

        Parameters
        ----------
        x : 2D numpy column array
            Point at which to evaluate constraints

        Returns
        -------
        2D numpy column array of floats or functions
            If x was given, returns column array of costs at x. Otherwise
            returns column array of functions
        """
        if self._ineq_const is not None:
            if x is not None:
                return np.array([[ineq(x)] for ineq in self._ineq_const])
            else:
                return np.array([ineq for ineq in self._ineq_const])
        else:
            return None

    def num_eq_const(self):
        """Returns number of equality constraints"""
        if self._eq_const is not None:
            return np.max(np.shape(self._eq_const))
        else:
            return 0

    def num_ineq_const(self):
        """Returns number of inequality constraints"""
        if self._ineq_const is not None:
            return np.max(np.shape(self._ineq_const))
        else:
            return 0


def steepest_descent(p, x, tol=1e-6, max_iter=999, hist=False):
    """Steepest descent optimization algorithm

    Parameters
    ----------
    p : Problem
        Problem to minimize
    x : 2D numpy column array of floats
        Initial guess at minimum
    tol : float
        When norm of gradient goes below this value, iteration stops
    max_iter : int
        Absolute maximum number of iterations before giving up and returning x
    hist : bool
        If True, returns array with value of x at every iteration. If False,
        just returns last x value.o

    Returns
    -------
    2D or 3D numpy column array of floats
        If hist is False, returns 2D numpy colmn array containing minimizing
        x of problem. Otherwise returns a 3D numpy array containing every
        value of x along the way.

    """

    i = 0
    x_hist = []
    while np.linalg.norm(p.grad(x)) > tol:
        if i > max_iter:
            break
        s = -p.grad(x).T
        w = _step_size(p, x, s)
        x_hist.append(x)
        x = x + w * s
        i += 1

    return x if not hist else np.array(x_hist)


def conjugate_gradient(p, x, tol=1e-6, rst_iter=99, max_iter=999, hist=False):
    """Conjugate gradient optimization algorithm

    Parameters
    ----------
    p : Problem
        Problem to minimize
    x : 2D numpy column array of floats
        Initial guess at minimum
    tol : float
        When norm of gradient goes below this value, iteration stops
    rst_iter : int
        Number of iterations to go before stopping iteration, resetting search
        direction to gradient, and restarting
    max_iter : int
        Absolute maximum number of iterations before giving up and returning x
    hist : bool
        If True, returns array with value of x at every iteration. If False,
        just returns last x value.o

    Returns
    -------
    2D or 3D numpy column array of floats
        If hist is False, returns 2D numpy colmn array containing minimizing
        x of problem. Otherwise returns a 3D numpy array containing every
        value of x along the way.

    """

    i = 0
    x_hist = []
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
        x_hist.append(x)
        x = x_prv + w * s
        beta = ((p.grad(x) - p.grad(x_prv)) @ p.grad(x).T) \
            / (p.grad(x_prv) @ p.grad(x_prv).T)
        s = -p.grad(x).T + beta * s
        i += 1

    return x if not hist else np.array(x_hist)


def secant(p, x, tol=1e-6, H=None, rst_iter=99, max_iter=999, hist=False):
    """Secant optimization algorithm

    Parameters
    ----------
    p : Problem
        Problem to minimize
    x : 2D numpy column array of floats
        Initial guess at minimum
    tol : float
        When norm of gradient goes below this value, iteration stops
    H : 2D numpy matrix
        Initial guess at Hessian if you have one. If not, it's set to identity
    rst_iter : int
        Number of iterations to go before stopping iteration, resetting search
        direction to gradient, and restarting
    max_iter : int
        Absolute maximum number of iterations before giving up and returning x
    hist : bool
        If True, returns array with value of x at every iteration. If False,
        just returns last x value.o

    Returns
    -------
    2D or 3D numpy column array of floats
        If hist is False, returns 2D numpy colmn array containing minimizing
        x of problem. Otherwise returns a 3D numpy array containing every
        value of x along the way.

    """

    if H is None:
        H = np.eye(np.max(np.shape(x)))

    i = 0
    x_hist = []
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
        x_hist.append(x)
        x = x_prv + w * s
        # Davidon-Fletcher-Powell (DFP) Algorithm
        dx = x - x_prv
        dg = p.grad(x) - p.grad(x_prv)
        H = H + (dx @ dx.T) / (dx.T @ dg.T) \
            - ((H @ dg.T) @ (H @ dg.T).T) / (dg @ H @ dg.T)
        i += 1

    return x if not hist else np.array(x_hist)


def penalty_function(p, x0, tol=1e-6, tol_const=1e-4, sigma_max=1e6, hist=False):
    """Constrained optimization algorithm using penalty function

    Parameters
    ----------
    p : Problem
        Problem to minimize (needs constraints)
    x0 : 2D numpy column array of floats
        Initial guess at minimum
    tol : float
        When norm of gradient goes below this value, unconstrained iteration
        stops
    tol_const : float
        When norm of costs goes below this value, iteration stops
    sigma_max : float
        Maximum value of sigma before iteration stops
    hist : bool
        If True, returns array with value of x at every iteration. If False,
        just returns last x value.o

    Returns
    -------
    2D or 3D numpy column array of floats
        If hist is False, returns 2D numpy colmn array containing minimizing
        x of problem. Otherwise returns a 3D numpy array containing every
        value of x along the way.

    """

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
    x_hist = []

    while cost_norm(x) > tol_const:
        up = Problem(partial(phi, p, sigma))
        x_hist.append(x)
        x = steepest_descent(up, x0, tol=tol)
        if sigma >= sigma_max:
            break
        sigma *= 10

    return x if not hist else np.array(x_hist)


def barrier_function(p, x0, tol=1e-6, tol_const=1e-4, sigma_max=1e6,
                     r_min=1e-6, mode='inv', hist=False):
    """Constrained optimization algorithm using barrier function

    Logarithmic barrier function does not seem to work properly. Use inverse.

    Parameters
    ----------
    p : Problem
        Problem to minimize (needs constraints)
    x0 : 2D numpy column array of floats
        Initial guess at minimum
    tol : float
        When norm of gradient goes below this value, unconstrained iteration
        stops
    tol_const : float
        When norm of costs goes below this value, iteration stops
    sigma_max : float
        Maximum value of sigma before iteration stops
    r_min : float
        Minimum value of r before iteration stops
    mode: string
        Either 'inv' for inverse barrier function or 'log' for logarithmic
        barrier function
    hist : bool
        If True, returns array with value of x at every iteration. If False,
        just returns last x value.o

    Returns
    -------
    2D or 3D numpy column array of floats
        If hist is False, returns 2D numpy colmn array containing minimizing
        x of problem. Otherwise returns a 3D numpy array containing every
        value of x along the way.

    """

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
    x_hist = []

    while cost_norm(x) > tol_const:
        up = Problem(partial(phi, p, sigma, r))
        x_hist.append(x)
        x = steepest_descent(up, x0, tol=tol)
        if sigma >= sigma_max or r <= r_min:
            break
        sigma *= 10
        r *= 0.1

    return x if not hist else np.array(x_hist)


def augmented_lagrange(p, x0, tol=1e-6, tol_const=1e-6, sigma_max=1e12, hist=False):
    """Constrained optimization algorithm using augmented Lagrange method

    Parameters
    ----------
    p : Problem
        Problem to minimize (needs constraints)
    x0 : 2D numpy column array of floats
        Initial guess at minimum
    tol : float
        When norm of gradient goes below this value, unconstrained iteration
        stops
    tol_const : float
        When norm of costs goes below this value, iteration stops
    sigma_max : float
        Maximum value of sigma before iteration stops
    hist : bool
        If True, returns array with value of x at every iteration. If False,
        just returns last x value.o

    Returns
    -------
    2D or 3D numpy column array of floats
        If hist is False, returns 2D numpy colmn array containing minimizing
        x of problem. Otherwise returns a 3D numpy array containing every
        value of x along the way.

    """

    def phi(p, lmb, sgm, x):
        """Unconstrained problem"""
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

    x_hist = []

    n_e = p.num_eq_const()
    n_i = p.num_ineq_const()
    n_c = n_e + n_i

    lmb = np.zeros((n_c, 1))
    sgm = np.ones((n_c, 1))

    x = x0
    c = 1e12 * np.ones((n_c, 1))

    while np.linalg.norm(c) > tol_const:
        # Create new problem to solve, but unconstrained
        up = Problem(partial(phi, p, lmb, sgm))
        x_hist.append(x)
        x = steepest_descent(up, x0, tol=tol)

        # Concatenate costs
        c_prv = c
        c_e = p.eq_const(x)
        c_i = p.ineq_const(x)
        if c_e is not None and c_i is not None:
            c = np.concatenate((c_e, c_i), axis=0)
        elif c_e is not None:
            c = c_e
        elif c_i is not None:
            c = c_i

        # Make sure sigma is not too big
        if any(sgm >= sigma_max):
            break

        # Update sigma
        if np.linalg.norm(c, np.inf) > 0.25 * np.linalg.norm(c_prv, np.inf):
            for i in range(0, n_c):
                if np.abs(c[i]) > 0.25 * np.linalg.norm(c_prv, np.inf):
                    sgm[i] *= 10
            continue

        lmb = lmb - (sgm * c)

    return x if not hist else np.array(x_hist)


def lagrange_newton(p, x0, tol=1e-6, hist=False):
    """Constrained optimization algorithm using Lagrange-Newton method

    Parameters
    ----------
    p : Problem
        Problem to minimize (needs constraints)
    x0 : 2D numpy column array of floats
        Initial guess at minimum
    tol : float
        When norm of gradient goes below this value, unconstrained iteration
        stops
    hist : bool
        If True, returns array with value of x at every iteration. If False,
        just returns last x value.o

    Returns
    -------
    2D or 3D numpy column array of floats
        If hist is False, returns 2D numpy colmn array containing minimizing
        x of problem. Otherwise returns a 3D numpy array containing every
        value of x along the way.

    """

    x_hist = []

    n_e = p.num_eq_const()
    n_i = p.num_ineq_const()
    n_c = n_e + n_i

    def W(x, lmb):
        lmb_e = lmb[0:n_e, :]
        lmb_i = lmb[n_e:n_c, :]
        hess_f = _fd_hessian(p.cost, x)
        hess_c_e = - np.sum([lmb_e[i] * _fd_hessian(p.eq_const()[i], x)
            for i in range(0, n_e)])
        hess_c_i = - np.sum([lmb_i[i] * _fd_hessian(p.ineq_const()[i], x)
            for i in range(0, n_i)])
        hess = hess_f + hess_c_e + hess_c_i
        return hess

    def A(x):
        grad_e = np.array([np.squeeze(_fd_grad(p.eq_const()[i], x))
                for i in range (0, n_e)])
        grad_i = np.array([np.squeeze(_fd_grad(p.ineq_const()[i], x))
                for i in range (0, n_i)])
        if n_e != 0 and n_i != 0:
            grad = np.concatenate((grad_e, grad_i), axis=0)
        elif n_e != 0:
            grad = grad_e
        elif n_i != 0:
            grad = grad_i
        return grad

    x = x0
    lmb = np.zeros((n_c, 1))

    # Concatenate costs
    c_e = p.eq_const(x)
    c_i = p.ineq_const(x)
    if c_e is not None and c_i is not None:
        c = np.concatenate((c_e, c_i), axis=0)
    elif c_e is not None:
        c = c_e
    elif c_i is not None:
        c = c_i

    delta_x = 1e12

    while delta_x  > tol:

        # Compute KKT matrix
        KKT = np.block([
            [W(x, lmb), -A(x).T],
            [-A(x), np.zeros((n_c, n_c))]
        ])

        # Compute gradient augmented with constraints
        if n_e != 0 and n_i != 0:
            f = np.block([
                [-_fd_grad(p.cost, x).T + A(x).T @ lmb],
                [p.eq_const(x)],
                [p.ineq_const(x)]
            ])
        elif n_e != 0:
            f = np.block([
                [-_fd_grad(p.cost, x).T + A(x).T @ lmb],
                [p.eq_const(x)]
            ])
        elif n_i != 0:
            f = np.block([
                [-_fd_grad(p.cost, x).T + A(x).T @ lmb],
                [p.ineq_const(x)]
            ])

        x_prv = x
        # Invert KKT matrix to get x and lambda increments
        X = np.linalg.solve(KKT, f)
        dim = np.max(np.shape(x))
        x_hist.append(x)
        # Apply x and lambda increments
        x = x + X[:dim, :]
        lmb = lmb + X[dim:, :]

        c_e = p.eq_const(x)
        c_i = p.ineq_const(x)

        if c_e is not None and c_i is not None:
            c = np.concatenate((c_e, c_i), axis=0)
        elif c_e is not None:
            c = c_e
        elif c_i is not None:
            c = c_i

        # Check distance from previous x
        delta_x = np.linalg.norm(x - x_prv)

    return x if not hist else np.array(x_hist)


def _step_size(p, x, s, gamma=1.5, mu=0.8):
    """Armijo algorithm for computing step size

    Parameters
    ----------
    gamma : float
        Parameter for increasing step size
    mu : float
        Parameter for decreasing step size

    Returns
    -------
    float
        Step size
    """

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
    """Finite difference approximation of the gradient

    Parameters
    ----------
    f : function of x (2D numpy column array)
        Function whose gradient you want to evaluate
    x : 2D numpy column array
        Point where you want to evaluate the gradient
    h : float
        Step size

    Returns
    -------
    2D numpy row array
        Gradient (which is a row vector)
    """

    dim = np.max(np.shape(x))
    grad_gen = ((f(x + h * np.eye(dim)[:, [i]]) - f(x)) / h
               for i in range(0, dim))
    grad = np.expand_dims(np.fromiter(grad_gen, np.float64), axis=0)
    return grad


def _fd_hessian(f, x, h=1e-8):
    """Finite different approximation of the Hessian

    Parameters
    ----------
    f : function of x (2D numpy column array)
        Function whose Hessian you want to evaluate
    x : 2D numpy column array
        Point where you want to evaluate the Hessian
    h : float
        Step size

    Returns
    -------
    2D numpy matrixx
        Hessian matrix
    """

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
    """Check the cone condition at a point

    Parameters
    ----------
    p : Problem
        Problem you're minimizing
    x : 2D numpy column array
        Point where you want to check cone condition
    s : 2D numpy column matrix
        Search direction
    theta : float
        Acceptable angle with gradient (degrees)

    Returns
    -------
    bool
        Returns True if s is within theta degrees of the gradient at x
    """

    gx = p.grad(x)
    cos_phi = (-gx @ s) / (np.linalg.norm(s) * np.linalg.norm(gx))
    cos_theta = np.cos(theta * 2 * np.pi / 360)

    return (cos_phi > cos_theta)
