import unittest
import numpy as np
import opt

np.set_printoptions(precision=20)

class TestProblemAGrad(unittest.TestCase):

    def setUp(self):
        a = 5
        b = np.array([[1], [4], [5], [4], [2], [1]])
        C = 2 * np.array([[9,  1,  7,  5,  4,  7],
                          [1, 11,  4,  2,  7,  5],
                          [7,  4, 13,  5,  0,  7],
                          [5,  2,  5, 17,  1,  9],
                          [4,  7,  0,  1, 21, 15],
                          [7,  5,  7,  9, 15, 27]])

        v = lambda x : a + b.T @ x + 0.5 * x.T @ C @ x
        del_v = lambda x : b.T + x.T @ C
        self.p = opt.Problem(v, del_v)
        self.x_opt = -np.linalg.solve(C, b)

    def test_sd(self):
        x = np.array([[0], [0], [0], [0], [0], [0]])
        x_opt = opt.steepest_descent(self.p, x)
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-6)

    def test_cg(self):
        x = np.array([[0], [0], [0], [0], [0], [0]])
        x_opt = opt.conjugate_gradient(self.p, x)
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-6)

    def test_sec(self):
        x = np.array([[0], [0], [0], [0], [0], [0]])
        x_opt = opt.secant(self.p, x)
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-6)


class TestProblemA(unittest.TestCase):

    def setUp(self):
        a = 5
        b = np.array([[1], [4], [5], [4], [2], [1]])
        C = 2 * np.array([[9,  1,  7,  5,  4,  7],
                          [1, 11,  4,  2,  7,  5],
                          [7,  4, 13,  5,  0,  7],
                          [5,  2,  5, 17,  1,  9],
                          [4,  7,  0,  1, 21, 15],
                          [7,  5,  7,  9, 15, 27]])

        v = lambda x : a + b.T @ x + 0.5 * x.T @ C @ x
        self.p = opt.Problem(v)
        self.x_opt = -np.linalg.solve(C, b)

    def test_sd(self):
        x = np.array([[0], [0], [0], [0], [0], [0]])
        x_opt = opt.steepest_descent(self.p, x, tol=1e-6)
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-6)

    def test_cg(self):
        x = np.array([[0], [0], [0], [0], [0], [0]])
        x_opt = opt.conjugate_gradient(self.p, x, tol=1e-6)
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-6)

    def test_sec(self):
        x = np.array([[0], [0], [0], [0], [0], [0]])
        x_opt = opt.secant(self.p, x, tol=1e-6)
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-6)


class TestProblemB(unittest.TestCase):

    def setUp(self):
        v = lambda x : -np.sqrt((x[0]**2 + 1) * (2 * x[1]**2 + 1)) \
                       / (x[0]**2 + x[1]**2 + 0.5)
        self.x_opt = np.array([[0], [0]])
        self.p = opt.Problem(v)

    def test_sd(self):
        x = np.array([[10], [10]])
        x_opt = opt.steepest_descent(self.p, x, tol=1e-4)
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-3)

    def test_cg(self):
        x = np.array([[10], [10]])
        x_opt = opt.conjugate_gradient(self.p, x, tol=1e-4)
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-3)

    def test_sec(self):
        x = np.array([[10], [10]])
        x_opt = opt.secant(self.p, x, tol=1e-4)
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-3)


class TestProblemC(unittest.TestCase):

    def setUp(self):
        a = 1
        b = np.array([[1], [2]])
        C = np.array([[12, 3], [3, 10]])
        v = lambda x : a + b.T @ x + x.T @ C @ x \
                       + 10 * np.log(1 + x[0]**4) * np.sin(100 * x[0]) \
                       + 10 * np.log(1 + x[1]**4) * np.cos(100 * x[1])
        # TODO Verify that this is the real optimum
        self.x_opt = np.array([[-0.01773056364041071], [-0.09577801844122487]])
        self.p = opt.Problem(v)

    def test_sd(self):
        x = np.array([[0], [0]])
        x_opt = opt.steepest_descent(self.p, x, tol=1e-4)
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-3)

    def test_cg(self):
        x = np.array([[0], [0]])
        x_opt = opt.conjugate_gradient(self.p, x, tol=1e-4)
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-3)

    def test_sec(self):
        x = np.array([[0], [0]])
        x_opt = opt.secant(self.p, x, tol=1e-4)
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-3)


class TestBasics(unittest.TestCase):

    def test_scalar_problem(self):
        v = lambda x: x * x
        del_v = lambda x: 2 * x
        p = opt.Problem(v, del_v)
        self.assertEqual(p.cost(-2), 4)
        self.assertEqual(p.grad(-2), -4)

    def test_multivar_problem(self):
        P = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 4]])
        v = lambda x: 0.5 * x.T @ P @ x
        del_v = lambda x: x.T @ P
        p = opt.Problem(v, del_v)
        self.assertEqual(p.cost(np.array([[1], [1], [1]])), 3.5)
        self.assertSequenceEqual(p.grad(np.array([[1], [1], [1]])).tolist(),
                                 np.array([[1, 2, 4]]).tolist())

    def test_fd_problem(self):
        P = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 4]])
        v = lambda x: 0.5 * x.T @ P @ x
        del_v = lambda x: x.T @ P
        x = np.array([[1], [2], [3]])
        # Default step
        p = opt.Problem(v)
        # Exact gradient
        g_ex = del_v(x)
        # Gradient evaluated at x
        g_fd1 = p.grad(x)
        # Get gradient then evaluate at x
        g = p.grad
        g_fd2 = g(x)
        self.assertTrue(np.linalg.norm(g_fd1 - g_ex) < 1e-3)
        self.assertTrue(np.linalg.norm(g_fd2 - g_ex) < 1e-3)
        # Specific step
        p = opt.Problem(v, grad_step=1e-9)
        # Exact gradient
        g_ex = del_v(x)
        # Gradient evaluated at x
        g_fd1 = p.grad(x)
        # Get gradient then evaluate at x
        g = p.grad
        g_fd2 = g(x)
        self.assertTrue(np.linalg.norm(g_fd1 - g_ex) < 1e-3)
        self.assertTrue(np.linalg.norm(g_fd2 - g_ex) < 1e-3)

    def test_step_size(self):
        P = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 4]])
        v = lambda x: 0.5 * x.T @ P @ x
        del_v = lambda x: x.T @ P
        p = opt.Problem(v, del_v)
        # Compare step sizes at (1, 1, 1)
        x = np.array([[1], [1], [1]])
        s = -p.grad(x).T
        w_ideal = -(p.grad(x) @ s) / (s.T @ P @ s)
        w_armijo = opt._step_size(p, x, s)
        self.assertTrue(abs(w_armijo - w_ideal) / w_ideal < 0.1)

    def test_sd(self):
        P = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 4]])
        v = lambda x: 0.5 * x.T @ P @ x
        del_v = lambda x: x.T @ P
        p = opt.Problem(v, del_v)
        x = np.array([[1], [1], [1]])
        x_opt = np.array([[0], [0], [0]])
        x_sd = opt.steepest_descent(p, x)
        self.assertTrue(np.linalg.norm(x_sd - x_opt) < 1e-6)

    def test_cg(self):
        P = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 4]])
        v = lambda x: 0.5 * x.T @ P @ x
        del_v = lambda x: x.T @ P
        p = opt.Problem(v, del_v)
        x = np.array([[1], [1], [1]])
        x_opt = np.array([[0], [0], [0]])
        x_cg = opt.conjugate_gradient(p, x)
        self.assertTrue(np.linalg.norm(x_cg - x_opt) < 1e-6)

    def test_sec(self):
        P = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 4]])
        v = lambda x: 0.5 * x.T @ P @ x
        del_v = lambda x: x.T @ P
        p = opt.Problem(v, del_v)
        x = np.array([[1], [1], [1]])
        x_opt = np.array([[0], [0], [0]])
        x_sec = opt.secant(p, x)
        self.assertTrue(np.linalg.norm(x_sec - x_opt) < 1e-6)

    def test_fd_grad(self):
        P = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 4]])
        v = lambda x: 0.5 * x.T @ P @ x
        del_v = lambda x: x.T @ P
        p = opt.Problem(v, del_v)
        x = np.array([[1], [2], [3]])
        g_ex = p.grad(x)
        g_fd = opt._fd_grad(p, x, h=1e-9)
        self.assertTrue(np.linalg.norm(g_fd - g_ex) < 1e-3)


class TestBasicConstraints(unittest.TestCase):

    def setUp(self):
        # Example 12.1.5 from Fletcher
        v = lambda x: -x[0] - x[1]
        del_v = lambda x: np.ndarray([[-1, -1]])
        c = [lambda x: 1 - x[0]**2 - x[1]**2]
        self.p = opt.Problem(v, eq_const=c)

    def test_eq_const_init(self):
        v = lambda x: -x[0] - x[1]
        del_v = lambda x: np.ndarray([[-1, -1]])
        c1 = lambda x: x + 1
        c2 = lambda x: x + 2
        c = [c1, c2]
        p = opt.Problem(v, grad=del_v, eq_const=c)
        self.assertTrue(p.eq_const(4)[0] == 5)
        self.assertTrue(p.eq_const(4)[1] == 6)

    def test_eq_const(self):
        x0 = np.array([[0], [0]])
        x = opt.penalty_function(self.p, x0, tol=1e-4)
        x_opt = np.array([[0.7071318], [0.7071093]])
        self.assertTrue(np.linalg.norm(x_opt - x) < 1e-4)


if __name__ == '__main__':
    unittest.main()
