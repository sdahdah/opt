import unittest
import numpy as np
import opt

np.set_printoptions(precision=20)

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


if __name__ == '__main__':
    unittest.main()
