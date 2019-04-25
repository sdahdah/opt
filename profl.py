
import unittest
from pstats import Stats
import cProfile
import numpy as np
import opt

np.set_printoptions(precision=20)


class ProfileProblemA(unittest.TestCase):

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
        self.p = opt.Problem(v, grad_step=1e-8)
        self.x_opt = -np.linalg.solve(C, b)
        self.pr = cProfile.Profile()
        self.pr.enable()

    def tearDown(self):
        p = Stats (self.pr)
        p.strip_dirs()
        p.sort_stats ('cumtime')
        p.print_stats()

    def test_sd_p(self):
        x = np.array([[0], [0], [0], [0], [0], [0]])
        x_opt = opt.steepest_descent(self.p, x, tol=1e-6)
        print('sd')
        print(x_opt)
        print(self.x_opt)
        print()
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-3)

    def test_cg_p(self):
        x = np.array([[0], [0], [0], [0], [0], [0]])
        x_opt = opt.conjugate_gradient(self.p, x, tol=1e-6)
        print('cg')
        print(x_opt)
        print(self.x_opt)
        print()
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-3)

    def test_sec_p(self):
        x = np.array([[0], [0], [0], [0], [0], [0]])
        x_opt = opt.secant(self.p, x, tol=1e-6)
        print('sec')
        print(x_opt)
        print(self.x_opt)
        print()
        self.assertTrue(np.linalg.norm(x_opt - self.x_opt) < 1e-3)


if __name__ == '__main__':
    unittest.main()
