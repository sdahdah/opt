import unittest
import numpy as np
import opt

class TestProblem(unittest.TestCase):

    def test_scalar_problem(self):
        v = lambda x : x * x
        del_v = lambda x : 2 * x
        p = opt.Problem(v, del_v)
        self.assertEqual(p.cost(-2), 4)
        self.assertEqual(p.grad(-2), -4)

    def test_multivar_problem(self):
        P = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 4]])
        v = lambda x : 0.5 * x.T @ P @ x
        del_v = lambda x : x.T @ P
        p = opt.Problem(v, del_v)
        self.assertEqual(p.cost(np.array([[1], [1], [1]])), 3.5)
        self.assertSequenceEqual(p.grad(np.array([[1], [1], [1]])).tolist(),
                                 np.array([[1, 2, 4]]).tolist())

if __name__ == '__main__':
    unittest.main()
