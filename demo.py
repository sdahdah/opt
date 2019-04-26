import unittest
import numpy as np
import matplotlib.pyplot as plt
import opt

np.set_printoptions(precision=20, linewidth=120)

@unittest.skip('Done generating plots for now')
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
        x_opt = opt.steepest_descent(self.p, x, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g)
        plt.xlabel('Iteration')
        plt.ylabel('Norm of Gradient')
        fig.savefig('./fig/sd-pA-grad.eps', format='eps')

    def test_cg(self):
        x = np.array([[0], [0], [0], [0], [0], [0]])
        x_opt = opt.conjugate_gradient(self.p, x, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g)
        plt.xlabel('Iteration')
        plt.ylabel('Norm of Gradient')
        fig.savefig('./fig/cg-pA-grad.eps', format='eps')

    def test_sec(self):
        x = np.array([[0], [0], [0], [0], [0], [0]])
        x_opt = opt.secant(self.p, x, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g)
        plt.xlabel('Iteration')
        plt.ylabel('Norm of Gradient')
        fig.savefig('./fig/sec-pA-grad.eps', format='eps')


@unittest.skip('Done generating plots for now')
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
        x_opt = opt.steepest_descent(self.p, x, tol=1e-6, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g)
        plt.xlabel('Iteration')
        plt.ylabel('Norm of Gradient')
        fig.savefig('./fig/sd-pA.eps', format='eps')

    def test_cg(self):
        x = np.array([[0], [0], [0], [0], [0], [0]])
        x_opt = opt.conjugate_gradient(self.p, x, tol=1e-6, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g)
        plt.xlabel('Iteration')
        plt.ylabel('Norm of Gradient')
        fig.savefig('./fig/cg-pA.eps', format='eps')

    def test_sec(self):
        x = np.array([[0], [0], [0], [0], [0], [0]])
        x_opt = opt.secant(self.p, x, tol=1e-6, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g)
        plt.xlabel('Iteration')
        plt.ylabel('Norm of Gradient')
        fig.savefig('./fig/sec-pA.eps', format='eps')


@unittest.skip('Done generating plots for now')
class TestProblemB(unittest.TestCase):

    def setUp(self):
        v = lambda x : -np.sqrt((x[0, 0]**2 + 1) * (2 * x[1, 0]**2 + 1)) \
                       / (x[0, 0]**2 + x[1, 0]**2 + 0.5)
        self.x_opt = np.array([[0], [0]])
        self.p = opt.Problem(v)

    def test_sd(self):
        x = np.array([[10], [10]])
        x_opt = opt.steepest_descent(self.p, x, tol=1e-4, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g)
        plt.xlabel('Iteration')
        plt.ylabel('Norm of Gradient')
        fig.savefig('./fig/sd-pB.eps', format='eps')

    def test_cg(self):
        x = np.array([[10], [10]])
        x_opt = opt.conjugate_gradient(self.p, x, tol=1e-4, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g)
        plt.xlabel('Iteration')
        plt.ylabel('Norm of Gradient')
        fig.savefig('./fig/cg-pB.eps', format='eps')

    def test_sec(self):
        x = np.array([[10], [10]])
        x_opt = opt.secant(self.p, x, tol=1e-4, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g)
        plt.xlabel('Iteration')
        plt.ylabel('Norm of Gradient')
        fig.savefig('./fig/sec-pB.eps', format='eps')


@unittest.skip('Done generating plots for now')
class TestProblemC(unittest.TestCase):

    def setUp(self):
        a = 1
        b = np.array([[1], [2]])
        C = np.array([[12, 3], [3, 10]])
        v = lambda x : a + b.T @ x + x.T @ C @ x \
                       + 10 * np.log(1 + x[0, 0]**4) * np.sin(100 * x[0, 0]) \
                       + 10 * np.log(1 + x[1, 0]**4) * np.cos(100 * x[1, 0])
        self.p = opt.Problem(v)

    def test_sd(self):
        x = np.array([[0], [0]])
        x_opt = opt.steepest_descent(self.p, x, tol=1e-4, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g)
        plt.xlabel('Iteration')
        plt.ylabel('Norm of Gradient')
        fig.savefig('./fig/sd-pC.eps', format='eps')

    def test_cg(self):
        x = np.array([[0], [0]])
        x_opt = opt.conjugate_gradient(self.p, x, tol=1e-4, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g)
        plt.xlabel('Iteration')
        plt.ylabel('Norm of Gradient')
        fig.savefig('./fig/cg-pC.eps', format='eps')

    def test_sec(self):
        x = np.array([[0], [0]])
        x_opt = opt.secant(self.p, x, tol=1e-4, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g)
        plt.xlabel('Iteration')
        plt.ylabel('Norm of Gradient')
        fig.savefig('./fig/sec-pC.eps', format='eps')


# @unittest.skip('Done generating plots for now')
class TestProblemD(unittest.TestCase):

    def setUp(self):
        v = lambda x: np.abs(x[0, 0] - 2) + np.abs(x[1, 0] - 2)
        h1 = lambda x: x[0, 0] - x[1, 0]**2
        h2 = lambda x: x[0, 0]**2 + x[1, 0]**2 - 1
        self.p = opt.Problem(v, eq_const=[h2], ineq_const=[h1])
        self.x_opt = np.array([[np.sqrt(2)/2], [np.sqrt(2)/2]])

    @unittest.skip('Temp')
    def test_penalty_function(self, tol=1e-3, tol_const=1e-3):
        x0 = np.array([[0], [0]])
        x_opt = opt.penalty_function(self.p, x0, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        c_e = np.array([np.linalg.norm(self.p.eq_const(x_opt[i]))
            for i in range(len(x_opt))])
        c_i = np.array([np.linalg.norm(np.minimum(self.p.ineq_const(x_opt[i]), 0))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g, label='Gradient Norm')
        plt.plot(np.arange(len(x_opt)), c_e, label='Equality Constraint Norm')
        plt.plot(np.arange(len(x_opt)), c_i, label='Inequality Constraint Norm')
        plt.xticks(np.arange(len(x_opt)))
        plt.xlabel('Iteration')
        plt.legend()
        fig.savefig('./fig/pe-pD.eps', format='eps')

    @unittest.skip('Temp')
    def test_inv_barrier_function(self, tol=1e-3, tol_const=1e-3):
        x0 = np.array([[0.1], [0.1]])
        x_opt = opt.barrier_function(self.p, x0, mode='inv', hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        c_e = np.array([np.linalg.norm(self.p.eq_const(x_opt[i]))
            for i in range(len(x_opt))])
        c_i = np.array([np.linalg.norm(np.minimum(self.p.ineq_const(x_opt[i]), 0))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g, label='Gradient Norm')
        plt.plot(np.arange(len(x_opt)), c_e, label='Equality Constraint Norm')
        plt.plot(np.arange(len(x_opt)), c_i, label='Inequality Constraint Norm')
        plt.xticks(np.arange(len(x_opt)))
        plt.xlabel('Iteration')
        plt.legend()
        fig.savefig('./fig/ba-pD.eps', format='eps')

    @unittest.skip('Log barrier function does not work correctly')
    def test_log_barrier_function(self, tol=1e-3, tol_const=1e-3):
        x0 = np.array([[0.1], [0.1]])
        x = opt.barrier_function(self.p, x0, mode='log')

    @unittest.skip('Temp')
    def test_aug_lag(self):
        x0 = np.array([[0.1], [0.1]])
        x_opt = opt.augmented_lagrange(self.p, x0, tol=1e-4, tol_const=1e-4, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        c_i = np.array([np.linalg.norm(np.minimum(self.p.ineq_const(x_opt[i]), 0))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g, label='Gradient Norm')
        plt.plot(np.arange(len(x_opt)), c_i, label='Inequality Constraint Norm')
        plt.xticks(np.arange(len(x_opt)))
        plt.xlabel('Iteration')
        plt.legend()
        fig.savefig('./fig/al-pD.eps', format='eps')

    def test_lag_new(self):
        x0 = np.array([[0.1], [0.1]])
        x_opt = opt.lagrange_newton(self.p, x0, tol=1e-4)
        print(x_opt)


@unittest.skip('Done generating plots for now')
class TestProblemE(unittest.TestCase):

    def setUp(self):
        v = lambda x: -x[0, 0] * x[1, 0]
        h1 = lambda x: -x[0, 0] - x[1, 0]**2 + 1
        h2 = lambda x: x[0, 0] + x[1, 0]
        self.p = opt.Problem(v, ineq_const=[h1, h2])

    def test_penalty_function(self, tol=1e-4, tol_const=1e-4):
        x0 = np.array([[1], [1]])
        x_opt = opt.penalty_function(self.p, x0, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        c_i = np.array([np.linalg.norm(np.minimum(self.p.ineq_const(x_opt[i]), 0))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g, label='Gradient Norm')
        plt.plot(np.arange(len(x_opt)), c_i, label='Inequality Constraint Norm')
        plt.xticks(np.arange(len(x_opt)))
        plt.xlabel('Iteration')
        plt.legend()
        fig.savefig('./fig/pe-pE.eps', format='eps')

    @unittest.skip('Does not terminate ... ?')
    def test_inv_barrier_function(self, tol=1e-4, tol_const=1e-4):
        x0 = np.array([[1], [1]])
        x_opt = opt.barrier_function(self.p, x0, mode='inv', hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        c_i = np.array([np.linalg.norm(np.minimum(self.p.ineq_const(x_opt[i]), 0))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g, label='Gradient Norm')
        plt.plot(np.arange(len(x_opt)), c_i, label='Inequality Constraint Norm')
        plt.xticks(np.arange(len(x_opt)))
        plt.xlabel('Iteration')
        plt.legend()
        fig.savefig('./fig/ba-pE.eps', format='eps')

    @unittest.skip('Log barrier function does not work correctly')
    def test_log_barrier_function(self, tol=1e-4, tol_const=1e-4):
        x0 = np.array([[1], [1]])
        x = opt.barrier_function(self.p, x0, mode='log')

    def test_aug_lag(self):
        x0 = np.array([[1], [1]])
        x_opt = opt.augmented_lagrange(self.p, x0, tol=1e-4, tol_const=1e-4, hist=True)
        g = np.array([np.linalg.norm(self.p.grad(x_opt[i]))
            for i in range(len(x_opt))])
        c_i = np.array([np.linalg.norm(np.minimum(self.p.ineq_const(x_opt[i]), 0))
            for i in range(len(x_opt))])
        fig = plt.figure()
        plt.plot(np.arange(len(x_opt)), g, label='Gradient Norm')
        plt.plot(np.arange(len(x_opt)), c_i, label='Inequality Constraint Norm')
        plt.xticks(np.arange(len(x_opt)))
        plt.xlabel('Iteration')
        plt.legend()
        fig.savefig('./fig/al-pE.eps', format='eps')


@unittest.skip('Does not work at all')
class TestProblemF(unittest.TestCase):

    def setUp(self):
        v = lambda x: np.log(x[0, 0]) - x[1, 0]
        h1 = lambda x: x[0, 0] - 1
        h2 = lambda x: x[0, 0]**2 + x[1, 0]**2 - 4
        self.p = opt.Problem(v, eq_const=[h2], ineq_const=[h1])

    def test_penalty_function(self, tol=1e-4, tol_const=1e-4):
        x0 = np.array([[1], [1]])
        x = opt.penalty_function(self.p, x0)

    def test_inv_barrier_function(self, tol=1e-4, tol_const=1e-4):
        x0 = np.array([[1], [1]])
        x = opt.barrier_function(self.p, x0, mode='inv')

    def test_log_barrier_function(self, tol=1e-4, tol_const=1e-4):
        x0 = np.array([[1], [1]])
        x = opt.barrier_function(self.p, x0, mode='log')

    def test_aug_lag(self):
        x0 = np.array([[2.1], [0.1]])
        x = opt.augmented_lagrange(self.p, x0, tol=1e-4, tol_const=1e-4)


if __name__ == '__main__':
    unittest.main()
