import unittest
import opt

class TestProblem(unittest.TestCase):

    def test_cost(self):
        v = lambda x : x * x
        del_v = lambda x : 2 * x
        p = opt.Problem(v, del_v)
        self.assertEqual(p.cost(-2), 4)
        self.assertEqual(p.grad(-2), -4)

if __name__ == '__main__':
    unittest.main()
