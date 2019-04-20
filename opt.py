class Problem:
    """Optimization problem"""

    def __init__(self, cost, grad):
        self._cost = cost
        self._grad = grad

    def cost(self, x):
        return self._cost(x)

    def grad(self, x):
        return self._grad(x)
