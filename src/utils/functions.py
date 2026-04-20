import numpy as np

class QuadraticBowl:
    """
    A simple quadratic function f(x, y) = 0.5 * (a*x^2 + b*y^2).
    The condition number kappa is max(a, b) / min(a, b).
    """
    def __init__(self, a: float = 1.0, b: float = 1.0):
        self.a = a
        self.b = b
        self.A = np.array([[a, 0.0],[0.0, b]])

    def __call__(self, x: np.ndarray) -> float:
        return 0.5 * np.sum(x * (self.A @ x), axis=0)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return self.A
