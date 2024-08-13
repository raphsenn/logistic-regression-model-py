import numpy as np


def sigmoid(X: np.array, derv=False) -> np.array:
    if derv:
        return sigmoid(X) * (1.0 - sigmoid(X))
    return 1.0 / (1.0 + np.exp(-X))


def step(x: float) -> int:
    if x >= 0.5:
        return 1.0
    return 0.0