import numpy as np


def sigmoid(X: np.array, derv=False) -> np.array:
    if derv:
        pass
    return 1.0 / (1.0 + np.exp(-X))