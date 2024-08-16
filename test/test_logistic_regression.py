from src.logistic_regression import LogisticRegression
import numpy as np


def test_constructor():
    lr = LogisticRegression(n_features=2, random_state=False)
    assert np.array_equal(np.zeros(2), lr.W) is True
    assert np.array_equal(np.zeros(1), lr.B) is True


