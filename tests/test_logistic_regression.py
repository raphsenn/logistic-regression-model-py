#!/usr/bin/env python3
# Author: Raphael Senn

from src.logistic_regression import LogisticRegression
import numpy as np


def test_constructor():
    lr = LogisticRegression(n_features=2, random_state=False)
    assert np.array_equal(np.zeros((2, 1)), lr.W) is True
    assert np.array_equal(np.zeros((1, 1)), lr.B) is True


# Logical NOT-Gate
def test_train_1():
    NOT_Gate = LogisticRegression(1, max_iterations=10000, learning_rate=0.01)
    X_train = np.array([[0], [1]])
    y_train = np.array([[1], [0]])
    NOT_Gate.fit(X_train, y_train, batch_size=1)
    np.testing.assert_array_equal(NOT_Gate.predict(X_train), y_train)


# Logical AND-Gate
def test_train_2():
    AND_Gate = LogisticRegression(2, max_iterations=10000, learning_rate=0.01)
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [0], [0], [1]])
    AND_Gate.fit(X_train, y_train, batch_size=1)
    np.testing.assert_array_equal(AND_Gate.predict(X_train), y_train)


# Logical OR-Gate
def test_train_3():
    OR_Gate = LogisticRegression(2, max_iterations=10000, learning_rate=0.01)
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [1]])
    OR_Gate.fit(X_train, y_train, batch_size=1)
    np.testing.assert_array_equal(OR_Gate.predict(X_train), y_train)