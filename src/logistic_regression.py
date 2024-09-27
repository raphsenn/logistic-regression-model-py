#!/usr/bin/env python3
# Author: Raphael Senn

import numpy as np
import matplotlib.pyplot as plt


# Sigmoid activation function.
def sigmoid(X: np.array) -> np.array:
    return 1.0 / (1.0 + np.exp(-X))


# Unit-Step function.
def step(x: np.array) -> np.array:
    return np.heaviside(x - 0.5, 0)


class LogisticRegression:
    def __init__(self, n_features: int, random_state: bool=True, learning_rate: float = 0.1, max_iterations: int = 1, d_type=np.float64):
        # Create weights and biases.
        if random_state:
            self.W = np.random.rand(n_features, 1)                     # n x 1
            self.B = np.random.rand(1, 1)                              # 1 x 1
        else:
            self.W = np.zeros((n_features, 1))                           # n x 1
            self.B = np.zeros((1, 1))                                    # 1 x 1

        # Set learning rate and max iterations.
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def forward(self, x: np.array) -> None:
        # Let k be the badge size (k natural number).
        self.Z = np.dot(x, self.W) + self.B                         # (k x n) @ (n x 1) = k x 1
        self.A = sigmoid(self.Z)                                    # k x 1

    def backward(self, X: np.array, y: np.array) -> None:
        # Let k be the badge size (k natural number).
        # Calculate output error. 
        self.output_error = (y - self.A)                            # k x 1

        # dim(dW) = (k x n)^T @ (k x 1) = n x 1
        dW = np.dot(X.T, self.output_error) / len(X)                # n x 1

        # dim(dB) = (1 x 1) 
        dB = np.sum(self.output_error) / len(X)                     # 1 x 1

        self.W += self.learning_rate * dW                           # n x 1
        self.B += self.learning_rate * dB                           # 1 x 1


    def predict(self, x: np.array) -> bool:
        self.forward(x)
        return step(self.A)

    def fit(self, X_train: np.array, y_train: np.array, batch_size: int=1, learning_rate:float=0.1,verbose: bool=False, plot: bool=False) -> None:
        """ Fit the data. """ 
        for epoch in range(self.max_iterations):
            for i in range(0, len(X_train), batch_size):
                self.forward(X_train[i:i+batch_size]) 
                self.backward(X_train[i:i+batch_size], y_train[i:i+batch_size])
            
            if verbose:
                print(f"Epoch {epoch}, Loss {self.loss}")