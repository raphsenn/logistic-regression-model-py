import numpy as np
from act import sigmoid


class LogisticRegression:
    def __init__(self, n_features: int, random_state: bool=True, learning_rate: float = 0.1, max_iterations: int = 1):
        if random_state:
            self.W = np.random.rand(n_features)
            self.B = np.random.rand(1)
        else:
            self.W = np.zeros(n_features)
            self.B = np.zeros(1)

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def forward(self, x: np.array) -> None:
        """ Forward propagation. """
        self.Z1 = np.dot(x, self.W) + self.B
        self.A1 = sigmoid(self.Z1)

    def backward(self, y: np.array) -> None:
        """ Backpropagation. """
        self.error = (y - self.A1) ** 2

    def fit(self, X_train: np.array, y_train: np.array, batch_size: int=1, verbose: bool=False) -> None:
        """ Fit the data. """ 
        for epoch in range(self.max_iterations):
            for i in range(0, len(X_train), batch_size):
                self.forward(X_train[i:i+batch_size]) 
                self.backward(y_train[i:i+batch_size])

if __name__ == "__main__":
    and_gate = LogisticRegression(2)
    print(and_gate.W)
    print(and_gate.B)
