import numpy as np


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
        pass

    def fit(self, X_train: np.array, y_train: np.array, batch_size: int=1) -> None:
        pass


if __name__ == "__main__":
    lr = LogisticRegression(2)
    print(lr.W)
    print(lr.B)
