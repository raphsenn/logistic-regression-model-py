import numpy as np
from act import sigmoid, step


class LogisticRegression:
    def __init__(self, n_features: int, random_state: bool=True, learning_rate: float = 0.1, max_iterations: int = 1, d_type=np.float64):
        # Create weights and biases.
        if random_state:
            self.W = np.random.rand(n_features)
            self.B = np.random.rand(1)
        else:
            self.W = np.zeros(n_features, dtype=d_type)
            self.B = np.zeros(1, dtype=d_type)
        
        # Set learning rate and max iterations.
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def forward(self, x: np.array) -> None:
        """ Forward propagation. """
        self.Z = np.dot(x, self.W) + self.B
        self.A = sigmoid(self.Z)

    def backward(self, X: np.array, y: np.array) -> None:
        """ Backpropagation. """
        self.loss = (y - self.A)

        dW = 2 * (self.loss) * sigmoid(self.Z, derv=True) * X
        dB = 2 * (self.loss) * sigmoid(self.Z, derv=True)

        self.W += dW
        self.B += dB

    def predict(self, x: np.array) -> bool:
        self.forward(x)
        return self.A
        # return step(self.A1)

    def fit(self, X_train: np.array, y_train: np.array, batch_size: int=1, learning_rate:float=0.1,verbose: bool=False) -> None:
        """ Fit the data. """ 
        for epoch in range(self.max_iterations):
            for i in range(0, len(X_train), batch_size):
                self.forward(X_train[i:i+batch_size]) 
                self.backward(X_train[i:i+batch_size], y_train[i:i+batch_size])
            
            if verbose:
                print(f"Epoch {epoch}, Loss {self.loss}")


if __name__ == "__main__":
    AND_Gate = LogisticRegression(2, max_iterations=1000, learning_rate=0.01)
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 0, 1])
    AND_Gate.fit(X_train, y_train, batch_size=2)

    print(AND_Gate.predict(np.array([0, 0])))
    print(AND_Gate.predict(np.array([0, 1])))
    print(AND_Gate.predict(np.array([1, 0])))
    print(AND_Gate.predict(np.array([1, 1])))