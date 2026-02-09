import numpy as np

def generate_boolean_inputs(n):
    X = np.array(np.meshgrid(*[[-1, 1]] * n)).T.reshape(-1, n)
    return X


class HiddenLayer:
    def __init__(self, n_features):
        self.patterns = generate_boolean_inputs(n_features)
        self.weights = self.patterns.copy()
        self.bias = -n_features + 0.5

    def forward(self, X):
        return np.sign(X @ self.weights.T + self.bias)
    
class OutputPerceptronThreshold:
    def __init__(self, n_inputs):
        self.w = np.zeros(n_inputs)
        self.b = 0

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

    def train(self, X, y, epochs=50):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                if np.sign(xi @ self.w + self.b) != yi:
                    self.w += yi * xi
                    self.b += yi

class OutputPerceptronSigmoid:
    def __init__(self, n_inputs):
        self.w = np.zeros(n_inputs)
        self.b = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        probs = self.sigmoid(X @ self.w + self.b)
        return np.where(probs >= 0.5, 1, -1)

    def train(self, X, y, epochs=50):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                pred = 1 if self.sigmoid(xi @ self.w + self.b) >= 0.5 else -1
                if pred != yi:
                    self.w += yi * xi
                    self.b += yi

class BooleanMLP:
    def __init__(self, n_features, mode="threshold"):
        self.hidden = HiddenLayer(n_features)
        n_hidden = 2 ** n_features

        if mode == "threshold":
            self.output = OutputPerceptronThreshold(n_hidden)
        else:
            self.output = OutputPerceptronSigmoid(n_hidden)

    def train(self, X, y):
        H = self.hidden.forward(X)
        self.output.train(H, y)

    def predict(self, X):
        H = self.hidden.forward(X)
        return self.output.predict(H)                    