import numpy as np

class LogisticRegressionModel:
    """
    Multi-class logistic regression using one-vs-rest strategy and batch gradient descent.

    Attributes:
        learning_rate: Step size for gradient descent.
        num_iterations: Number of training iterations.
        weights: shape (n_features, n_classes). Set after fit().
        bias: shape (n_classes,). Set after fit().
        class_labels: Sorted unique class names. shape (n_classes,). Set after fit().
    """
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.class_labels: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train model using batch gradient descent (one-vs-rest).

        args:
            X: shape (n_samples, n_features)
            y: shape (n_samples,). Class labels e.g. ["Gryffindor", ...]
        """
        n_samples, n_features = X.shape

        self.class_labels = np.unique(y)
        self.weights = np.zeros((n_features, len(self.class_labels)))
        self.bias = np.zeros(len(self.class_labels))

        # shape (n_samples, n_classes). r[i][j] = 1 if sample i belongs to class j
        y_binary = (y.reshape(n_samples, 1) == self.class_labels).astype(int)

        for _ in range(self.num_iterations):
            weight_gradient, bias_gradient = self.calculate_learning_step(X, y_binary)

            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        returns:
            shape (n_samples,). E.g. ["Gryffindor", "Slytherin", "Gryffindor", ...]
        """
        predicted_probabilities = self.predict_probability(X)  # (n_samples, n_classes)

        return self.class_labels[np.argmax(predicted_probabilities, axis=1)]

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        """
        Compute class membership probabilities via sigmoid.

        returns:
            shape (n_samples, n_classes). r[i][j] = P(sample i belongs to class j)
        """
        linear_output = np.dot(X, self.weights) + self.bias  # (n_samples, n_classes)

        return self.sigmoid_activation(linear_output)

    def calculate_learning_step(self, X: np.ndarray, y_binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients of binary cross-entropy loss.

        formula:
            ∂J/∂w = (1/m) * X.T @ (hθ(x) - y)
            ∂J/∂b = (1/m) * sum(hθ(x) - y)

        returns:
            weight_gradient: shape (n_features, n_classes)
            bias_gradient: shape (n_classes,)
        """
        error = self.predict_probability(X) - y_binary  # (n_samples, n_classes)

        weight_gradient = np.dot(X.T, error) / X.shape[0]
        bias_gradient = np.mean(error, axis=0)

        return weight_gradient, bias_gradient

    def sigmoid_activation(self, z: np.ndarray) -> np.ndarray:
        """
        Maps any real value to (0, 1): σ(z) = 1 / (1 + e^-z)
        """
        return 1 / (1 + np.exp(-z))
