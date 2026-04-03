import numpy as np


class LogisticRegressionModel:
    """
    Multi-class logistic regression using one-vs-rest strategy and batch gradient descent.

    Attributes:
        learning_rate: Step size for gradient descent.
        num_iterations: Number of training iterations.
        weights: shape (n_features, n_classes). Set after fit().
        bias: shape (n_classes,). Set after fit().
        mean: shape (n_features,). Set after fit(). Used for feature normalization.
        std: shape (n_features,). Set after fit(). Used for feature normalization.
        class_labels: Sorted unique class names. shape (n_classes,). Set after fit().
    """

    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.class_labels: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train model using batch gradient descent (one-vs-rest).

        args:
            X: shape (n_samples, n_features)
            y: shape (n_samples,). Class labels e.g. ["Gryffindor", ...]

        comments:
            - y (new) is converted to binary matrix of shape (n_samples, n_classes) where
                r[i][j] = 1 if sample i belongs to class j.
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.class_labels = np.unique(y)
        self.weights = np.zeros((n_features, len(self.class_labels)))
        self.bias = np.zeros(len(self.class_labels))

        # Normalize features and convert labels to binary matrix for multi-class classification
        X = self._normalization(X)
        y = (y.reshape(n_samples, 1) == self.class_labels).astype(int)

        # Gradient descent loop
        for _ in range(self.num_iterations):
            weight_gradient, bias_gradient = self._calculate_learning_step(X, y)

            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        returns:
            a list of predicted class labels. Return the highest probability class for each sample.
            shape (n_samples,). E.g. ["Gryffindor", "Slytherin", "Gryffindor", ...]
        """
        if self.weights is None or self.bias is None or self.mean is None or self.std is None:
            raise ValueError("Model must be fitted before prediction.")

        normalized_X = self._normalization(X)
        probabilities = self._compute_probabilities(normalized_X)

        return self.class_labels[np.argmax(probabilities, axis=1)]  # highest probability class

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        """Compute class membership probabilities via sigmoid.

        returns:
            shape (n_samples, n_classes). r[i][j] = P(sample i belongs to class j)
        """
        normalized_X = self._normalization(X)

        return self._compute_probabilities(normalized_X)

    def _compute_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Compute class membership probabilities via sigmoid.

        returns:
            shape (n_samples, n_classes). r[i][j] = P(sample i belongs to class j)
        """
        linear_output = np.dot(X, self.weights) + self.bias  # (n_samples, n_classes)

        return self._sigmoid_activation(linear_output)

    def _calculate_learning_step(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients of binary cross-entropy loss.

        formula:
            ∂J/∂w = (1/m) * X.T @ (hθ(x) - y)
            ∂J/∂b = (1/m) * sum(hθ(x) - y)

        returns:
            weight_gradient: shape (n_features, n_classes)
            bias_gradient: shape (n_classes,)
        """
        error = self._compute_probabilities(X) - y  # (n_samples, n_classes)

        weight_gradient = np.dot(X.T, error) / X.shape[0]
        bias_gradient = np.mean(error, axis=0)

        return weight_gradient, bias_gradient

    def _sigmoid_activation(self, z: np.ndarray) -> np.ndarray:
        """
        Maps any real value to (0, 1): σ(z) = 1 / (1 + e^-z)
        """
        # return 1 / (1 + np.exp(-z))
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),  # For z >= 0, compute sigmoid directly
            np.exp(z) / (1 + np.exp(z)),  # For z < 0, alternative formula to avoid overflow
        )

    def _normalization(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features to have mean 0 and std 1.

        formula:
            X_normalized = (X - mean) / std
        """
        return (X - self.mean) / (self.std + 1e-8)  # Avoid division by zero


    @staticmethod
    def compare_predictions(prediction: np.ndarray, truth: np.ndarray) -> float:
        """Compute accuracy between two sets of predictions.

        returns:
            Accuracy as a float in [0, 1].
        """
        return np.mean(prediction == truth)
