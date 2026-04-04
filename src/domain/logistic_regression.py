import json
from collections.abc import Generator

import numpy as np

from .gradient_descent_strategy import GradientDescentStrategy
from .utils import requires_training

# ── Constants ────────────────────────────────────────────────────────────────
LEARNING_RATE = 0.01
NUM_ITERATIONS = 1000
NORMALIZATION_EPSILON = 1e-8  # Prevents division by zero during standardization
CLIP_MIN = 1e-15  # Prevents log(0) in cross-entropy loss
CLIP_MAX = 1 - CLIP_MIN


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
        labels: Sorted unique class names. shape (n_classes,). Set after fit().
    """

    def __init__(
        self,
        learning_rate: float = LEARNING_RATE,
        num_iterations: int = NUM_ITERATIONS,
        gd_strategy: GradientDescentStrategy = GradientDescentStrategy.BATCH(),
    ) -> None:
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.gd_strategy = gd_strategy
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.labels: np.ndarray | None = None

    def __repr__(self) -> str:
        return (
            f"LogisticRegressionModel("
            f"learning_rate={self.learning_rate}, "
            f"num_iterations={self.num_iterations}, "
            f"fitted={self.weights is not None})"
        )

    # ── Alternative constructors ─────────────────────────────────────────────

    @classmethod
    def from_json(cls, file_path: str) -> "LogisticRegressionModel":
        """Load model parameters from a JSON file."""
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)

        model = cls()
        model.weights = np.asarray(data["weights"], dtype=float)
        model.bias = np.asarray(data["bias"], dtype=float)
        model.mean = np.asarray(data["mean"], dtype=float)
        model.std = np.asarray(data["std"], dtype=float)
        model.labels = np.asarray(data["labels"])
        return model

    # ── Public interface ─────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using batch gradient descent (one-vs-rest).

        Args:
            X:  shape (n_samples, n_features). Unnormalized feature matrix.
            y:  shape (n_samples,). Class labels e.g. ["Gryffindor", ...].
        """
        # weights, bias, mean, std, and labels
        self._initialize_parameters(X, y)

        X = self._normalization(X)
        y = self._encode_labels(y, X.shape[0])

        # Gradient descent loop
        for _ in range(self.num_iterations):
            # Divide data into batches depending on the strategy (batch, mini-batch, stochastic)
            for X_batch, y_batch in self._generate_batches(X, y):
                probabilities = self._compute_probabilities(X_batch)

                w_step, b_step = self._calculate_learning_step(X_batch, y_batch, probabilities)

                self.weights -= self.learning_rate * w_step  # + Regularization?
                self.bias -= self.learning_rate * b_step

    @requires_training
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        returns:
            a list of predicted class labels. Return the highest probability class for each sample.
            shape (n_samples,). E.g. ["Gryffindor", "Slytherin", "Gryffindor", ...]
        """
        normalized_X = self._normalization(X)
        probabilities = self._compute_probabilities(normalized_X)

        return self.labels[np.argmax(probabilities, axis=1)]  # highest probability class

    @requires_training
    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        """Compute class membership probabilities via sigmoid.

        returns:
            shape (n_samples, n_classes). r[i][j] = P(sample i belongs to class j)
        """
        normalized_X = self._normalization(X)

        return self._compute_probabilities(normalized_X)

    @requires_training
    def save_json(self, file_path: str) -> None:
        """Save model parameters to a JSON file."""
        data = {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "labels": self.labels.tolist(),
        }

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file)

    def is_trained(self) -> None:
        """Raise if the model has not been trained yet."""
        if any(p is None for p in (self.weights, self.bias, self.mean, self.std, self.labels)):
            raise ValueError("Model is not fitted yet. Call fit() first.")

    # ── Private helpers ──────────────────────────────────────────────────────

    def _initialize_parameters(self, X: np.ndarray, y: np.ndarray) -> None:
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.labels = np.unique(y)
        self.weights = np.zeros((X.shape[1], len(self.labels)))
        self.bias = np.zeros(len(self.labels))

    def _encode_labels(self, y: np.ndarray, n_samples: int) -> np.ndarray:
        """Convert label vector to binary matrix. shape (n_samples, n_classes)."""
        # Compare each label with all unique labels
        binary_matrix = y.reshape(n_samples, 1) == self.labels

        # Convert boolean matrix to integers (True -> 1, False -> 0)
        return binary_matrix.astype(int)

    def _compute_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Compute class membership probabilities via sigmoid.

        formula:
            z = X * weights + bias
            probabilities = sigmoid(z) -> σ(z) = 1 / (1 + e^-z)

        returns:
            shape (n_samples, n_classes). r[i][j] = P(sample i belongs to class j)
        """
        linear_output = np.dot(X, self.weights) + self.bias  # (n_samples, n_classes)

        return self._sigmoid_activation(linear_output)

    def _calculate_learning_step(
        self, X: np.ndarray, y: np.ndarray, probabilities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute gradients of binary cross-entropy loss.

        formula:
            ∂J/∂w = (1/m) * X.T @ (hθ(x) - y)
            ∂J/∂b = (1/m) * sum(hθ(x) - y)

        returns:
            weight_gradient: shape (n_features, n_classes)
            bias_gradient: shape (n_classes,)
        """
        error = probabilities - y  # (n_samples, n_classes)

        weight_gradient = np.dot(X.T, error) / X.shape[0]
        bias_gradient = np.mean(error, axis=0)

        return weight_gradient, bias_gradient

    def _compute_loss(self, probabilities: np.ndarray, y: np.ndarray) -> float:
        """Compute binary cross-entropy loss.

        formula:
            J(θ) = -(1/m) * sum(y * log(hθ(x)) + (1 - y) * log(1 - hθ(x)))

        returns:
            Loss as a float. Lower is better.
        """
        # Clip probabilities to avoid log(0)
        probabilities = np.clip(probabilities, CLIP_MIN, CLIP_MAX)

        return -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))

    def _normalization(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features to have mean 0 and std 1.

        formula:
            X_normalized = (X - mean) / std
        """
        return (X - self.mean) / (self.std + NORMALIZATION_EPSILON)

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

    def _generate_batches(
        self, X: np.ndarray, y: np.ndarray
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """Yield batches of data for mini-batch or stochastic gradient descent."""
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)  # Shuffle indices for randomness

        if self.gd_strategy.batch_size is None:  # Batch gradient descent
            yield X, y
            return

        for start in range(0, n_samples, self.gd_strategy.batch_size):
            batch_indices = indices[start : start + self.gd_strategy.batch_size]
            yield X[batch_indices], y[batch_indices]

    @staticmethod
    def compare_predictions(prediction: np.ndarray, truth: np.ndarray) -> float:
        """Compute accuracy between two sets of predictions.

        returns:
            Accuracy as a float in [0, 1].
        """
        return np.mean(prediction == truth)
