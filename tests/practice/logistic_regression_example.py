"""
NumPy notation used in this file:

- np.asarray(x, dtype=np.float64): converts input data to a NumPy array.
- np.unique(y, return_inverse=True): returns the distinct labels and the encoded class index for each sample.
- np.zeros(shape, dtype=np.float64): creates an array filled with zeros.
- np.eye(n, dtype=np.float64): creates the identity matrix; used here to build one-hot labels.
- np.argmax(array, axis=1): returns the index of the largest value in each row.
- np.exp(x): computes the exponential of each element.
- np.empty_like(x, dtype=np.float64): creates an uninitialized array with the same shape as x.
- np.sum(x, axis=...): sums values along the given axis.
- np.clip(x, min, max): limits values to a fixed range.
- np.log(x): computes the natural logarithm of each element.
"""

import numpy as np


class LogisticRegression:
    """
    Multiclass logistic regression (one-vs-rest sigmoid) trained with batch gradient descent.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        num_iterations: int = 1000,
    ):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights: np.ndarray | None = None  # shape (n_features, n_classes)
        self.bias: np.ndarray | None = None  # shape (n_classes,)
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train a multiclass logistic regression model using sigmoid-based one-vs-rest learning.

        This implementation supports K classes directly by optimizing independent
        binary cross-entropy objectives, one per class.

        args:
            X: Input feature matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).
        """
        X_checked = np.asarray(X, dtype=np.float64)
        y_checked = np.asarray(y)
        num_samples, num_features = X_checked.shape

        self.classes_, y_encoded = np.unique(y_checked, return_inverse=True)
        num_classes = self.classes_.shape[0]

        self.weights = np.zeros((num_features, num_classes), dtype=np.float64)
        self.bias = np.zeros(num_classes, dtype=np.float64)

        y_one_hot = np.eye(num_classes, dtype=np.float64)[y_encoded]

        for _ in range(self.num_iterations):
            linear_model = X_checked @ self.weights + self.bias
            y_predicted = self._sigmoid(linear_model)

            error = y_predicted - y_one_hot
            dw = (X_checked.T @ error) / num_samples
            db = np.sum(error, axis=0) / num_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes_[predicted_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_checked = np.asarray(X, dtype=np.float64)
        linear_model = X_checked @ self.weights + self.bias
        y_predicted = self._sigmoid(linear_model)
        return y_predicted

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Apply a numerically stable sigmoid function element-wise.

        Args:
            z: Input array of shape (n_samples, n_classes).
        Returns:
            Probabilities of shape (n_samples, n_classes), each value in (0, 1).
        """
        z = np.asarray(z, dtype=np.float64)
        positive_mask = z >= 0
        negative_mask = ~positive_mask
        result = np.empty_like(z, dtype=np.float64)
        result[positive_mask] = 1 / (1 + np.exp(-z[positive_mask]))
        exp_z = np.exp(z[negative_mask])
        result[negative_mask] = exp_z / (1 + exp_z)
        return result

    def _cross_entropy_loss(self, y: np.ndarray, probabilities: np.ndarray) -> float:
        """

        this one has to know which class we are working with
        """
        eps = 1e-12
        clipped_probabilities = np.clip(probabilities, eps, 1.0)
        data_loss = -np.mean(
            np.sum(
                y * np.log(clipped_probabilities)
                + (1 - y) * np.log(1 - clipped_probabilities),
                axis=1,
            )
        )
        return float(data_loss)
    


    # def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
    #     """
    #     Compute the multiclass binary cross-entropy loss for the given data.
    #     """
    #     X_checked = np.asarray(X, dtype=np.float64)
    #     y_checked = np.asarray(y)
    #     class_to_index = {label: index for index, label in enumerate(self.classes_)}
    #     y_encoded = np.array([class_to_index[label] for label in y_checked], dtype=int)

    #     y_one_hot = np.eye(self.classes_.shape[0], dtype=np.float64)[y_encoded]
    #     probabilities = self.predict_proba(X_checked)
    #     return self._cross_entropy_loss(y_one_hot, probabilities)
        