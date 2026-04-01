import numpy as np

class LogisticRegressionModel:
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model using gradient descent.

        Initializes weights and bias to zero, then iteratively updates them based on the gradient 
        of the loss function with respect to the weights and bias. The number of iterations and 
        learning rate control the training process.

        Args:
            X: Input feature matrix of shape (n_samples, n_features).
            y: Target binary labels of shape (n_samples,).
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels for input features.
        
        Uses the learned weights and bias to compute the linear combination,
        applies sigmoid activation function, and converts probabilities to
        binary class labels (0 or 1) using a 0.5 threshold.
        
        Args:
            X: Input feature matrix of shape (n_samples, n_features).
        
        Returns:
            Array of binary class predictions (0 or 1) for each sample.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)