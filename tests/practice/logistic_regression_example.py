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
        

import numpy as np

class LogisticRegressionModel:
    """
    Logistic Regression model for multi-class classification.

    Attributes:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for training.
        weights (np.ndarray | None): 
            The weights of the model. Shape (n_features, n_classes).
            e.g. [[w11, w12, ..., w1f], [w21, w22, ..., w2f], ..., [wc1, wc2, ..., wcf]]

        bias (np.ndarray | None): 
            The bias of the model. Shape (n_classes,).
            e.g. [b1, b2, ..., bc]
        
        class_labels (np.ndarray | None): 
            The unique class labels. Shape (n_classes,).
            e.g. ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
    """
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights: np.ndarray | None = None # shape (n_features, n_classes).
        self.bias: np.ndarray | None = None # shape (n_classes,)
        self.class_labels: np.ndarray | None = None # shape (n_classes,)

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input features.

        args:
            X: Data point. shape (n_samples, n_features). 
                E.g. [[1.0, 2.0, ..., 3.0], [4.0, 5.0, ..., 6.0], ...]

        returns:
            predicted probability of each class. 
                shape: n_samples, n_classes
                example: [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], ...]
                r[s][c]: the predicted probability of the s-th sample point belonging to the c-th class.
        """
        # linear_model: type(np.ndarray), shape(n_samples, n_classes)
        # 
        # e.g. [[z11, z12, ..., z1c], [z21, z22, ..., z2c], ...]
        # so zij is the linear combination of the features for the i-th data point and j-th class. 
        # z11 = w11*x1 + w12*x2 + ... + w1f*xf + b1
        linear_model = np.dot(X, self.weights) + self.bias # TODO: weights assumed to be shape (n_features, n_classes)

        return self.sigmoid_activation(linear_model)    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input features.

        args: 
            X: Data point. shape (n_samples, n_features). 
                E.g. [[1.0, 2.0, ..., 3.0], [4.0, 5.0, ..., 6.0], ...]
            classes: Array of class labels. shape (n_classes,).
                E.g. ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]

        returns:
            Array of predicted class labels. E.g. ["Gryffindor", "Slytherin", "Gryffindor", ...]
        """
        predicted_probabilities = self.predict_probability(X) # [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], ...]

        # [self.class_labels[np.argmax(probabilities)] for probabilities in predicted_probabilities]
        return self.class_labels[np.argmax(predicted_probabilities, axis=1)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Train the logistic regression model using gradient descent.

        args: 
            X: Data points. shape (n_samples, n_features). 
                E.g. [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], ...]
            y: Target labels. shape (n_samples,). 
                E.g. ["Gryffindor", "Slytherin", "Gryffindor", ...]
        """
        # Initialize weights and bias
        n_samples, n_features = X.shape

        self.class_labels = np.unique(y)
        self.weights = np.zeros((n_features, len(self.class_labels)))
        self.bias = np.zeros(len(self.class_labels))

        # Convert y to binary labels for each class
        y_binary = (y.reshape(n_samples, 1) == self.class_labels).astype(int) # shape (n_samples, n_classes)

        for _ in range(self.num_iterations):
            weight_learning_step, bias_learning_step = self.calculate_learning_step(X, y_binary)

            self.weights -= self.learning_rate * weight_learning_step
            self.bias -= self.learning_rate * bias_learning_step
                
    def calculate_learning_step(self, X: np.ndarray, y_binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the gradient of the loss function

        formula:
            ∂J/∂w = (1/m) * X.T @ (hθ(x) - y)   -> shape (n_features, n_classes)
            ∂J/∂b = (1/m) * sum(hθ(x) - y)       -> shape (n_classes,)

        args:
            X: shape (n_samples, n_features)
            y_binary: shape (n_samples, n_classes). Binary labels. E.g. [[1, 0, 0, 1, ...], [0, 1, 0, 0, ...], ...]

        returns:
            weight_learning_step: shape (n_features, n_classes)
            bias_learning_step: shape (n_classes,)
        """
        error = self.predict_probability(X) - y_binary # shape (n_samples, n_classes)

        weight_learning_step = np.dot(X.T, error) / X.shape[0] # shape (n_features, n_classes)
        bias_learning_step = np.mean(error, axis=0) # shape (n_classes,)

        return weight_learning_step, bias_learning_step


    def sigmoid_activation(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        comments: 
            - Basically map the linear combination of features and weights to a value between 0 and 1, which can be interpreted as a probability.
            - The sigmoid function is defined as: sigmoid(z) = 1 / (1 + exp(-z))

        args:
            z: Linear combination of features and weights. shape (n_samples, n_classes). 
                E.g. [[z11, z12, ..., z1c], [z21, z22, ..., z2c], ...]
        returns:
            Sigmoid of z. shape (n_samples, n_classes). 
                E.g. [[p11, p12, ..., p1c], [p21, p22, ..., p2c], ...]
        """
        return 1 / (1 + np.exp(-z))

