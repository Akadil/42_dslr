import numpy as np

class LogisticRegressionModel:
    """
    Logistic Regression model for multi-class classification.

    Attributes:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for training.
        weights (np.ndarray | None): 
            The weights of the model. Shape (n_classes, n_features).
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
        self.weights: np.ndarray | None = None # shape (n_classes, n_features).
        self.bias: np.ndarray | None = None # shape (n_classes,)
        self.class_labels: np.ndarray | None = None # shape (n_classes,)

    def predict_probability(self, X: np.ndarray, class_idx: int | None = None) -> np.ndarray:
        """
        Predict class labels for input features.

        args:
            X: Data point. shape (n_samples, n_features). 
                E.g. [[1.0, 2.0, ..., 3.0], [4.0, 5.0, ..., 6.0], ...]

        returns:
            Array of predicted probabilities for each class. shape (n_samples, n_classes)
                e.g. [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], ...]
        """
        # linear_model: type(np.ndarray), shape(n_samples, n_classes)
        # 
        # e.g. [[z11, z12, ..., z1c], [z21, z22, ..., z2c], ...]
        # so zij is the linear combination of the features for the i-th data point and j-th class. 
        # z11 = w11*x1 + w12*x2 + ... + w1f*xf + b1
        linear_model = np.dot(X, self.weights) + self.bias # TODO: weights assumed to be shape (n_features, n_classes)
        probabilities = self.sigmoid_activation(linear_model)

        return probabilities if class_idx is None else probabilities[:, class_idx]
    
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

        Process: 
            1. Initialize weights and bias.
            2. For each iteration:
                a. Compute linear model: z = X * weights + bias
                b. Apply sigmoid activation: predicted_probabilities = sigmoid(z)
                c. Compute gradients for weights and bias.
                d. Update weights and bias using the gradients and learning rate.

        args: 
            X: Data points. shape (n_samples, n_features). 
                E.g. [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], ...]
            y: Target labels. shape (n_samples,). 
                E.g. ["Gryffindor", "Slytherin", "Gryffindor", ...]
        """
        # Initialize weights and bias
        _, n_features = X.shape
        self.class_labels = np.unique(y) # e.g. ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
        self.weights = np.zeros((len(self.class_labels), n_features))
        self.bias = np.zeros(len(self.class_labels))

        # Iterate each class and perform One-vs-Rest training
        for label_idx, label in enumerate(self.class_labels): # label = "Gryffindor", label_idx = 0, 
            # One-vs-Rest training
            y_binary = np.where(y == label, 1, 0) # e.g. [1, 0, 1, ...]

            for _ in range(self.num_iterations):
                """
                calculate new weights and bias 
                """
                probabilities = self.predict_probability(X, class_idx=label_idx) # of each data point. shape (n_samples,). e.g. [0.8, 0.3, 0.9, ...]

                weight_learning_step = np.dot(X.T, (probabilities - y_binary)) / len(y_binary)
                bias_learning_step = np.mean(probabilities - y_binary) 

                self.weights[label_idx] -= self.learning_rate * weight_learning_step
                self.bias[label_idx] -= self.learning_rate * bias_learning_step
                

    def calculate_learning_step(self, X: np.ndarray, y_binary: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Calculate the gradient of the loss function with respect to weights and bias.

        formula:
            ∂J/∂w = (1/m) * X.T @ (hθ(x) - y)   -> shape (n_features,)
            ∂J/∂b = (1/m) * sum(hθ(x) - y)       -> scalar

        args:
            X: shape (n_samples, n_features)
            y_binary: shape (n_samples,). Binary labels for one class. E.g. [1, 0, 0, 1, ...]
            probabilities: shape (n_samples,). Predicted probabilities for one class. E.g. [0.8, 0.2, ...]

        returns:
            weight_learning_step: shape (n_features,)
            bias_learning_step: float
        """
        error = probabilities - y_binary
        weight_learning_step = np.dot(X.T, error) / len(y_binary)
        bias_learning_step = np.mean(error)

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
