import numpy as np

from function import (
    add_intercept,
    binary_cross_entropy_cost,
    binary_cross_entropy_gradient,
    multiclass_cross_entropy_cost,
    multiclass_cross_entropy_gradient,
    predict_binary,
    predict_multiclass,
    sigmoid,
    softmax,
)
from gradient import optimize


def sigmoid_function(values):
    """Return sigmoid probabilities for the supplied scores."""
    return sigmoid(values)


def compute_logistic_cost(X, y, theta):
    """Return binary cross-entropy for logistic regression."""
    return binary_cross_entropy_cost(X, y, theta)


def compute_logistic_gradient(X, y, theta):
    """Return the binary logistic-regression gradient."""
    return binary_cross_entropy_gradient(X, y, theta)


def compute_multiclass_cost(X, y, theta):
    """Return multiclass cross-entropy for softmax regression."""
    return multiclass_cross_entropy_cost(X, y, theta)


def compute_multiclass_gradient(X, y, theta):
    """Return the multiclass softmax-regression gradient."""
    return multiclass_cross_entropy_gradient(X, y, theta)


class LogisticRegression:
    """Binary logistic regression trained with gradient descent."""

    def __init__(
        self,
        learning_rate=0.1,
        epochs=500,
        batch_size=32,
        mode="mini_batch",
        add_bias=True,
        random_state=None,
        collect_path=False,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.mode = mode
        self.add_bias = add_bias
        self.random_state = random_state
        self.collect_path = collect_path
        self.theta = None
        self.history = None
        self.optimization_path = None

    def fit(self, X, y):
        features = _prepare_features(X, self.add_bias)
        labels = np.asarray(y).reshape(-1)
        if features.shape[0] != labels.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if not np.all(np.isin(labels, (0, 1))):
            raise ValueError("binary labels must be 0 or 1")

        result = optimize(
            features,
            labels,
            np.zeros(features.shape[1]),
            compute_logistic_cost,
            compute_logistic_gradient,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            mode=self.mode,
            random_state=self.random_state,
            collect_path=self.collect_path,
        )
        if self.collect_path:
            self.theta, self.history, self.optimization_path = result
        else:
            self.theta, self.history = result
        return self

    def predict_proba(self, X):
        self._check_fitted()
        features = _prepare_features(X, self.add_bias)
        positive = sigmoid(features @ self.theta)
        return np.column_stack((1 - positive, positive))

    def predict(self, X):
        self._check_fitted()
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        labels = np.asarray(y).reshape(-1)
        return float(np.mean(self.predict(X) == labels))

    def _check_fitted(self):
        if self.theta is None:
            raise RuntimeError("call fit before prediction")


class SoftmaxRegression:
    """Multiclass softmax regression trained with gradient descent."""

    def __init__(
        self,
        learning_rate=0.1,
        epochs=500,
        batch_size=32,
        mode="mini_batch",
        add_bias=True,
        random_state=None,
        collect_path=False,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.mode = mode
        self.add_bias = add_bias
        self.random_state = random_state
        self.collect_path = collect_path
        self.theta = None
        self.classes_ = None
        self.history = None
        self.optimization_path = None

    def fit(self, X, y):
        features = _prepare_features(X, self.add_bias)
        labels = np.asarray(y).reshape(-1)
        if features.shape[0] != labels.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        self.classes_, encoded_labels = np.unique(labels, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("multiclass classification requires at least two classes")

        result = optimize(
            features,
            encoded_labels,
            np.zeros((features.shape[1], self.classes_.size)),
            compute_multiclass_cost,
            compute_multiclass_gradient,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            mode=self.mode,
            random_state=self.random_state,
            collect_path=self.collect_path,
        )
        if self.collect_path:
            self.theta, self.history, self.optimization_path = result
        else:
            self.theta, self.history = result
        return self

    def predict_proba(self, X):
        self._check_fitted()
        features = _prepare_features(X, self.add_bias)
        return softmax(features @ self.theta)

    def predict(self, X):
        self._check_fitted()
        encoded_predictions = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[encoded_predictions]

    def score(self, X, y):
        labels = np.asarray(y).reshape(-1)
        return float(np.mean(self.predict(X) == labels))

    def _check_fitted(self):
        if self.theta is None:
            raise RuntimeError("call fit before prediction")


def _prepare_features(X, add_bias):
    if add_bias:
        return add_intercept(X)
    features = np.asarray(X, dtype=float)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2:
        raise ValueError("X must be a 1-D or 2-D array")
    return features


__all__ = [
    "LogisticRegression",
    "SoftmaxRegression",
    "compute_logistic_cost",
    "compute_logistic_gradient",
    "compute_multiclass_cost",
    "compute_multiclass_gradient",
    "predict_binary",
    "predict_multiclass",
    "sigmoid_function",
]
