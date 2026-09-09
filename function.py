import numpy as np

from gradient import optimize


def add_intercept(X):
    """Return a 2-D feature matrix with a leading bias column."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 1-D or 2-D array")
    return np.column_stack((np.ones(X.shape[0]), X))


def sigmoid(values):
    """Compute sigmoid without overflow for large positive or negative values."""
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return result


def softmax(scores):
    """Convert class scores to probabilities along the class axis."""
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)


def regression_cost(X, y, theta):
    predictions = np.asarray(X) @ theta
    residuals = predictions - np.asarray(y)
    return float(np.mean(residuals**2) / 2)


def regression_gradient(X, y, theta):
    X = np.asarray(X)
    residuals = X @ theta - np.asarray(y)
    return X.T @ residuals / X.shape[0]


def binary_cross_entropy_cost(X, y, theta):
    probabilities = np.clip(sigmoid(np.asarray(X) @ theta), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=float).reshape(-1)
    return float(
        -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
    )


def binary_cross_entropy_gradient(X, y, theta):
    probabilities = sigmoid(np.asarray(X) @ theta)
    y = np.asarray(y, dtype=float).reshape(-1)
    return np.asarray(X).T @ (probabilities - y) / len(y)


def multiclass_cross_entropy_cost(X, y, theta):
    probabilities = np.clip(softmax(np.asarray(X) @ theta), 1e-15, 1)
    labels = np.asarray(y)
    if labels.ndim == 1:
        return float(
            -np.mean(np.log(probabilities[np.arange(len(labels)), labels.astype(int)]))
        )
    return float(-np.mean(np.sum(labels * np.log(probabilities), axis=1)))


def multiclass_cross_entropy_gradient(X, y, theta):
    X = np.asarray(X)
    probabilities = softmax(X @ theta)
    labels = np.asarray(y)
    if labels.ndim == 1:
        encoded = np.zeros_like(probabilities)
        encoded[np.arange(len(labels)), labels.astype(int)] = 1
        labels = encoded
    return X.T @ (probabilities - labels) / X.shape[0]


def fit_gradient(
    X,
    y,
    initial_theta,
    cost_func,
    grad_func,
    *,
    add_bias=True,
    **optimizer_options,
):
    """Fit any model whose cost and gradient callbacks use ``(X, y, theta)``."""
    features = add_intercept(X) if add_bias else np.asarray(X, dtype=float)
    return optimize(
        features,
        y,
        initial_theta,
        cost_func,
        grad_func,
        **optimizer_options,
    )


def predict_binary(X, theta, *, add_bias=True):
    features = add_intercept(X) if add_bias else np.asarray(X, dtype=float)
    return (sigmoid(features @ theta) >= 0.5).astype(int)


def predict_multiclass(X, theta, *, add_bias=True):
    features = add_intercept(X) if add_bias else np.asarray(X, dtype=float)
    return np.argmax(softmax(features @ theta), axis=1)
