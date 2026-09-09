import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as LibraryLinearRegression
from sklearn.linear_model import LogisticRegression as LibraryLogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from classification_models import LogisticRegression, SoftmaxRegression
from function import regression_cost, regression_gradient
from gradient import optimize

MODEL_TYPES = (
    "Regression",
    "Binary classification",
    "Multiclass classification",
)


def make_dataset(model_type, sample_count=180, noise=0.5, random_state=None):
    """Create a reproducible dataset for one laboratory experiment."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"unknown model type: {model_type}")
    if sample_count < 3:
        raise ValueError("sample_count must be at least 3")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    rng = np.random.default_rng(random_state)
    if model_type == "Regression":
        X = rng.uniform(-4, 4, size=(sample_count, 1))
        y = 1.5 + 2.2 * X[:, 0] + rng.normal(0, noise, sample_count)
        return X, y

    if model_type == "Binary classification":
        half = sample_count // 2
        left = rng.normal((-1.5, -1.0), noise, size=(half, 2))
        right = rng.normal((1.5, 1.0), noise, size=(sample_count - half, 2))
        return np.vstack((left, right)), np.concatenate(
            (np.zeros(half, dtype=int), np.ones(sample_count - half, dtype=int))
        )

    centers = np.array([(-2.0, -1.0), (2.0, -1.0), (0.0, 2.0)])
    labels = np.arange(3).repeat(sample_count // 3)
    remainder = sample_count - labels.size
    if remainder:
        labels = np.concatenate((labels, np.arange(remainder)))
    X = np.vstack([rng.normal(centers[label], noise, size=(1, 2)) for label in labels])
    return X, labels


def compare_with_library(
    model_type,
    sample_count=300,
    noise=0.5,
    learning_rate=0.1,
    epochs=500,
    batch_size=32,
    mode="mini_batch",
    test_size=0.25,
    random_state=7,
):
    """Compare a custom model with its scikit-learn equivalent.

    Both implementations receive the same deterministic train/test split.
    The returned frame is suitable for a report, table, or frontend chart.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    X, y = make_dataset(model_type, sample_count, noise, random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )

    if model_type == "Regression":
        custom = ComparisonExperiment(
            X_train,
            y_train,
            model_type,
            learning_rate,
            epochs,
            batch_size,
            mode,
            random_state,
        ).execute()
        library = LibraryLinearRegression().fit(X_train, y_train)
        custom_predictions = _regression_predictions(custom, X_test)
        library_predictions = library.predict(X_test)
        custom_loss = custom.history[-1]
        library_loss = float(np.mean((library.predict(X_train) - y_train) ** 2) / 2)
        custom_score = _r_squared(y_test, custom_predictions)
        library_score = library.score(X_test, y_test)
    else:
        if model_type == "Binary classification":
            custom = LogisticRegression(
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                mode=mode,
                random_state=random_state,
                collect_path=True,
            ).fit(X_train, y_train)
            library = LibraryLogisticRegression(max_iter=2000).fit(X_train, y_train)
        elif model_type == "Multiclass classification":
            custom = SoftmaxRegression(
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                mode=mode,
                random_state=random_state,
                collect_path=True,
            ).fit(X_train, y_train)
            library = LibraryLogisticRegression(max_iter=2000).fit(X_train, y_train)
        else:
            raise ValueError(f"unknown model type: {model_type}")

        custom_predictions = custom.predict(X_test)
        library_predictions = library.predict(X_test)
        custom_score = float(np.mean(custom_predictions == y_test))
        library_score = float(np.mean(library_predictions == y_test))
        custom_loss = custom.history[-1]
        library_loss = float(log_loss(y_train, library.predict_proba(X_train)))

    return pd.DataFrame(
        [
            {
                "model": model_type,
                "implementation": "Custom gradient descent",
                "test_score": custom_score,
                "final_train_loss": custom_loss,
                "prediction_count": len(custom_predictions),
            },
            {
                "model": model_type,
                "implementation": "scikit-learn",
                "test_score": library_score,
                "final_train_loss": library_loss,
                "prediction_count": len(library_predictions),
            },
        ]
    )


def _regression_predictions(experiment, X):
    features = np.column_stack((np.ones(X.shape[0]), X))
    return features @ experiment.theta


def _r_squared(targets, predictions):
    denominator = np.sum((targets - targets.mean()) ** 2)
    if denominator == 0:
        return 1.0 if np.allclose(targets, predictions) else 0.0
    return float(1 - np.sum((targets - predictions) ** 2) / denominator)


class ComparisonExperiment:
    """Train one model and retain metrics and optimization-path data."""

    def __init__(
        self,
        X,
        y,
        model_type,
        learning_rate=0.1,
        epochs=500,
        batch_size=32,
        mode="mini_batch",
        random_state=None,
    ):
        if model_type not in MODEL_TYPES:
            raise ValueError(f"unknown model type: {model_type}")
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.mode = mode
        self.random_state = random_state
        self.model = None
        self.theta = None
        self.history = None
        self.optimization_path = None

    def execute(self):
        if self.model_type == "Regression":
            features = np.column_stack((np.ones(self.X.shape[0]), self.X))
            self.theta, self.history, self.optimization_path = optimize(
                features,
                self.y,
                np.zeros(features.shape[1]),
                regression_cost,
                regression_gradient,
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                batch_size=self.batch_size,
                mode=self.mode,
                random_state=self.random_state,
                collect_path=True,
            )
        elif self.model_type == "Binary classification":
            self.model = LogisticRegression(
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                batch_size=self.batch_size,
                mode=self.mode,
                random_state=self.random_state,
                collect_path=True,
            ).fit(self.X, self.y)
            self.theta = self.model.theta
            self.history = self.model.history
            self.optimization_path = self.model.optimization_path
        else:
            self.model = SoftmaxRegression(
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                batch_size=self.batch_size,
                mode=self.mode,
                random_state=self.random_state,
                collect_path=True,
            ).fit(self.X, self.y)
            self.theta = self.model.theta
            self.history = self.model.history
            self.optimization_path = self.model.optimization_path
        return self

    def score(self):
        """Return R-squared for regression or accuracy for classification."""
        self._check_executed()
        if self.model_type == "Regression":
            features = np.column_stack((np.ones(self.X.shape[0]), self.X))
            predictions = features @ self.theta
            denominator = np.sum((self.y - self.y.mean()) ** 2)
            if denominator == 0:
                return 1.0 if np.allclose(self.y, predictions) else 0.0
            return float(1 - np.sum((self.y - predictions) ** 2) / denominator)
        return float(self.model.score(self.X, self.y))

    def loss_frame(self):
        self._check_executed()
        return pd.DataFrame(
            {"epoch": np.arange(1, len(self.history) + 1), "loss": self.history}
        )

    def path_frame(self):
        self._check_executed()
        values = self.optimization_path.reshape(self.optimization_path.shape[0], -1)
        return pd.DataFrame(
            {
                "step": np.arange(values.shape[0]),
                **{
                    f"theta_{index}": values[:, index]
                    for index in range(values.shape[1])
                },
            }
        )

    def _check_executed(self):
        if self.theta is None or self.history is None:
            raise RuntimeError("execute the experiment before requesting results")


__all__ = [
    "MODEL_TYPES",
    "ComparisonExperiment",
    "compare_with_library",
    "make_dataset",
]
