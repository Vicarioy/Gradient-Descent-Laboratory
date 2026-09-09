"""Matplotlib visualizations for the gradient descent laboratory."""

import matplotlib.pyplot as plt
import numpy as np


def plot_loss(experiment, ax=None):
    """Plot loss by epoch and return the Matplotlib axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    frame = experiment.loss_frame()
    ax.plot(frame["epoch"], frame["loss"], color="#d97706", linewidth=2)
    ax.set_title(f"{experiment.model_type}: loss by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    return ax


def plot_optimization_path(experiment, ax=None):
    """Plot every collected parameter value across the optimization path."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    frame = experiment.path_frame()
    theta_columns = [column for column in frame if column.startswith("theta_")]
    for column in theta_columns:
        ax.plot(frame["step"], frame[column], label=column)
    ax.set_title(f"{experiment.model_type}: parameter path")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Parameter value")
    ax.grid(alpha=0.25)
    if len(theta_columns) <= 8:
        ax.legend(loc="best")
    return ax


def plot_predictions(experiment, ax=None):
    """Plot observed and predicted values for regression experiments."""
    if experiment.model_type != "Regression":
        raise ValueError("plot_predictions only supports regression experiments")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    features = np.column_stack((np.ones(experiment.X.shape[0]), experiment.X))
    predictions = features @ experiment.theta
    order = np.argsort(experiment.X[:, 0])
    ax.scatter(experiment.X[:, 0], experiment.y, alpha=0.65, label="Observed")
    ax.plot(
        experiment.X[order, 0],
        predictions[order],
        color="#2563eb",
        linewidth=2,
        label="Model",
    )
    ax.set_title("Regression fit")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return ax


def plot_classification_data(experiment, ax=None):
    """Plot two-dimensional classification samples colored by their labels."""
    if experiment.model_type == "Regression":
        raise ValueError(
            "plot_classification_data requires a classification experiment"
        )
    if experiment.X.shape[1] != 2:
        raise ValueError("classification visualization requires exactly two features")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(
        experiment.X[:, 0],
        experiment.X[:, 1],
        c=experiment.y,
        cmap="viridis",
        alpha=0.75,
    )
    ax.set_title(f"{experiment.model_type}: dataset")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.25)
    return ax


__all__ = [
    "plot_classification_data",
    "plot_loss",
    "plot_optimization_path",
    "plot_predictions",
]
