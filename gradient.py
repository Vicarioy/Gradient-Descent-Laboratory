import numpy as np


def optimize(
    X,
    y,
    initial_theta,
    cost_func,
    grad_func,
    learning_rate=0.1,
    batch_size=32,
    epochs=500,
    mode="mini_batch",
    random_state=None,
    collect_path=False,
):
    """Minimize a user-provided cost function with gradient descent."""
    X = np.asarray(X)
    y = np.asarray(y)
    theta = np.asarray(initial_theta, dtype=float).copy()

    if X.ndim == 0 or y.ndim == 0 or X.shape[0] == 0:
        raise ValueError("X and y must contain at least one sample")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if theta.ndim == 0:
        raise ValueError("initial_theta must be an array with at least one dimension")

    if mode not in ("batch", "sgd", "mini_batch"):
        raise ValueError("mode must be 'batch', 'sgd', 'mini_batch' ")
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    sample_count = X.shape[0]
    if mode == "sgd":
        effective_batch_size = 1
    elif mode == "batch":
        effective_batch_size = sample_count
    else:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        effective_batch_size = min(batch_size, sample_count)

    rng = np.random.default_rng(random_state)
    history = []
    optimization_path = [theta.copy()] if collect_path else None

    for _ in range(epochs):
        indices = rng.permutation(sample_count)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, sample_count, effective_batch_size):
            end = start + effective_batch_size
            x_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            gradients = np.asarray(grad_func(x_batch, y_batch, theta))
            if gradients.shape != theta.shape:
                raise ValueError(
                    "grad_func must return an array matching initial_theta"
                )
            if not np.all(np.isfinite(gradients)):
                raise ValueError("grad_func returned non-finite values")
            theta -= learning_rate * gradients

        cost = float(cost_func(X, y, theta))
        if not np.isfinite(cost):
            raise ValueError("cost_func returned a non-finite value")
        history.append(cost)
        if collect_path:
            optimization_path.append(theta.copy())

    if collect_path:
        return theta, np.asarray(history), np.asarray(optimization_path)
    return theta, np.asarray(history)


def optimize_all(
    X,
    y,
    initial_theta,
    cost_func,
    grad_func,
    learning_rate=0.1,
    epochs=500,
    batch_size=32,
    random_state=None,
):
    """Run all update methods and return the lowest-cost result."""
    results = {}
    modes = ("batch", "sgd", "mini_batch")

    for mode_index, mode in enumerate(modes):
        theta, history = optimize(
            X,
            y,
            initial_theta,
            cost_func,
            grad_func,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            mode=mode,
            random_state=None if random_state is None else random_state + mode_index,
        )
        results[mode] = {
            "theta": theta,
            "history": history,
            "final_cost": history[-1],
        }

    best_mode = min(results, key=lambda mode: results[mode]["final_cost"])
    return best_mode, results[best_mode]["theta"], results
