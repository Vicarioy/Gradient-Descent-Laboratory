# Gradient Descent Laboratory

A small machine-learning laboratory for understanding gradient descent through implementation, visualization, and comparison with scikit-learn.

## What this project demonstrates

- Linear regression trained with gradient descent
- Binary logistic regression
- Multiclass softmax regression
- Batch, stochastic, and mini-batch updates
- Loss history and parameter optimization paths
- Matplotlib visualizations
- A Streamlit interface for interactive experiments
- Comparison against equivalent scikit-learn estimators

## Project structure

| File | Purpose |
| --- | --- |
| `function.py` | Shared numerical functions, losses, gradients, sigmoid, and softmax |
| `gradient.py` | General gradient-descent optimizer |
| `classification_models.py` | Logistic and softmax regression model classes |
| `compare.py` | Dataset generation, experiment execution, and scikit-learn comparison |
| `visualize.py` | Matplotlib figures for loss, parameter paths, fits, and datasets |
| `streamlit_app.py` | Interactive frontend |
| `EXPERIMENT_REPORT.md` | One-page experiment report |

## Setup

This project uses Python and a local virtual environment. From the project directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the Streamlit laboratory

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The application lets you select a model, choose an update strategy, execute training, inspect the loss curve, inspect the optimization path, view Matplotlib figures, and download the collected path.

## Compare against scikit-learn

The comparison uses the same generated data and deterministic train/test split for both implementations:

```powershell
.\.venv\Scripts\python.exe -c "from compare import compare_with_library; print(compare_with_library('Regression').to_string(index=False))"
```

Python usage:

```python
from compare import compare_with_library

results = compare_with_library(
    "Binary classification",
    sample_count=300,
    noise=0.5,
    learning_rate=0.1,
    epochs=300,
    mode="mini_batch",
    random_state=7,
)
print(results)
```

The returned DataFrame contains the implementation name, test score, final training loss, and number of test predictions.

## Notes on interpretation

The custom implementation is intended for learning and inspection. Scikit-learn is a production-quality reference implementation with mature numerical safeguards, solver choices, convergence handling, and broader feature support. Similar scores do not mean the implementations are identical internally.

The current experiments use synthetic, relatively clean datasets. Results should not be treated as a general benchmark for real-world machine-learning performance.

## License

Add the license you want to use before publishing this repository.
