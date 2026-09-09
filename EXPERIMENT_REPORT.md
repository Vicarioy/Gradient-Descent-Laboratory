# Experiment Report: Gradient Descent Compared with scikit-learn

## Objective

This experiment evaluates whether the custom gradient-descent implementations can learn the same basic relationships as established scikit-learn estimators. The comparison covers linear regression, binary logistic regression, and multiclass softmax regression.

## Method

Synthetic datasets were generated with random seed `7`, `300` observations, and noise level `0.5`. Each dataset was split into 75% training data and 25% test data using the same deterministic split for both implementations. The custom models used mini-batch gradient descent with learning rate `0.1`, batch size `32`, and `300` epochs.

The custom implementations were compared with `LinearRegression` and `LogisticRegression` from scikit-learn. Regression was evaluated with $R^2$ on the test set. Classification was evaluated with accuracy. Training loss was also recorded: half mean squared error for regression and cross-entropy for classification.

## Results

| Task | Custom test score | scikit-learn test score | Custom final train loss | Library final train loss |
| --- | ---: | ---: | ---: | ---: |
| Regression | 0.9893 | 0.9894 | 0.1024 | 0.0998 |
| Binary classification | 1.0000 | 1.0000 | 0.0028 | 0.0102 |
| Multiclass classification | 1.0000 | 1.0000 | 0.0042 | 0.0136 |

## Interpretation

The custom models matched the library models closely on these datasets. Regression produced nearly identical $R^2$ scores, with scikit-learn slightly ahead because its closed-form/optimized solver reaches the least-squares solution directly. Both classifiers achieved perfect test accuracy because the generated classes are well separated. The low cross-entropy values show that the custom classifiers also learned confident class probabilities rather than only correct class labels.

These results validate the core implementation: feature handling, gradients, update modes, loss calculation, and prediction logic are working for the tested cases. The collected optimization path adds educational value because it makes parameter movement visible after every epoch.

## Limitations

This is not a performance benchmark. The datasets are small, synthetic, and easy to separate. Only one random seed and one train/test split were used. The custom optimizer does not yet include regularization, early stopping, learning-rate schedules, validation monitoring, or the solver alternatives available in scikit-learn. The comparison also measures predictive behavior, not production reliability or runtime at scale.

## Conclusion

The laboratory implementation is suitable for studying how gradient descent trains regression and classification models. The results are consistent with the scikit-learn reference implementations on the selected experiments. Further work will add repeated splits, real datasets, regularization, decision-boundary visualization, and automated tests before drawing broader conclusions.
