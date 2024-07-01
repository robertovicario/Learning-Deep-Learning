# Model Selection

## Overview

Model selection is a critical step in the machine learning pipeline. It involves choosing the best model and the best set of hyperparameters for a given task.

When training a machine learning model, the data is usually split into three subsets:

- **Training Set:** This is used to train the model.

- **Validation Set:** This is used to evaluate the model's performance during the training phase and to tune hyperparameters.

- **Test Set:** This is used to assess the final model's performance.

<table>
    <tr>
        <td><img src="/model-selection/img/1.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Data Subsets</td>
    </tr>
</table>

## Cross-Validation

Cross-validation is a technique used to evaluate the performance of a model more reliably by partitioning the data into multiple subsets and using each subset for both training and validation.

### K-fold Cross-Validation

The most common form of cross-validation is k-fold cross-validation, where the data is divided into k subsets (folds). The model is trained k times, each time using a different fold as the validation set and the remaining folds as the training set.

<table>
    <tr>
        <td><img src="/model-selection/img/2.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">K-Fold Cross-Validation</td>
    </tr>
</table>

### scikit-learn Implementation

```py
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kf)

print("Cross-Validation scores:", scores)
print("Average Cross-Validation score:", scores.mean())
```

## Reference

- [Stanford CS229 Machine Learning Cheatsheet](https://stanford.edu/~shervine/teaching/cs-229) by Afshine Amidi and Shervine Amidi
