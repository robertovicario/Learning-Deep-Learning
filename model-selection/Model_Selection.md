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

## Regularization

Regularization is a set of techniques used to prevent overfitting in machine learning models by introducing additional information or constraints. These techniques aim to improve the generalization performance of the model, ensuring that it performs well on new, unseen data.

### Dropout

Dropout is a regularization technique where randomly selected neurons are ignored during training. This means that their contribution to the activation of downstream neurons is temporarily removed on the forward pass and any weight updates are not applied to the neuron on the backward pass. This helps prevent overfitting and provides a way of approximately combining exponentially many different neural network architectures efficiently.

<table>
    <tr>
        <td><img src="/model-selection/img/3.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Dropout</td>
    </tr>
</table>

#### TensorFlow Implementation

```py
# Hidden Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout
model.add(Dense(64, activation='relu'))
```

### Weight Regularization

Weight regularization involves adding a penalty to the loss function to constrain the size of the weights. This can be done using L1 regularization (Lasso), which adds the absolute value of the weights to the loss function, or L2 regularization (Ridge), which adds the squared value of the weights to the loss function.

<table>
    <tr>
        <td><img src="/model-selection/img/4.png" width="256"></td>
        <td><img src="/model-selection/img/5.png" width="256"></td>
    </tr>
    <tr>
        <td align="center">L1 (Lasso)</td>
        <td align="center">L2 (Ridge)</td>
    </tr>
</table>

#### TensorFlow Implementation:

```py
# Hidden Layers
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))  # Weight Regularization
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))  # Weight Regularization

# Output Layer
model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01)))  # Weight Regularization
```

### Early Stopping

Early stopping is a technique to stop training the neural network once the performance on a validation dataset starts to degrade. This helps prevent overfitting as the model stops training when it starts to overfit the training data.

<table>
    <tr>
        <td><img src="/model-selection/img/6.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Early Stopping</td>
    </tr>
</table>

#### TensorFlow Implementation

```py
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Early Stopping
```

## Reference

- [Stanford CS229 Machine Learning Cheatsheet](https://stanford.edu/~shervine/teaching/cs-229) by Afshine Amidi and Shervine Amidi
