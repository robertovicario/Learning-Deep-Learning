# Regularization

## Overview

Regularization is a set of techniques used to prevent overfitting in machine learning models by introducing additional information or constraints. These techniques aim to improve the generalization performance of the model, ensuring that it performs well on new, unseen data.

## Dropout

Dropout is a regularization technique where randomly selected neurons are ignored during training. This means that their contribution to the activation of downstream neurons is temporarily removed on the forward pass and any weight updates are not applied to the neuron on the backward pass. This helps prevent overfitting and provides a way of approximately combining exponentially many different neural network architectures efficiently.

<table>
    <tr>
        <td><img src="/regularization/img/1.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Dropout</td>
    </tr>
</table>

### TensorFlow Implementation

```py
# Hidden Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout Layer
model.add(Dense(64, activation='relu'))
```

## Weight Regularization

Weight regularization involves adding a penalty to the loss function to constrain the size of the weights. This can be done using L1 regularization (Lasso), which adds the absolute value of the weights to the loss function, or L2 regularization (Ridge), which adds the squared value of the weights to the loss function.

<table>
    <tr>
        <td><img src="/regularization/img/2.png" width="512"></td>
        <td><img src="/regularization/img/3.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">L1 (Lasso)</td>
        <td align="center">L2 (Ridge)</td>
    </tr>
</table>

### TensorFlow Implementation

```py
# Hidden Layers
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))  # Weight Regularization
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))  # Weight Regularization

# Output Layer
model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01)))  # Weight Regularization
```

## Early Stopping

Early stopping is a technique to stop training the neural network once the performance on a validation dataset starts to degrade. This helps prevent overfitting as the model stops training when it starts to overfit the training data.

<table>
    <tr>
        <td><img src="/regularization/img/4.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Early Stopping</td>
    </tr>
</table>

### TensorFlow Implementation

```py
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Early Stopping
```

## Reference

- [github.com/afshinea/stanford-cs-230-deep-learning](https://github.com/afshinea/stanford-cs-230-deep-learning)
