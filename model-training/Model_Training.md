# Model Training

## Overview

Model training involves finding the optimal parameters that minimize the error of the model's predictions. This process typically includes initializing the model's parameters, propagating the input data through the model to make predictions, computing the loss between the predictions and actual values, and updating the parameters based on the computed gradients.

## Initialization

...

### Xavier Initialization

...

Mathematically:

$$...$$

## Forward Propagation

Forward propagation involves passing the input data through the model's layers to obtain the output predictions. Each layer applies a transformation to the input data, such as a linear transformation followed by a non-linear activation function.

Mathematically, for a single layer:

$$z = Wx + b$$

$$y = \sigma(z)$$

where $W$ and $b$ are the weights and biases, $x$ is the input, $z$ is the linear transformation, $\sigma$ is the activation function, and $y$ is the output of the layer.

## Loss Function


The loss function measures the difference between the predicted values and the actual values.

### Cross-Entropy Loss

...

Mathematically:

$$L = -\frac{1}{m} \sum_{i=1}^m \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### Mean Squared Error (MSE)

...

Mathematically:

$$L = \frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2$$

where $y_i$ is the actual value, $\hat{y}_i$ is the predicted value, and $m$ is the number of samples.

## Backward Propagation

Backward propagation involves computing the gradients of the loss function with respect to the model's parameters and using these gradients to update the parameters. This process is essential for minimizing the loss function.

Mathematically, using the chain rule:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial W}$$

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial b}$$

<table>
    <tr>
        <td><img src="/model-traning/img/1.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Backward Propagation</td>
    </tr>
</table>

### Weight Update

During weight update, the model's parameters are adjusted in the direction that reduces the loss. This adjustment is typically done using an optimization algorithm such as Gradient Descent.

Mathematically:

$$
W = W - \alpha \frac{\partial L}{\partial W}
$$

$$
b = b - \alpha \frac{\partial L}{\partial b}
$$

where $\alpha$ is the learning rate.

<table>
    <tr>
        <td><img src="/model-traning/img/2.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Weight Update</td>
    </tr>
</table>

## Gradient Descent Optimization

Gradient Descent is an optimization algorithm used to minimize the loss function by iteratively updating the model's parameters.

Mathematically, the general update rule for Gradient Descent is:

$$\theta = \theta - \alpha \nabla_\theta J(\theta)$$

where $\theta$ represents the model's parameters, $\alpha$ is the learning rate, and $J(\theta)$ is the loss function.

### Mini-batch

Mini-Batch splits the training data into small batches and performs an update for each batch. This approach balances the benefits of both SGD and Batch Gradient Descent.

Mathematically, the update rule for Mini-batch is:

$$\theta = \theta - \alpha \nabla_\theta J(\theta; x^{(i:i+n)}, y^{(i:i+n)})$$

where $x^{(i:i+n)}$ and $y^{(i:i+n)}$ are the input and output batches, respectively.

## Iteration and Convergence

...

- **Epochs:** ...

- **Batch Size:** ...

- **Learning Rate:** ...

- **Model Complexity:** ...

## Optimization

Optimization involves techniques that improve the efficiency and effectiveness of the training process.

### Momentum

Momentum helps accelerate the gradient descent algorithm by adding a fraction of the previous update to the current update.

### RMSprop

RMSprop adjusts the learning rate for each parameter based on the magnitude of recent gradients, helping to address issues with the oscillations in the parameter updates.

### Adam

Adam combines the benefits of both Momentum and RMSprop, providing an adaptive learning rate for each parameter and accelerating convergence.







































## Backward Propagation

Backward propagation involves computing the gradients of the loss function with respect to the model's parameters and using these gradients to update the parameters. This process is essential for minimizing the loss function.

Mathematically, using the chain rule:

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial W}
$$

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial b}
$$

<table>
    <tr>
        <td><img src="/model-traning/img/1.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Backward Propagation</td>
    </tr>
</table>

### Weight Update

During weight update, the model's parameters are adjusted in the direction that reduces the loss. This adjustment is typically done using an optimization algorithm such as Gradient Descent.

Mathematically:

$$
W = W - \alpha \frac{\partial L}{\partial W}
$$

$$
b = b - \alpha \frac{\partial L}{\partial b}
$$

where $\alpha$ is the learning rate.

<table>
    <tr>
        <td><img src="/model-traning/img/2.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Weight Update</td>
    </tr>
</table>


























## Regularization

Regularization is a set of techniques used to prevent overfitting in machine learning models by introducing additional information or constraints. These techniques aim to improve the generalization performance of the model, ensuring that it performs well on new, unseen data.

### Dropout

Dropout is a regularization technique where randomly selected neurons are ignored during training. This means that their contribution to the activation of downstream neurons is temporarily removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.

<table>
    <tr>
        <td><img src="/model-traning/img/3.png" width="512"></td>
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
        <td><img src="/model-traning/img/4.png" width="256"></td>
        <td><img src="/model-traning/img/5.png" width="256"></td>
    </tr>
    <tr>
        <td align="center">L1 (Lasso)</td>
        <td align="center">L2 (Ridge)</td>
    </tr>
</table>

#### TensorFlow Implementation

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
        <td><img src="/model-traning/img/6.png" width="512"></td>
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

- [Stanford CS230 Deep Learning Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230) by Afshine Amidi and Shervine Amidi
