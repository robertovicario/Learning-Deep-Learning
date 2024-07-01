# Model Training

## Overview

...

## Training Process

...

### Initialization

### 1. **Initialization:**
   - **Weights and Biases Initialization:** The network's weights and biases are initialized to small random values. This randomness helps to break symmetry and ensure that neurons learn different features.
  
### 2. **Forward Propagation:**
   - **Input to Output:** Data is fed into the network, and calculations are made from the input layer through the hidden layers to the output layer. Each neuron applies a weighted sum of its inputs, passes it through an activation function, and forwards the result to the next layer.
   - **Activation Functions:** Functions like ReLU (Rectified Linear Unit), sigmoid, or tanh are used to introduce non-linearity into the network.

### 3. **Loss Function:**
   - **Calculating the Error:** The output of the network is compared to the actual target values using a loss function (such as Mean Squared Error for regression tasks or Cross-Entropy Loss for classification tasks). The loss function measures how far the network's predictions are from the true values.

### 4. **Backward Propagation (Backpropagation):**
   - **Gradient Calculation:** The gradients of the loss function with respect to each weight are calculated using the chain rule of calculus. This process involves propagating the error backwards through the network from the output layer to the input layer.
   - **Error Propagation:** The error is distributed back through the network, with each weight receiving a share of the error proportional to its contribution to the final output.

### 5. **Gradient Descent Optimization:**
   - **Weight Update:** The network's weights are updated in the opposite direction of the gradient to minimize the loss function. The update rule for the weights \( w \) is typically:
     \[
     w = w - \eta \cdot \nabla L
     \]
     where \( \eta \) is the learning rate, and \( \nabla L \) is the gradient of the loss function with respect to the weights.
   - **Learning Rate:** The learning rate \( \eta \) controls the size of the steps taken towards the minimum of the loss function. Choosing the right learning rate is crucial for the convergence and efficiency of the learning process.

### 6. **Iteration and Convergence:**
   - **Epochs:** The process of forward and backward propagation is repeated for many epochs (complete passes through the training dataset). 
   - **Convergence:** The training process continues until the loss function converges to a minimum value, indicating that the network has learned the underlying pattern in the data.

### Variations and Improvements:
- **Mini-Batch Gradient Descent:** Instead of using the entire dataset to compute gradients, mini-batch gradient descent uses small random batches of data, combining the benefits of both batch gradient descent and stochastic gradient descent.
- **Optimizers:** Advanced optimization algorithms like Adam, RMSprop, and AdaGrad adaptively adjust the learning rate and improve convergence speed.
- **Regularization:** Techniques like L1/L2 regularization, dropout, and batch normalization are used to prevent overfitting and improve generalization.

### Example of a Basic Training Loop in Pseudocode:

```python
for epoch in range(num_epochs):
    for mini_batch in get_mini_batches(training_data, batch_size):
        # Forward propagation
        outputs = forward_propagate(mini_batch)
        
        # Compute loss
        loss = compute_loss(outputs, targets)
        
        # Backward propagation
        gradients = backward_propagate(loss)
        
        # Update weights
        update_weights(gradients, learning_rate)
```

### Summary:
The learning algorithm in neural networks involves initializing weights, performing forward propagation to compute outputs, calculating the loss, using backpropagation to compute gradients, and updating the weights using gradient descent. This process iterates until the network's performance stabilizes, achieving minimal error on the training data.

## Reference

- [Stanford CS230 Deep Learning Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230) by Afshine Amidi and Shervine Amidi
