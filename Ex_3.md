To solve the MLP (1), (2), and (3) questions from the document, we need to follow the specific tasks provided in each section. Here are the detailed solutions for each of these problems:

### MLP (1)
1. **Compute the number of trainable parameters for a MLP with layer sizes 100, 50, 20 that processes images of size 3x20x20.**

    - **Input layer to the first hidden layer:**
      $$
      \text{Number of parameters} = (3 \times 20 \times 20 + 1) \times 100 = (1200 + 1) \times 100 = 120100
      $$

    - **First hidden layer to the second hidden layer:**
      $$
      \text{Number of parameters} = (100 + 1) \times 50 = 101 \times 50 = 5050
      $$

    - **Second hidden layer to the output layer:**
      $$
      \text{Number of parameters} = (50 + 1) \times 20 = 51 \times 20 = 1020
      $$

    - **Total number of trainable parameters:**
      $$
      120100 + 5050 + 1020 = 126170
      $$

2. **Given a binary classification task for images of shape 3x64x64 with a network with a single hidden layer of 100 units. Compute how much RAM is needed to represent the trainable model parameters in memory (assume that each parameter is represented by float values of 64 bits).**

    - **Input layer to hidden layer:**
      $$
      \text{Number of parameters} = (3 \times 64 \times 64 + 1) \times 100 = (12288 + 1) \times 100 = 1228900
      $$

    - **Hidden layer to output layer:**
      $$
      \text{Number of parameters} = (100 + 1) \times 1 = 101
      $$

    - **Total number of parameters:**
      $$
      1228900 + 101 = 1229001
      $$

    - **Memory required (each float is 64 bits or 8 bytes):**
      $$
      \text{Memory in bytes} = 1229001 \times 8 = 9832008 \text{ bytes} \approx 9.38 \text{ MB}
      $$

3. **Compute the softmax values for the given matrix of logits (samples in the rows). Express the result with exponentials without summation symbol.**

    Given matrix of logits $L$:
    $$
    \text{Softmax}(L_{ij}) = \frac{e^{L_{ij}}}{\sum_{k} e^{L_{ik}}}
    $$
    Without the summation symbol, we express the softmax value for an element $L_{ij}$ as:
    $$
    \text{Softmax}(L_{ij}) = \frac{e^{L_{ij}}}{\text{Sum of exponentials of all logits in the same row}}
    $$

### MLP (2)
1. **Forward propagation for given weights and bias using different activation functions and determine the order of activations:**

    Given $ x = [4, 2, 1] $, $ w = [0.5, -0.1, 0] $, and $ b = 0.2 $:
    $$
    z = w \cdot x + b = (0.5 \times 4) + (-0.1 \times 2) + (0 \times 1) + 0.2 = 2 - 0.2 + 0 + 0.2 = 2
    $$

    For each activation function:
    - **Sigmoid:**
      $$
      a_1 = \sigma(z) = \frac{1}{1 + e^{-2}} \approx 0.8808
      $$
    - **ReLU:**
      $$
      a_2 = \max(0, z) = \max(0, 2) = 2
      $$
    - **Heaviside:**
      $$
      a_3 = \Theta(z) = 1 \text{ (since } z > 0)
      $$
    - **Linear:**
      $$
      a_4 = z = 2
      $$

    Order: $ a_1 \leq a_3 \leq a_2 = a_4 $.

2. **Importance of bias:**
    - Bias allows the model to fit the data better by providing an additional degree of freedom to shift the activation function. Without bias, the model might not capture patterns that do not pass through the origin.

3. **Random weight initialization and its scale:**
    - Random initialization breaks symmetry and ensures that neurons learn different features.
    - The scale of the initial weights is crucial to avoid vanishing or exploding gradients. Proper scaling helps maintain the gradients within a reasonable range during training.

### MLP (3)
1. **Shapes of the weight matrix and bias vector for a fully connected layer with input image shape 3x500x500 and 100 hidden units:**

    - **Input size:** $ 3 \times 500 \times 500 = 750000 $
    - **Weight matrix shape:** $ (100, 750000) $
    - **Bias vector shape:** $ (100, 1) $

2. **Parameters in a convolutional layer with 10 filters, kernel size 5x5, and input shape 3x500x500:**

    - **Number of parameters per filter:** $ 3 \times 5 \times 5 = 75 $
    - **Total number of parameters:** $ 10 \times 75 = 750 $

These solutions cover the problems specified in the MLP (1), (2), and (3) sections from the provided document.

### Batch-Normalization (1)
**Fill in the blanks — numpy code (numpy-like pseudo code) for computing a batch-normalised ReLU layer. Also specify the shape.**

Given:
- $Z$ is the input to the batch-normalization layer.
- $\mu$ is the mean of $Z$.
- $\sigma$ is the standard deviation of $Z$.
- $\gamma$ is the scale parameter.
- $\beta$ is the shift parameter.
- $Z_{\text{norm}}$ is the normalized $Z$.
- $Z_{\text{bn}}$ is the batch-normalized $Z$.
- $A$ is the activation after applying ReLU.

```python
import numpy as np

# Shapes (batchsize, n1)
Z = np.random.randn(batchsize, n1)

# Compute mean and variance
mu = np.mean(Z, axis=0)                # Shape: (n1,)
sigma = np.std(Z, axis=0)              # Shape: (n1,)

# Normalize
Z_norm = (Z - mu) / (sigma + 1e-8)     # Shape: (batchsize, n1)

# Scale and shift
gamma = np.ones((n1,))                 # Shape: (n1,)
beta = np.zeros((n1,))                 # Shape: (n1,)
Z_bn = gamma * Z_norm + beta           # Shape: (batchsize, n1)

# Apply ReLU activation
A = np.maximum(0, Z_bn)                # Shape: (batchsize, n1)
```

### Batch-Normalization (2)
1. **Compute the normalised z-values (znorm) and the activations of the layer (expressed as a matrix) by assuming the scale and shift parameters introduced by batch norm to be 1 or 0, respectively (apply batch norm on z before applying ‘relu’).**

Given:
$$
z = \begin{bmatrix}
12 & 0 & -5 \\
14 & 10 & 5 \\
14 & 10 & 5 \\
12 & 0 & -5
\end{bmatrix}
$$

**Step-by-step solution:**

- **Compute mean ($\mu$) and standard deviation ($\sigma$) of z:**

$$
\mu = \begin{bmatrix}
13 & 5 & 0
\end{bmatrix}
$$
$$
\sigma = \begin{bmatrix}
1 & 5 & 5
\end{bmatrix}
$$

- **Normalize z:**
$$
z_{\text{norm}} = \frac{z - \mu}{\sigma} = \begin{bmatrix}
\frac{12-13}{1} & \frac{0-5}{5} & \frac{-5-0}{5} \\
\frac{14-13}{1} & \frac{10-5}{5} & \frac{5-0}{5} \\
\frac{14-13}{1} & \frac{10-5}{5} & \frac{5-0}{5} \\
\frac{12-13}{1} & \frac{0-5}{5} & \frac{-5-0}{5}
\end{bmatrix} = \begin{bmatrix}
-1 & -1 & -1 \\
1 & 1 & 1 \\
1 & 1 & 1 \\
-1 & -1 & -1
\end{bmatrix}
$$

- **Assuming $\gamma = 1$ and $\beta = 0$:**
$$
z_{\text{bn}} = \gamma \cdot z_{\text{norm}} + \beta = z_{\text{norm}}
$$

- **Apply ReLU activation:**
$$
A = \max(0, z_{\text{bn}}) = \begin{bmatrix}
0 & 0 & 0 \\
1 & 1 & 1 \\
1 & 1 & 1 \\
0 & 0 & 0
\end{bmatrix}
$$

2. **How many parameters (for scale and shift) are added by batch normalization?**

    - For each feature in the layer, batch normalization adds 2 parameters: $\gamma$ (scale) and $\beta$ (shift).
    - If there are $n$ features in the layer, the total number of parameters added is $2n$.

3. **Give two benefits of using batch normalization.**

    - **Stabilizes Learning:** Batch normalization helps in stabilizing the learning process by normalizing the inputs to each layer, which mitigates issues related to the internal covariate shift.
    - **Improved Performance and Regularization:** Batch normalization often leads to faster convergence and can have a slight regularizing effect, reducing the need for other regularization techniques like dropout.

4. **Describe how batch-normalization is handled at test or production time.**

    - During test or production time, the mean and variance used for normalization are typically calculated over the entire training dataset rather than a single batch. This ensures consistent normalization across different batches.
    - These values are stored and used to normalize the input in the same manner as during training, but without computing new mean and variance on the fly.

### Regularization
1. **Consider an arbitrary model with loss function $L$ and add L2 regularization. How is gradient descent update rule modified by the regularization? Why do we often refer to L2-regularization as “weight decay”? Derive a mathematical expression to explain your point.**

    - **Original loss function:**
      $$
      L = \text{Loss} + \frac{\lambda}{2} \sum_{i} w_i^2
      $$
    - **Gradient of the regularized loss function:**
      $$
      \nabla L = \nabla \text{Loss} + \lambda w
      $$
    - **Update rule:**
      $$
      w_{\text{new}} = w - \eta \nabla L = w - \eta (\nabla \text{Loss} + \lambda w) = w - \eta \nabla \text{Loss} - \eta \lambda w
      $$
    - **Weight decay interpretation:**
      $$
      w_{\text{new}} = w (1 - \eta \lambda) - \eta \nabla \text{Loss}
      $$
      This shows that L2 regularization effectively reduces the weights by a factor of $(1 - \eta \lambda)$ at each step, which is why it is called "weight decay".

2. **Explain the steps to be performed (at training and test time) when implementing dropout and explain why dropout in a neural network acts as a regularizer.**

    - **Training time:**
      - Randomly set a fraction $p$ of the input units to zero at each update during training.
      - Scale the remaining units by $ \frac{1}{1-p} $ to maintain the same expected output.
    - **Test time:**
      - Use the entire network but scale the weights by $ p $ to match the expected value during training.
    - **Regularization effect:**
      - Dropout prevents units from co-adapting too much, forcing the network to learn more robust features that generalize better. It reduces overfitting by ensuring that the network does not rely on any single feature, promoting independence among neurons.

3. **Why is mini-batch gradient descent considered as having a regularising effect?**

    - Mini-batch gradient descent introduces noise in the gradient estimation due to the use of a subset of the data, which can help in escaping local minima and lead to better generalization. This stochasticity acts as a form of regularization.

4. **Why is batch-norm considered as having a regularising effect?**

    - Batch normalization helps regularize the model by reducing internal covariate shift, stabilizing the learning process, and often reducing the need for other regularization techniques like dropout. It also allows for higher learning rates, which can lead to faster convergence and better generalization.

Let's address the tasks regarding computational graphs, backpropagation, and gradient descent using PyTorch.

### Computational Graphs

A computational graph is a way to represent and visualize the operations involved in computing the output of a function, and it’s particularly useful in the context of neural networks.

**Example Function:** $ f(x, y, z) = (x + y) * z $

**Computational Graph:**

1. $a = x + y$
2. $b = a * z$

Here, $x, y, z$ are input nodes, and $a$ and $b$ are intermediate nodes.

**Local Gradients:**
- $\frac{\partial a}{\partial x} = 1$
- $\frac{\partial a}{\partial y} = 1$
- $\frac{\partial b}{\partial a} = z$
- $\frac{\partial b}{\partial z} = a$

**Accumulated Gradients:**
- $\frac{\partial b}{\partial x} = \frac{\partial b}{\partial a} * \frac{\partial a}{\partial x} = z$
- $\frac{\partial b}{\partial y} = \frac{\partial b}{\partial a} * \frac{\partial a}{\partial y} = z$
- $\frac{\partial b}{\partial z} = a = (x + y)$

### Backpropagation Example

Consider a simple network with two hidden layers:

**Forward Pass:**
1. $ z_1 = W_1 x + b_1 $
2. $ a_1 = \sigma(z_1) $
3. $ z_2 = W_2 a_1 + b_2 $
4. $ a_2 = \sigma(z_2) $
5. $ z_3 = W_3 a_2 + b_3 $
6. $ \hat{y} = \sigma(z_3) $

**Backpropagation Steps:**
1. Compute loss: $ L = \frac{1}{2} (\hat{y} - y)^2 $
2. Compute gradients w.r.t output: $ \delta_3 = \hat{y} - y $
3. Backpropagate to second layer: $ \delta_2 = \delta_3 W_3 \sigma'(z_2) $
4. Backpropagate to first layer: $ \delta_1 = \delta_2 W_2 \sigma'(z_1) $

### Gradient Descent with PyTorch

Here’s an example of implementing mini-batch gradient descent using PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Example model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Generating some random data
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# Creating DataLoader
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Initializing model, loss function, and optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
n_epochs = 20
for epoch in range(n_epochs):
    for data in loader:
        inputs, targets = data
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

print("Training complete!")
```

**Explanation:**

- `model(inputs)` performs a forward pass.
- `loss.backward()` computes the gradients.
- `optimizer.step()` updates the model parameters based on the gradients.
- `optimizer.zero_grad()` clears old gradients before computing new ones.

### Specific Questions from the Exam

1. **Explain `torch.no_grad()`:** It is used to disable gradient calculation, which reduces memory consumption for computations that do not need gradients (e.g., model evaluation).

2. **Why `optimizer.zero_grad()` is needed:** To reset the gradients of model parameters before backpropagation. Gradients accumulate by default in PyTorch, so they must be zeroed before the next iteration.

3. **Autograd:** PyTorch's autograd module provides automatic differentiation for all operations on Tensors. It records operations as a graph and computes gradients through backpropagation.

Feel free to ask for more detailed explanations or specific code examples related to other parts of the questions!