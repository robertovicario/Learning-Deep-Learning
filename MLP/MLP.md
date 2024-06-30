
# Multi-Layer Perceptron (MLP)

## Overview

A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural network (ANN). An MLP consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training.

## Architecture


## Learning Algorithm

### Initialization

Weights ($W$) and biases ($b$) are initialized, typically with small random values.

$$
W_{ij} \sim \mathcal{N}(0, \sigma^2) \quad \text{or} \quad W_{ij} \sim \mathcal{U}(-\epsilon, \epsilon)
$$

### Forward Pass

Input data ($X$) is passed through the network. Each neuron computes a weighted sum of its inputs, applies an activation function ($\phi$), and passes the result to the next layer.

$$
z_j = \sum_{i} W_{ij} x_i + b_j
$$

$$
a_j = \phi(z_j)
$$

### Loss Calculation

The output of the network ($\hat{y}$) is compared to the true target values ($y$) using a loss function ($L$) to compute the error.

$$
\hat{y} = \phi \left( \sum_{j} W_{jk} a_j + b_k \right)
$$

$$
L = \frac{1}{n} \sum_{i=1}^{n} \ell(\hat{y}_i, y_i)
$$

### Backward Pass (Backpropagation)

The error is propagated back through the network to update the weights. This is done using the chain rule of calculus to compute gradients.

$$
\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_j} \cdot \frac{\partial z_j}{\partial W_{ij}}
$$

$$
\frac{\partial L}{\partial b_j} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_j} \cdot \frac{\partial z_j}{\partial b_j}
$$

### Weight Update

Weights are updated using an optimization algorithm such as Stochastic Gradient Descent (SGD). This involves adjusting the weights in the direction that reduces the loss.

$$
W_{ij} \leftarrow W_{ij} - \eta \frac{\partial L}{\partial W_{ij}}
$$

$$
b_j \leftarrow b_j - \eta \frac{\partial L}{\partial b_j}
$$

Where $\eta$ is the learning rate.

## PyTorch Implementation

### Model Definition

```py
# Defining the model
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)
```

### Training and Evaluation

```py
# Creating DataLoader for batch processing
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
```

```py
# Instantiating the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

```py
# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    
    print(f'Epoch {epoch+1}/{n_epochs} - loss: {avg_loss:.4f} - accuracy: {accuracy:.4f}')
```

## Empirical Risk Minimization (ERM)

Empirical Risk Minimization is a principle in statistical learning theory which states that a model should minimize the average loss on the training data. Given a hypothesis class $H$, the goal is to find a hypothesis $h \in H$ that minimizes the empirical risk:

$$
\hat{R}(h) = \frac{1}{n} \sum_{i=1}^{n} \ell(h(x_i), y_i)
$$

where $\ell$ is the loss function, $x_i$ are the input samples, and $y_i$ are the corresponding target values.

## Bias and Variance

Bias and variance are key concepts in understanding the performance of learning algorithms:

- **Bias:** Refers to the error introduced by approximating a real-world problem, which may be complex, by a simplified model.

- **Variance:** Refers to the error introduced by the model's sensitivity to small fluctuations in the training set.

### Bias-Variance Tradeoff

The bias-variance tradeoff is a fundamental concept in machine learning that describes the tradeoff between the error due to bias and the error due to variance.

- **High Bias:** A model with high bias pays little attention to the training data and oversimplifies the model. It leads to high error on training and test data (underfitting).

- **Underfitting:** When a model is too simple, it cannot capture the underlying structure of the data, resulting in high bias.

- **High Variance:** A model with high variance pays too much attention to the training data, capturing noise along with the underlying pattern. It performs well on training data but poorly on test data (overfitting).

- **Overfitting:** When a model is too complex, it captures the noise in the training data, resulting in high variance.

## Regularization

Regularization techniques are used to prevent overfitting by adding a penalty term to the loss function. Common regularization methods include weight penalties, early stopping, and dropout.

### Weight Penalty

Weight penalties, also known as weight decay, add a term to the loss function that penalizes large weights. This encourages the model to find simpler solutions that generalize better.

$$
L_{\text{reg}} = L + \lambda \sum_{i} W_{i}^2
$$

### Early Stopping

Early stopping involves monitoring the performance of the model on a validation set and stopping training when the performance starts to deteriorate. This helps prevent overfitting to the training data.

### Dropout

Dropout is a regularization technique where randomly selected neurons are ignored during training. This prevents the model from becoming too dependent on specific neurons, thereby improving generalization.

Implementation:

```py
# Adding Dropout to the model
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 2)
)
```
