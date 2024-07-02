# Multi-Layer Perceptron (MLP)

## Overview

A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural networks (ANN). MLPs can approximate complex functions and relationships within data and are useful in image and speech recognition. MLPs requires significant computational power, especially for large networks, and without proper regularization can overfit to training data.

## Architecture

The architecture of an MLP is characterized by its layered structure, where each layer contains multiple neurons. The fundamental layers of an MLP include the input layer, one or more hidden layers, and the output layer.

<table>
    <tr>
        <td><img src="/MLP/img/1.png" width="256"></td>
    </tr>
    <tr>
        <td align="center">Architecture</td>
    </tr>
</table>

### Input Layer

The input layer is the first layer in the MLP and is responsible for receiving the input features. Each neuron in this layer represents a single input feature from the dataset. This layer does not perform any computations but simply forwards the input data to the next layer.

Mathematically, given an input vector:

$$x = [x_1, x_2, \ldots, x_n]$$

where $x_i$ represents the $i$-th feature of the input data, the input layer forwards this vector to the next layer without any modification.

### Hidden Layers

The hidden layers are where most of the computation happens in an MLP. Each neuron in a hidden layer takes a weighted sum of the inputs from the previous layer, adds a bias term, and applies an activation function to introduce non-linearity into the model. The purpose of these layers is to extract features and patterns from the input data.

Mathematically, for the $k$-th hidden layer, the computation for the $j$-th neuron is given by:

$$z_j^{(k)} = \sum_{i=1}^{n} w_{ji}^{(k)} a_i^{(k-1)} + b_j^{(k)}$$

where $z_j^{(k)}$ is the weighted sum of inputs for the $j$-th neuron in the $k$-th layer, $w_{ji}^{(k)}$ is the weight connecting the $i$-th neuron of the $(k-1)$-th layer to the $j$-th neuron of the $k$-th layer, $a_i^{(k-1)}$ is the activation of the $i$-th neuron in the $(k-1)$-th layer (with $a_i^{(0)} = x_i$ for the input layer), and $b_j^{(k)}$ is the bias term for the $j$-th neuron in the $k$-th layer. The activation $a_j^{(k)}$ of the $j$-th neuron in the $k$-th layer is obtained by applying an activation function $f$ to $z_j^{(k)}$:

$$a_j^{(k)} = z_j^{(k)}$$

### Output Layer

The output layer is the final layer of the MLP, and it provides the network's predictions.

Mathematically, for the output layer, the computation for the $j$-th output neuron is given by:

$$z_j^{(L)} = \sum_{i=1}^{n} w_{ji}^{(L)} a_i^{(L-1)} + b_j^{(L)}$$

where $z_j^{(L)}$ is the weighted sum of inputs for the $j$-th neuron in the output layer, $w_{ji}^{(L)}$ is the weight connecting the $i$-th neuron of the last hidden layer to the $j$-th neuron of the output layer, $a_i^{(L-1)}$ is the activation of the $i$-th neuron in the last hidden layer, and $b_j^{(L)}$ is the bias term for the $j$-th neuron in the output layer.

The final output $a_j^{(L)}$ of the $j$-th neuron in the output layer is given by:

$$a_j^{(L)} = z_j^{(L)}$$

## Activation Functions

Activation functions introduce non-linearity into the model, allowing it to learn complex patterns.

### Sigmoid

The Sigmoid function is an S-shaped curve that maps any real-valued number into the range (0, 1). It's often used in the output layer of binary classification problems.

Mathematically, the Sigmoid function is defined as:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

<table>
    <tr>
        <td><img src="/MLP/img/2.png" width="256"></td>
    </tr>
    <tr>
        <td align="center">Sigmoid</td>
    </tr>
</table>

### Tanh

The Tanh function is similar to the Sigmoid function but maps values to the range (-1, 1). It is zero-centered, which makes it a better choice than the Sigmoid function in some cases.

Mathematically, the Tanh function is expressed as:

$$\sigma(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

<table>
    <tr>
        <td><img src="/MLP/img/3.png" width="256"></td>
    </tr>
    <tr>
        <td align="center">Tanh</td>
    </tr>
</table>

### ReLU

The Rectified Linear Unit (ReLU) is a widely used activation function in deep learning. It helps in solving the vanishing gradient problem, making the training of deep networks more efficient. ReLU outputs the input directly if it is positive; otherwise, it will output zero.

Mathematically, the ReLU function is given by:

$$\sigma(x) = \max(0, x)$$

<table>
    <tr>
        <td><img src="/MLP/img/4.png" width="256"></td>
    </tr>
    <tr>
        <td align="center">ReLU</td>
    </tr>
</table>

### Leaky ReLU

Leaky ReLU is a variation of the ReLU function. Instead of outputting zero for negative input values, it will allow a small, non-zero, constant gradient (usually 0.01).

Mathematically, the Leaky ReLU function is defined as:

$$
\sigma(x) = \begin{cases} 
    x & x \ge 0 \\
    \alpha x & x < 0 
\end{cases}
$$

where $\alpha$ is a small constant (typically 0.01).

<table>
    <tr>
        <td><img src="/MLP/img/5.png" width="256"></td>
    </tr>
    <tr>
        <td align="center">Leaky ReLU</td>
    </tr>
</table>

## PyTorch Implementation

### Model Definition

```py
# Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(28*28, 128)
        self.dense2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.softmax(self.output(x), dim=1)
        return x

model = MLP()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Training and Evaluation

```py
# Training and Evaluation
epochs = 20
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    total_train = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_loss /= len(train_loader)
    train_accuracy = 100 * train_correct / total_train
    train_loss_history.append(train_loss)
    train_acc_history.append(train_accuracy)
    
    model.eval()
    val_loss = 0
    val_correct = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss /= len(test_loader)
    val_accuracy = 100 * val_correct / total_val
    val_loss_history.append(val_loss)
    val_acc_history.append(val_accuracy)
    
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
```

## Reference

- [Stanford CS230 Deep Learning Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230) by Afshine Amidi and Shervine Amidi
