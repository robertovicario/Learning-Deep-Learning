# Multi-Layer Perceptron (MLP)

## Overview

A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural networks (ANN). MLPs can approximate complex functions and relationships within data and are useful in image and speech recognition. MLPs requires significant computational power, especially for large networks, and without proper regularization can overfit to training data.

## Architecture

The architecture of an MLP is characterized by its layered structure, where each layer contains multiple neurons. The fundamental layers of an MLP include the input layer, one or more hidden layers, and the output layer.

<table>
    <tr>
        <td><img src="/MLP/img/1.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Architecture</td>
    </tr>
</table>

### Input Layer

The input layer is the first layer in the MLP and is responsible for receiving the input features. Each neuron in this layer represents a single input feature from the dataset. This layer does not perform any computations but simply forwards the input data to the next layer.

Mathematically, given an input vector:

$$\mathbf{x} = [x_1, x_2, \ldots, x_n]$$

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

## TensorFlow Implementation

### Model Definition

```py
# Model Definition
model = Sequential()

model.add(Flatten(input_shape=input_shape))  # Input Layer

# Hidden Layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))  # Output Layer

model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### Training and Evaluation

```py
# Training and Evaluation
epochs = 20
batch_size = 128

history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
```

## Reference

- [Stanford CS230 Deep Learning Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230) by Afshine Amidi and Shervine Amidi
