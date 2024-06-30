# Multi-Layer Perceptron (MLP)

## Overview

A Multi-Layer Perceptron (MLP) is a type of artificial neural network consisting of an input layer, one or more hidden layers, and an output layer. Each node in one layer connects to every node in the next with a specific weight. MLPs use backpropagation for training and excel in tasks like classification, regression, and pattern recognition, making them valuable in image and speech recognition, and financial forecasting. However, they require large amounts of labeled data and significant computational resources, and can overfit with small or noisy datasets.

## Architecture

The architecture of a Multi-Layer Perceptron (MLP) consists of an input layer, hidden layers, and an output layer. Each neuron in one layer connects to every neuron in the next through weighted connections.

<table>
    <tr>
        <td><img src="/MLP/img/1.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Architecture</td>
    </tr>
</table>

### Input Layer

The input layer is the first layer in the MLP and is responsible for receiving the input data. The number of nodes in this layer corresponds to the number of features in the input dataset. Each node in the input layer passes the input data to the next layer without any transformation.

### Hidden Layer

The hidden layers lie between the input and output layers and are the core of the MLP's learning capabilities. These layers perform most of the computations and learning through the neurons, each of which applies a nonlinear activation function to transform the input data.

Each neuron in the hidden layer computes a weighted sum of its inputs, adds a bias, and applies an activation function:

$$z_j = \sum_{i=1}^{n} w_{ij}x_i + b_j$$

Where:

- $z_j$ is the input to the activation function of the $j$-th neuron.

- $x_i$ is the $i$-th input.

- $w_{ij}$ is the weight between the $i$-th input and the $j$-th neuron.

- $b_j$ is the bias of the $j$-th neuron.

The activation function $\phi$ is then applied to $z_j$:

$$a_j = \phi(z_j)$$

### Output Layer

The output layer is the final layer in the MLP and is responsible for producing the final output of the network. The number of nodes in this layer corresponds to the number of classes in a classification problem or the number of output values in a regression problem.

For a classification task, the output layer often uses the softmax activation function:

$$\text{softmax}(z_k) = \exp(z_k) \cdot \left( \sum_{j=1}^{K} \exp(z_j) \right)^{-1}$$

where $z_k$ is the input to the $k$-th output neuron and $K$ is the number of output neurons.

For a regression task, the output layer may use a linear activation function:

$$a_k = z_k$$

## TensorFlow Implementation

### Model Definition

```py
model = Sequential()

model.add(Flatten(input_shape=input_shape))  # Input Layer

# Hidden Layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))  # Output Layer

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### Training and Evaluation

```py
history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
```

### Plot History

```py
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
```

## Reference

- [Stanford CS230 Deep Learning Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230) by Afshine Amidi and Shervine Amidi
