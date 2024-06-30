# Convolutional Neural Network (CNN)

## Overview

... capabilities, limitations

## Architecture

... layers, neuron roles

<table>
    <tr>
        <td><img src="/CNN/img/1.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Architecture</td>
    </tr>
</table>

### Convolutional Layer

The convolutional layer is the core building block of a CNN. It applies a convolution operation to the input, passing the result to the next layer. The key concepts in convolutional layers include:

**Receptive Fields:** The receptive field is the region of the input image that affects a particular output value. For a given layer, the receptive field $R$ is determined by the size of the filter and the cumulative stride and padding of all preceding layers.

**Stride:** The stride is the step size with which the convolution filter moves across the input image. A larger stride reduces the spatial dimensions of the output. If the stride is $s$, the position of the filter is moved by $s$ units each time. The output dimension $O$ for an input dimension $I$, filter size $F$, and stride $s$ is given by:

$$O = \left\lfloor \frac{I - F}{s} \right\rfloor + 1$$

**Padding:** Padding involves adding extra pixels around the input image to control the spatial size of the output. Common padding strategies include "valid" (no padding) and "same" (padding to keep the output size equal to the input size). For an input dimension $I$, filter size $F$, stride $s$, and padding $P$, the output dimension $O$ is given by:

$$O = \left\lfloor \frac{I + 2P - F}{s} \right\rfloor + 1$$

**Activation Maps:** After applying the convolution operation, the resulting output is called an activation map (or feature map), which highlights the presence of features in the input. If $f$ is the filter applied to an input $I$, the activation map $A$ at position $(i, j)$ is given by:

$$A(i, j) = (I * f)(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) f(m, n)$$

<table>
    <tr>
        <td><img src="/CNN/img/2.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Convolutional Layer</td>
    </tr>
</table>

### Pooling Layer

The pooling layer reduces the spatial dimensions of the activation maps, which helps decrease the computational load and the number of parameters. Pooling operations include:

**Max Pooling:** Selects the maximum value from each patch of the feature map. For a pooling window of size $p \times p$ at position $(i, j)$, the output $P_{max}(i, j)$ is:

$$P_{max}(i, j) = \max \{ A(i + m, j + n) \mid 0 \leq m, n < p \}$$

**Average Pooling:** Computes the average value of each patch of the feature map. For a pooling window of size $p \times p$ at position $(i, j)$, the output $P_{avg}(i, j)$ is:

$$P_{avg}(i, j) = \frac{1}{p^2} \sum_{m=0}^{p-1} \sum_{n=0}^{p-1} A(i + m, j + n)$$

<table>
    <tr>
        <td><img src="/CNN/img/3.png" width="256"></td>
        <td><img src="/CNN/img/4.png" width="256"></td>
    </tr>
    <tr>
        <td align="center">Max Pooling</td>
        <td align="center">Average Pooling</td>
    </tr>
</table>

### Fully Connected Layer

The fully connected layer connects every neuron in one layer to every neuron in the next layer. This layer is typically used at the end of the network to combine features learned by convolutional and pooling layers and to produce the final output. If $\mathbf{x}$ is the input vector to a fully connected layer, $\mathbf{W}$ is the weight matrix, and $\mathbf{b}$ is the bias vector, the output $\mathbf{y}$ is given by:

$$\mathbf{y} = \mathbf{W} \mathbf{x} + \mathbf{b}$$

<table>
    <tr>
        <td><img src="/CNN/img/5.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Fully Connected Layer</td>
    </tr>
</table>

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
