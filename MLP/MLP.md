# Multi-Layer Perceptron (MLP)

## Overview

A Multi-Layer Perceptron (MLP) is a type of artificial neural network consisting of an input layer, one or more hidden layers, and an output layer. Each node in one layer connects to every node in the next with a specific weight. MLPs use backpropagation for training and excel in tasks like classification, regression, and pattern recognition, making them valuable in image and speech recognition, and financial forecasting. However, they require large amounts of labeled data and significant computational resources, and can overfit with small or noisy datasets.

## Architecture

<table>
    <tr>
        <td><img src="/MLP/img/1.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Architecture</td>
    </tr>
</table>

---

### Input Layer

The input layer is the first layer in the MLP and is responsible for receiving the input data. The number of nodes in this layer corresponds to the number of features in the input dataset. Each node in the input layer passes the input data to the next layer without any transformation.

### Hidden Layer

The hidden layers lie between the input and output layers and are the core of the MLP's learning capabilities. These layers perform most of the computations and learning through the neurons, each of which applies a nonlinear activation function to transform the input data. The commonly used activation functions include ReLU (Rectified Linear Unit), sigmoid, and tanh.

### Output Layer

The output layer is the final layer in the MLP and is responsible for producing the final output of the network. The number of nodes in this layer corresponds to the number of classes in a classification problem or the number of output values in a regression problem. The output layer typically uses an activation function such as softmax for classification tasks or a linear activation function for regression tasks.

## TensorFlow Implementation

### Model Definition

```py
model = Sequential()

input_shape = (timesteps, input_dim)
output_dim = num_classes

# Input Layer
model.add(Flatten(input_shape=input_shape))

# Hidden Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout Layer
model.add(Dense(64, activation='relu'))

# Output Layer
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### Training and Evaluation

```py
epochs = ...
batch_size = ...

history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
```

### Plot

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
