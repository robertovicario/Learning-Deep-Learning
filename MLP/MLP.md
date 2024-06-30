# Multi-Layer Perceptron (MLP)

## Overview

A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural network (ANN). An MLP consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training.

## Architecture

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

The hidden layers lie between the input and output layers and are the core of the MLP's learning capabilities. These layers perform most of the computations and learning through the neurons, each of which applies a nonlinear activation function to transform the input data. The commonly used activation functions include ReLU (Rectified Linear Unit), sigmoid, and tanh.

- **Dropout Layer:** A dropout layer is often added to the hidden layers to prevent overfitting. During training, it randomly sets a fraction of input units to zero at each update, which helps to prevent the network from becoming too reliant on specific neurons and improves generalization.

### Output Layer

The output layer is the final layer in the MLP and is responsible for producing the final output of the network. The number of nodes in this layer corresponds to the number of classes in a classification problem or the number of output values in a regression problem. The output layer typically uses an activation function such as softmax for classification tasks or a linear activation function for regression tasks.

## TensorFlow Implementation

### Model Definition

> [!NOTE]
>
> **Input Shape:** For a sequential model processing any sequence data, the input shape typically consists of the number of timesteps (sequence length) and the number of features per timestep (input dimension).

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

> [!NOTE]
>
> **Epochs:** One epoch means that each sample in the training dataset has had an opportunity to update the model's parameters once.
>
> If model underfits increase epochs, else if model overfits decrease epochs.

> [!NOTE]
>
> **Batch Size:** The number of training samples used to compute a single gradient update.
>
> Adjust batch size to balance training speed and model accuracy. Small batch sizes may lead to noisy training, while large batch sizes may cause overfitting.

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

## Model Evaluation

<table>
    <tr>
        <td><img src="/MLP/img/1.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Model Evaluation</td>
    </tr>
</table>

### Model Loss

- **Train Loss:** The training loss decreases steadily from above 0.5 to below 0.05, indicating that the model is learning well during the training phase.

- **Validation Loss:** The validation loss also decreases, although it starts lower than the training loss and plateaus around the same value as the training loss after about 5 epochs. This suggests that the model is generalizing well to the validation data without significant overfitting.

### Model Accuracy

- **Train Accuracy:** The training accuracy increases sharply at the beginning and then more gradually, reaching around 96% by the end of the training period.

- **Validation Accuracy:** The validation accuracy also improves rapidly at the beginning and reaches about 98%, slightly higher than the training accuracy, indicating good generalization.
