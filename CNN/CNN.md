# Convolutional Neural Network (CNN)

## Overview

Convolutional Neural Networks (CNNs) are a class of deep neural networks commonly used for analyzing visual imagery. They are inspired by the structure and function of the human visual cortex and are particularly well-suited for image classification, object detection, and other tasks involving grid-like data. CNNs leverage three main ideas: local receptive fields, shared weights, and spatial subsampling, to achieve high performance on tasks involving visual data.

## Architecture

### Convolutional Layer

The convolutional layer is the core building block of a CNN. It applies a convolution operation to the input, passing the result to the next layer. The key concepts in convolutional layers include:

- **2D Signals, 1D Signals:** Convolutional layers can handle different types of data, including 2D signals (e.g., images) and 1D signals (e.g., time series data).

- **Receptive Fields:** The receptive field is the region of the input image that affects a particular output value. In CNNs, neurons in early layers have small receptive fields, while neurons in deeper layers have larger receptive fields.

- **Activation Maps:** After applying the convolution operation, the resulting output is called an activation map (or feature map), which highlights the presence of features in the input.

- **Stride:** The stride is the step size with which the convolution filter moves across the input image. A larger stride reduces the spatial dimensions of the output.

- **Padding:** Padding involves adding extra pixels around the input image to control the spatial size of the output. Common padding strategies include "valid" (no padding) and "same" (padding to keep the output size equal to the input size).

### Pooling Layer

The pooling layer reduces the spatial dimensions of the activation maps, which helps decrease the computational load and the number of parameters. Pooling operations include:

- **Max Pooling:** Selects the maximum value from each patch of the feature map.

- **Average Pooling:** Computes the average value of each patch of the feature map.

### Dense Layer

The dense layer, or fully connected layer, connects every neuron in one layer to every neuron in the next layer. This layer is typically used at the end of the network to combine features learned by convolutional and pooling layers and to produce the final output. Dense layers are often followed by an activation function, such as ReLU (Rectified Linear Unit) or softmax, depending on the task.

- **Dropout Layer:** A dropout layer randomly sets a fraction of the input units to zero during training to prevent overfitting. This helps the model generalize better to new data.

## TensorFlow Implementation

### Model Definition

```py
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

```

### Training and Evaluation

```py
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
```
