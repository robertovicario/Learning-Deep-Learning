# Convolutional Neural Network (CNN)

## Overview

Convolutional Neural Networks (CNNs) are a class of deep neural networks specifically designed for analyzing visual data. They have proven highly effective in tasks such as image classification, object detection, and segmentation. CNNs mimic the way humans perceive visual information, utilizing a hierarchical structure that captures spatial hierarchies in images. The core idea is to apply convolutional operations, which involve sliding filters over the input data to detect local patterns. CNNs have limitations, including the need for large amounts of labeled data and significant computational resources for training.

## Architecture

CNNs consist of a series of convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification.

<table>
    <tr>
        <td><img src="/CNN/img/1.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Architecture</td>
    </tr>
</table>

### Convolutional Layer

The convolutional layer is the core building block of a CNN. It applies a convolution operation to the input, passing the result to the next layer.

Mathematically:

$$...$$

The key concepts in convolutional layers include:

- **Receptive Field:** The receptive field is the region of the input image that affects a particular output value. It's determined by the size of the filter.

- **Stride:** The stride is the step size with which the convolution filter moves across the input image.

- **Padding:** Padding involves adding extra pixels around the input image to control the spatial size of the output.

- **Activation Map:** After applying the convolution operation, the resulting output is called an activation map.

<table>
    <tr>
        <td><img src="/CNN/img/2.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Convolutional Layer</td>
    </tr>
</table>

### Pooling Layer

The pooling layer reduces the spatial dimensions of the activation maps, which helps decrease the computational load and the number of parameters.

Mathematically:

$$...$$

Pooling functions include:

- **Max Pooling:** Selects the maximum value from each patch of the feature map.

- **Average Pooling:** Computes the average value of each patch of the feature map.

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

The fully connected layer connects every neuron in one layer to every neuron in the next layer. This layer is typically used at the end of the network to combine features learned by convolutional and pooling layers and to produce the final output.

Mathematically:

$$...$$

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

# Convolutional Layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

# Output Layers
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
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
