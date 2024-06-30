# Convolutional Neural Network (CNN)

## Overview

...

capabilities, limitations

## Architecture

...

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

## TensorFlow Implementation

### Model Definition

```py

```

### Training and Evaluation

```py

```
