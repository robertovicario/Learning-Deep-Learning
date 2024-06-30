# Convolutional Neural Network (CNN)

## Overview

Convolutional Neural Networks (CNNs) are a class of deep learning models specifically designed to process data that has a grid-like topology, such as images. They are particularly effective for tasks like image classification, object detection, and segmentation. CNNs have limitations, they require large amounts of labeled data and significant computational resources for training.

## Architecture

A typical CNN architecture consists of several key layers that transform the input image into an output class score.

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

The convolution operation for a given input $I$ and a filter $K$ is defined as:

$$(I * K)(i,j) = \sum_{m}\sum_{n} I(i-m,j-n)K(m,n)$$

Where $(i, j)$ are the coordinates of the output matrix.

The key concepts in convolutional layers include:

- **Receptive Fields:** The receptive field is the region of the input image that affects a particular output value. In CNNs, neurons in early layers have small receptive fields, while neurons in deeper layers have larger receptive fields.

- **Activation Maps:** After applying the convolution operation, the resulting output is called an activation map (or feature map), which highlights the presence of features in the input.

- **Stride:** The stride is the step size with which the convolution filter moves across the input image. A larger stride reduces the spatial dimensions of the output.

- **Padding:** Padding involves adding extra pixels around the input image to control the spatial size of the output. Common padding strategies include "valid" (no padding) and "same" (padding to keep the output size equal to the input size).

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

- **Max Pooling:** Selects the maximum value from each patch of the feature map.

- **Average Pooling:** Computes the average value of each patch of the feature map.

For max pooling with a pool size of $p \times p$:

$$P(i,j) = \max \{ I(m,n) : (m,n) \in \text{window}(i,j) \}$$

Where $\text{window}(i,j)$ defines the $p \times p$ region over which the maximum is taken.

<table>
    <tr>
        <td align="center">Max Pooling</td>
        <td align="center">Average Pooling</td>
    </tr>
    <tr>
        <td><img src="/CNN/img/3.png" width="256"></td>
        <td><img src="/CNN/img/4.png" width="256"></td>
    </tr>
</table>

### Fully Connected Layer

The fully connected layer connects every neuron in one layer to every neuron in the next layer. This layer is typically used at the end of the network to combine features learned by convolutional and pooling layers and to produce the final output.

The output of a fully connected layer is calculated as:

$$y = Wx + b$$

Where $y$ is the output vector, $W$ is the weight matrix, $x$ is the input vector, and $b$ is the bias vector.

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

```

### Training and Evaluation

```py

```

### Plot History

```py

```

## Reference

- [Stanford CS230 Deep Learning Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230) by Afshine Amidi and Shervine Amidi
