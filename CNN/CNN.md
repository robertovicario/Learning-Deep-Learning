# Convolutional Neural Network (CNN)

## Overview

...

Capabilities:

- 

Limitations:

- 


### Convolution Layer (2D Signals)

- **Spatial Structure:** Preserve spatial relationships in the input data.
- **Convolution Process:** Slide filter over the image, computing dot products with the window of pixels.
- **Activation Map:** Result of convolving the filter over all spatial locations.
- **Sparse Connectivity:** Connects to spatially constrained portions of the input.
- **Parameter Sharing:** One filter's parameters are shared among many portions of the input.

### Convolution Layer (1D Signals)

- **Principle:** Same as 2D but applied to time series.
- **Output:** 1D signal with preserved spatial structure.

### Filters in Convolution Layer
- **Purpose:** Detect specific patterns like edges, strokes, or textures.
- **Training:** Filters are learned through the training process.

### Additional Layers
- **Max Pooling Layer:** Reduces spatial size of representation while keeping significant activations.
- **Dropout Layer:** Randomly sets activations to zero to make the network robust and avoid overfitting.
- **Dense Layer:** Fully connected layer used at the output of CNN for classification.

### Full Architecture for Image Recognition
- **Typical Configuration:** Stack sequences of CONV-RELU-POOL layers, ending with a fully connected dense layer.
