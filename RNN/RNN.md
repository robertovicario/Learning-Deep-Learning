# Recurrent Neural Network (RNN)

## Overview

Recurrent Neural Networks (RNNs) are a class of artificial neural networks designed to recognize patterns in sequences of data, such as time series or natural language. RNNs are powerful tools for sequential data processing, they come with significant challenges such as the time-consuming, and the prone to overfitting, especially when trained on small datasets.

## Architecture

An RNN consists of a network of nodes, or neurons, arranged in layers. Each neuron in a given layer is connected to every neuron in the next layer. What sets RNNs apart from feedforward neural networks is the presence of recurrent connections, where the output of a neuron at time step $t$ is used as input to the same neuron at time step $t+1$. This recurrent connection allows the network to retain information about previous inputs, effectively creating a memory.

### Simple Layer

In the simplest form, an RNN layer consists of neurons where each neuron has a recurrent connection to itself across time steps.

Mathemathically, this is often represented as follows:

$$h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

where $h_t$ is the hidden state at time step $t$, $W_{hh}$ is the weight matrix for the hidden state, $h_{t-1}$ is the hidden state from the previous time step, $W_{xh}$ is the weight matrix for the input $x_t$, $x_t$ is the input at time step $t$, $b_h$ is the bias term, and$\sigma$ is the activation function.

<table>
    <tr>
        <td><img src="/RNN/img/1.png" width="256"></td>
        <td><img src="/RNN/img/2.png" width="256"></td>
    </tr>
    <tr>
        <td align="center">Architecture</td>
        <td align="center">Single Cell</td>
    </tr>
</table>

### Advanced Layers

These layers go beyond basic feedforward neural networks by introducing mechanisms to capture temporal dependencies, improve memory, and enhance model expressiveness. Below are descriptions of some advanced layers:

- **Long Short-Term Memory (LSTM):** These networks are a type of RNN designed to capture long-range dependencies and mitigate the vanishing gradient problem. LSTMs introduce a more complex cell structure, which includes gates that regulate the flow of information.

- **Gated Recurrent Unit (GRU):** This is a simplified version of LSTM that combines the forget and input gates into a single update gate and merges the cell state and hidden state. This makes GRUs computationally more efficient while still addressing the vanishing gradient problem.

<table>
    <tr>
        <td><img src="/RNN/img/3.png" width="256"></td>
        <td><img src="/RNN/img/4.png" width="256"></td>
    </tr>
    <tr>
        <td align="center">LSTM</td>
        <td align="center">GRU</td>
    </tr>
</table>

### Dense Layer

In an RNN architecture, the output from the RNN layers is often passed through one or more dense (fully connected) layers to perform the final prediction.

## TensorFlow Implementation

### Model Definition

```py
# Model Definition
model = Sequential()

model.add(SimpleRNN(128, activation='relu', input_shape=input_shape))  # Input Layer

# Hidden Layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))  # Output Layer

model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### Training and Evaluation

```py
# Training and Evaluation
epochs = 10
batch_size = 32

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
