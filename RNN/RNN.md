# Recurrent Neural Network (RNN)

## Overview

Recurrent Neural Networks (RNNs) are a type of neural network architecture designed for sequential data. Unlike traditional feedforward neural networks, RNNs have connections that form directed cycles, allowing them to maintain a hidden state that can capture information about previous inputs. This makes them particularly well-suited for tasks where the order of the data matters, such as time series prediction, natural language processing, and more.

## Architecture

An RNN consists of a network of nodes, or neurons, arranged in layers. Each neuron in a given layer is connected to every neuron in the next layer. What sets RNNs apart from feedforward neural networks is the presence of recurrent connections, where the output of a neuron at time step $t$ is used as input to the same neuron at time step $t+1$. This recurrent connection allows the network to retain information about previous inputs, effectively creating a memory.

### Simple Layer

The basic RNN layer comprises neurons that receive input from both the current time step and the hidden state from the previous time step. The hidden state is updated at each time step based on the current input and the previous hidden state, allowing the network to maintain a memory of past inputs.

A single RNN cell consists of an input vector $x_t$, a hidden state vector $h_t$, and an output vector $y_t$. The hidden state $h_t$ is updated based on the previous hidden state $h_{t-1}$ and the current input $x_t$. The update process can be described mathematically as follows:

$$\boxed{h_t = \sigma(W_h \cdot h_{t-1} + W_x \cdot x_t + b_h)}$$

$$\boxed{y_t = W_y \cdot h_t + b_y}$$

where $\sigma$ is a non-linear activation function, $W_h$, $W_x$, and $W_y$ are weight matrices, and $b_h$ and $b_y$ are bias vectors.

<table>
    <tr>
        <td><img src="/RNN/img/1.png" width="512"></td>
        <td><img src="/RNN/img/2.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">Architecture</td>
        <td align="center">Single Cell</td>
    </tr>
</table>

### Advanced Layers

- **LSTM:** Long Short-Term Memory (LSTM) networks are a type of RNN designed to capture long-range dependencies and mitigate the vanishing gradient problem. LSTMs introduce a more complex cell structure, which includes gates that regulate the flow of information.

- **GRU:** Gated Recurrent Unit (GRU) is a simplified version of LSTM that combines the forget and input gates into a single update gate and merges the cell state and hidden state. This makes GRUs computationally more efficient while still addressing the vanishing gradient problem.

### Dense Layer

The dense layer, also known as a fully connected layer, is a traditional neural network layer where each neuron is connected to every neuron in the previous layer. In the context of RNNs, dense layers are typically used after the recurrent layers to process the sequential data output by the RNN or LSTM/GRU layers and produce the final output.

- **Dropout Layer:** A dropout layer randomly sets a fraction of the input units to zero during training to prevent overfitting. This helps the model generalize better to new data.

## TensorFlow Implementation

### Model Definition

> [!NOTE]
>
> **Input Shape:** For a sequential model processing any sequence data, the input shape typically consists of the number of timesteps (sequence length) and the number of features per timestep (input dimension).
>

```py
model = Sequential()

input_shape = (timesteps, input_dim)
output_dim = num_classes

model.add(SimpleRNN(units=50, activation='tanh', input_shape=input_shape))
model.add(Dense(units=output_dim, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### Training and Evaluation

> [!NOTE]
>
> **Epochs:** One epoch means that each sample in the training dataset has had an opportunity to update the model's parameters once.
>
> **[!!]** If model underfits increase epochs, else if model overfits decrease epochs.
>
> **Batch Size:** The number of training samples used to compute a single gradient update.
>
> **[!!]** Adjust batch size to balance training speed and model accuracy. Small batch sizes may lead to noisy training, while large batch sizes may cause overfitting.
>

```py
epochs = 50
batch_size = 32

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
```

## Reference

- [github.com/afshinea/stanford-cs-230-deep-learning](https://github.com/afshinea/stanford-cs-230-deep-learning)
