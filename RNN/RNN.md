Certainly! Let's complete the lecture on Recurrent Neural Network (RNN).

# Recurrent Neural Network (RNN)

## Overview

Recurrent Neural Networks (RNNs) are a type of neural network architecture designed for sequential data. Unlike traditional feedforward neural networks, RNNs have connections that form directed cycles, allowing them to maintain a hidden state that can capture information about previous inputs. This makes them particularly well-suited for tasks where the order of the data matters, such as time series prediction, natural language processing, and more.

## Architecture

### Simple Layer

The basic RNN layer consists of neurons that not only receive input from the current time step but also from the previous time step's hidden state. The hidden state is updated at each time step based on the current input and the previous hidden state, enabling the network to maintain a memory of past inputs.

The mathematical formulation for the hidden state $h_t$ at time step $t$ is as follows:

$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

where:
- $x_t$ is the input at time step $t$,
- $h_{t-1}$ is the hidden state from the previous time step,
- $W_{xh}$ is the weight matrix for the input,
- $W_{hh}$ is the weight matrix for the hidden state,
- $b_h$ is the bias term,
- $\tanh$ is the hyperbolic tangent activation function.

### LSTM Layer

Long Short-Term Memory (LSTM) networks are a type of RNN designed to capture long-range dependencies and mitigate the vanishing gradient problem. LSTMs introduce a more complex cell structure, which includes gates that regulate the flow of information.

The key components of an LSTM cell include:

- **Forget Gate:** $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
- **Input Gate:** $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
- **Cell State Update:** $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
- **New Cell State:** $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
- **Output Gate:** $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
- **New Hidden State:** $h_t = o_t \odot \tanh(C_t)$

where $\sigma$ is the sigmoid activation function and $\odot$ denotes element-wise multiplication.

### GRU Layer

Gated Recurrent Unit (GRU) is a simplified version of LSTM that combines the forget and input gates into a single update gate and merges the cell state and hidden state. This makes GRUs computationally more efficient while still addressing the vanishing gradient problem.

The key components of a GRU cell include:

- **Update Gate:** $z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$
- **Reset Gate:** $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$
- **Candidate Activation:** $\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$
- **New Hidden State:** $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$


### Dense Layer

The dense layer, also known as a fully connected layer, is a traditional neural network layer where each neuron is connected to every neuron in the previous layer. In the context of RNNs, dense layers are typically used after the recurrent layers to process the sequential data output by the RNN or LSTM/GRU layers and produce the final output.

- **Dropout Layer:** A dropout layer randomly sets a fraction of the input units to zero during training to prevent overfitting. This helps the model generalize better to new data.

## TensorFlow Implementation

### Model Definition

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

```py
epochs = 25
batch_size = 64
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
```
