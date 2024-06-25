# Recurrent Neural Network (RNN)

## Overview

Recurrent Neural Networks (RNNs) are a type of neural network architecture designed for sequential data. Unlike traditional feedforward neural networks, RNNs have connections that form directed cycles, allowing them to maintain a hidden state that can capture information about previous inputs. This makes them particularly well-suited for tasks where the order of the data matters, such as time series prediction, natural language processing, and more.

## Architecture

An RNN consists of a network of nodes, or neurons, arranged in layers. Each neuron in a given layer is connected to every neuron in the next layer. What sets RNNs apart from feedforward neural networks is the presence of recurrent connections, where the output of a neuron at time step $t$ is used as input to the same neuron at time step $t+1$. This recurrent connection allows the network to retain information about previous inputs, effectively creating a memory.

### Simple Layer

The basic RNN layer comprises neurons that receive input from both the current time step and the hidden state from the previous time step. The hidden state is updated at each time step based on the current input and the previous hidden state, allowing the network to maintain a memory of past inputs.

A single RNN cell consists of an input vector $x_t$, a hidden state vector $h_t$, and an output vector $y_t$. The hidden state $h_t$ is updated based on the previous hidden state $h_{t-1}$ and the current input $x_t$. The update process can be described mathematically as follows:

$$h_t = \sigma(W_h \cdot h_{t-1} + W_x \cdot x_t + b_h)$$
$$y_t = W_y \cdot h_t + b_y$$

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

### RNNs Types

RNNs can be categorized into different types based on the input and output sequences:

<table>
    <tr>
        <td align="center">Type</td>
        <td align="center">Diagram</td>
    </tr>
    <tr>
        <td align="center">One-to-one</td>
        <td><img src="/RNN/img/3.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">One-to-many</td>
        <td><img src="/RNN/img/4.png" width="512"></td>
    </tr>
        <tr>
        <td align="center">Many-to-one</td>
        <td><img src="/RNN/img/5.png" width="512"></td>
    </tr>
        <tr>
        <td align="center">Many-to-many<br>(Eq-Target)</td>
        <td><img src="/RNN/img/6.png" width="512"></td>
    </tr>
        <tr>
        <td align="center">Many-to-many<br>(Neq-target)</td>
        <td><img src="/RNN/img/7.png" width="512"></td>
    </tr>
</table>

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

## Reference

- [github.com/afshinea/stanford-cs-230-deep-learning](https://github.com/afshinea/stanford-cs-230-deep-learning)







## Architecture

### Basic Structure

An RNN consists of a network of nodes, or neurons, arranged in layers. Each neuron in a given layer is connected to every neuron in the next layer. What sets RNNs apart from feedforward neural networks is the presence of recurrent connections, where the output of a neuron at time step $$t$$ is used as input to the same neuron at time step $$t+1$$. This recurrent connection allows the network to retain information about previous inputs, effectively creating a memory.

### Recurrent Unit

A single RNN cell consists of an input vector $$x_t$$, a hidden state vector $$h_t$$, and an output vector $$y_t$$. The hidden state $$h_t$$ is updated based on the previous hidden state $$h_{t-1}$$ and the current input $$x_t$$. The update process can be described mathematically as follows:

\[ h_t = \sigma(W_h \cdot h_{t-1} + W_x \cdot x_t + b_h) \]
\[ y_t = W_y \cdot h_t + b_y \]

where $$\sigma$$ is a non-linear activation function, $$W_h$$, $$W_x$$, and $$W_y$$ are weight matrices, and $$b_h$$ and $$b_y$$ are bias vectors.

### Types

RNNs can be categorized into different types based on the input and output sequences:

- **One-to-one:** Standard neural network architecture where each input corresponds to a single output.
- **One-to-many:** Single input leads to a sequence of outputs. Example: image captioning.
- **Many-to-one:** Sequence of inputs produces a single output. Example: sentiment analysis.
- **Many-to-many (Target equal):** Sequence of inputs produces a sequence of outputs where the input and output lengths are equal. Example: video classification.
- **Many-to-many (Target not equal):** Sequence of inputs produces a sequence of outputs where the input and output lengths are different. Example: machine translation.

<table>
    <tr>
        <td align="center">Type</td>
        <td align="center">Diagram</td>
    </tr>
    <tr>
        <td align="center">One-to-one</td>
        <td><img src="/RNN/img/3.png" width="512"></td>
    </tr>
    <tr>
        <td align="center">One-to-many</td>
        <td><img src="/RNN/img/4.png" width="512"></td>
    </tr>
        <tr>
        <td align="center">Many-to-one</td>
        <td><img src="/RNN/img/5.png" width="512"></td>
    </tr>
        <tr>
        <td align="center">Many-to-many<br>(Target eq)</td>
        <td><img src="/RNN/img/6.png" width="512"></td>
    </tr>
        <tr>
        <td align="center">Many-to-many<br>(Target neq)</td>
        <td><img src="/RNN/img/7.png" width="512"></td>
    </tr>
</table>

### Advanced Layers

- **LSTM Layer:** LSTM networks include special units called memory cells that can maintain their state over long periods. An LSTM cell contains three gates: input gate, forget gate, and output gate, which control the flow of information.

- **GRU Layer:** GRU is a simplified version of LSTM with two gates: update gate and reset gate. This architecture makes GRUs computationally more efficient while still addressing the vanishing gradient problem.

### Additional Layers

- **Dense Layer:** After the recurrent layers, a dense layer processes the output to produce the final result. This fully connected layer helps in transforming the high-dimensional data into a suitable output format.
  
- **Dropout Layer:** Dropout layers are used to prevent overfitting by randomly setting a fraction of input units to zero during training. This encourages the network to develop redundant representations and improves generalization.

## TensorFlow Implementation

### Model Definition

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

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

```python
epochs = 25
batch_size = 64

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
```

## Reference

- [github.com/afshinea/stanford-cs-230-deep-learning](https://github.com/afshinea/stanford-cs-230-deep-learning)