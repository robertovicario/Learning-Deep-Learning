# Recurrent Neural Network (RNN)

## Overview

Recurrent Neural Networks (RNNs) are a type of neural network architecture designed for sequential data. Unlike traditional feedforward neural networks, RNNs have connections that form directed cycles, allowing them to maintain a hidden state that can capture information about previous inputs. This makes them particularly well-suited for tasks where the order of the data matters, such as time series prediction, natural language processing, and more.

## Architecture

...

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

...

<table>
    <tr>
        <td align="center">One-to-one</td>
        <td><img src="/RNN/img/3.png" width="128"></td>
    </tr>
    <tr>
        <td align="center">One-to-many</td>
        <td><img src="/RNN/img/4.png" width="128"></td>
    </tr>
        <tr>
        <td align="center">Many-to-one</td>
        <td><img src="/RNN/img/5.png" width="128"></td>
    </tr>
        <tr>
        <td align="center">Many-to-many</td>
        <td><img src="/RNN/img/6.png" width="128"></td>
    </tr>
        <tr>
        <td align="center">Many-to-many</td>
        <td><img src="/RNN/img/7.png" width="128"></td>
    </tr>
</table>

### Simple Layer

The basic RNN layer comprises neurons that receive input from both the current time step and the hidden state from the previous time step. The hidden state is updated at each time step based on the current input and the previous hidden state, allowing the network to maintain a memory of past inputs.

RNN models are widely used in natural language processing and speech recognition. Their applications can be categorized as follows:

<table>
    <tr>
        <td>One-to-one<br> $T_x = T_y = 1$ </td>
        <td><img src="/RNN/img/3.png" width="128"></td>
    </tr>
    <tr>
        <td>One-to-many<br> $T_x = 1, T_y > 1$ </td>
        <td><img src="/RNN/img/4.png" width="128"></td>
    </tr>
    <tr>
        <td>Many-to-one<br> $T_x > 1, T_y = 1$ </td>
        <td><img src="path/to/RNN/img/5.png" width="128"></td>
    </tr>
    <tr>
        <td>Many-to-many<br> $T_x = T_y$ </td>
        <td><img src="path/to/RNN/img/6.png" width="128"></td>
    </tr>
    <tr>
        <td>Many-to-many<br> $T_x \ne T_y$ </td>
        <td><img src="path/to/RNN/img/7.png" width="128"></td>
    </tr>
</table>

### LSTM Layer

Long Short-Term Memory (LSTM) networks are a type of RNN designed to capture long-range dependencies and mitigate the vanishing gradient problem. LSTMs introduce a more complex cell structure, which includes gates that regulate the flow of information.

### GRU Layer

Gated Recurrent Unit (GRU) is a simplified version of LSTM that combines the forget and input gates into a single update gate and merges the cell state and hidden state. This makes GRUs computationally more efficient while still addressing the vanishing gradient problem.

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

## Reference

- [github.com/afshinea/stanford-cs-230-deep-learning](https://github.com/afshinea/stanford-cs-230-deep-learning)
