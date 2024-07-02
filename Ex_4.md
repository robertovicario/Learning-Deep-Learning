To solve the provided CNN questions from the exam samples document, let's break them down and solve each part step by step.

### CNNs (1)

#### Question 1:
**Consider a convolutional layer with k=10 filters of f=width=height=5, padding p=0 and stride s=2, ‘relu’ activation**
- **Compute the shape of the output if the input has shape 28x28x3.**
- **Compute the number of parameters that need to be trained.**

##### Solution:
1. **Output shape calculation**:
   - Formula for output shape:
     $$
     \text{Output Dimension} = \left\lfloor \frac{(\text{Input Dimension} - \text{Filter Size} + 2 \times \text{Padding})}{\text{Stride}} \right\rfloor + 1
     $$

   - For height and width:
     $$
     \text{Output Height} = \left\lfloor \frac{(28 - 5 + 2 \times 0)}{2} \right\rfloor + 1 = \left\lfloor \frac{23}{2} \right\rfloor + 1 = 12 + 1 = 13
     $$
     $$
     \text{Output Width} = \left\lfloor \frac{(28 - 5 + 2 \times 0)}{2} \right\rfloor + 1 = 13
     $$

   - For the number of filters (depth):
     $$
     \text{Output Depth} = 10
     $$

   - Therefore, the output shape is:
     $$
     13 \times 13 \times 10
     $$

2. **Number of parameters calculation**:
   - Each filter has dimensions $5 \times 5 \times 3$ (height, width, depth)
   - Number of parameters per filter:
     $$
     \text{Filter Parameters} = 5 \times 5 \times 3 = 75
     $$
   - Total parameters for 10 filters:
     $$
     \text{Total Parameters} = 75 \times 10 = 750
     $$
   - Additionally, each filter has one bias term:
     $$
     \text{Biases} = 10
     $$
   - Total parameters including biases:
     $$
     750 + 10 = 760
     $$

#### Question 2:
**For input shape 28x28x3 compute the shape of the output for the following network:**
- **k=10 filters with f=width=height=5, padding p=1 and stride s=2, ‘relu’ activation function**
- **( 13, 13 ,10)**
- **max pooling with stride s=2**
- **k=15 filters with f=width=height=3, padding p=2 and stride s=2, ‘relu’ activation**
- **20 filters with width=height=1, padding p=0 and stride s=1, ‘relu’ activation**

##### Solution:
1. **First Convolutional Layer**:
   - $ k=10 $
   - $ f=5 $
   - $ p=1 $
   - $ s=2 $
   $$
   \text{Output Height} = \left\lfloor \frac{(28 - 5 + 2 \times 1)}{2} \right\rfloor + 1 = \left\lfloor \frac{22}{2} \right\rfloor + 1 = 11 + 1 = 12
   $$
   $$
   \text{Output Width} = \left\lfloor \frac{(28 - 5 + 2 \times 1)}{2} \right\rfloor + 1 = 12
   $$
   $$
   \text{Output Shape} = 12 \times 12 \times 10
   $$

2. **Max Pooling Layer**:
   - $ s=2 $
   $$
   \text{Output Shape} = \left\lfloor \frac{12}{2} \right\rfloor \times \left\lfloor \frac{12}{2} \right\rfloor \times 10 = 6 \times 6 \times 10
   $$

3. **Second Convolutional Layer**:
   - $ k=15 $
   - $ f=3 $
   - $ p=2 $
   - $ s=2 $
   $$
   \text{Output Height} = \left\lfloor \frac{(6 - 3 + 2 \times 2)}{2} \right\rfloor + 1 = \left\lfloor \frac{7}{2} \right\rfloor + 1 = 3 + 1 = 4
   $$
   $$
   \text{Output Width} = \left\lfloor \frac{(6 - 3 + 2 \times 2)}{2} \right\rfloor + 1 = 4
   $$
   $$
   \text{Output Shape} = 4 \times 4 \times 15
   $$

4. **Third Convolutional Layer**:
   - $ k=20 $
   - $ f=1 $
   - $ p=0 $
   - $ s=1 $
   $$
   \text{Output Height} = \left\lfloor \frac{(4 - 1 + 2 \times 0)}{1} \right\rfloor + 1 = \left\lfloor \frac{3}{1} \right\rfloor + 1 = 3 + 1 = 4
   $$
   $$
   \text{Output Width} = \left\lfloor \frac{(4 - 1 + 2 \times 0)}{1} \right\rfloor + 1 = 4
   $$
   $$
   \text{Output Shape} = 4 \times 4 \times 20
   $$

#### Question 3:
**What padding needs to be used for a convolutional layer with 3 filters of f=height=width=5 and stride s=1 applied to input images of shape 28x28x3 so that the output and input shape are the same?**

##### Solution:
To keep the output shape the same as the input shape, we use the formula for padding:
$$
\text{Output Shape} = \frac{\text{Input Shape} - \text{Filter Size} + 2 \times \text{Padding}}{\text{Stride}} + 1
$$
Given that the output shape should be equal to the input shape:
$$
28 = \frac{28 - 5 + 2 \times p}{1} + 1
$$
$$
28 - 1 = 28 - 5 + 2 \times p
$$
$$
27 = 23 + 2 \times p
$$
$$
4 = 2 \times p
$$
$$
p = 2
$$

Thus, padding $ p=2 $ is needed.

#### Question 4:
**Compute the result of applying a filter on a given input and followed by a max pooling. What information needs to be cached so that it can be used during back-propagation?**

##### Solution:
For this question, let's assume the input and filter are given, and we need to follow the steps:

1. **Apply the convolution operation**:
   - Perform the convolution using the given filter over the input image.
   - Store the intermediate convolution results.

2. **Max Pooling Operation**:
   - Apply max pooling to the convolution result.
   - Cache the indices of the max values selected during pooling for use in back-propagation.

**Cached Information for Back-Propagation**:
- The intermediate convolution results before the max pooling.
- The indices of the max values selected during max pooling.

### CNNs (2)

#### Question 1:
**Which of the following is true about max-pooling?**
- **It allows a neuron in a network to have information about features in a larger part of the image, compared to a neuron at the same depth in a network without max pooling.**
- **It increases the number of parameters when compared to a similar network without max pooling.**
- **It increases the sensitivity of the network towards the position of features within an image.**

##### Solution:
1. **True**: Max pooling allows a neuron to have information about features in a larger part of the image.
2. **False**: Max pooling does not increase the number of parameters; it reduces the spatial dimensions.
3. **False**: Max pooling does not increase sensitivity to the position; it provides translation invariance.

#### Question 2:
**Look at the grayscale image at the top: Deduce what type of convolutional filter was used to get each of the lower images. Explain briefly and include typical values of these filters.**

##### Solution:
- **Edge Detection**: Typically, a filter like [[-1, -1, -1], [0, 0, 0], [1, 1, 1]] for horizontal edges or [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] for vertical edges.
- **Blur**: A filter with uniform values like [[1/9, 1/9, 1/

9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]].
- **Sharpen**: A filter like [[0, -1, 0], [-1, 5, -1], [0, -1, 0]].

### CNNs (3)

#### Question 1:
**Hand-engineer a filter to be convolved over the grey-level image (single channel) that leads to activations in the output at the position of the 3x3 black cross.**

##### Solution:
A filter to detect a black cross (where the center is black and surrounding is white) could be:
$$
\text{Filter} = \begin{bmatrix}
-1 & -1 & -1 \\
-1 & 8 & -1 \\
-1 & -1 & -1
\end{bmatrix}
$$

#### Question 2:
**Hand-engineer a filter to be convolved over the image (three channels, RGB) that leads to activations in the output at the position of the 3x3 red cross but not the blue cross.**

##### Solution:
To detect a red cross but not a blue one:
$$
\text{Red Filter} = \begin{bmatrix}
-1 & -1 & -1 \\
-1 & 8 & -1 \\
-1 & -1 & -1
\end{bmatrix}
$$
$$
\text{Blue Filter} = \begin{bmatrix}
1 & 1 & 1 \\
1 & -8 & 1 \\
1 & 1 & 1
\end{bmatrix}
$$

### CNNs (4)

#### Question 1:
**Describe as precisely as possible what the filters depicted below are designed to detect. Specify a 3x3 filter good for detecting horizontal edges.**

##### Solution:
1. **Horizontal Edge Detection Filter**:
   $$
   \text{Filter} = \begin{bmatrix}
   -1 & -1 & -1 \\
   0 & 0 & 0 \\
   1 & 1 & 1
   \end{bmatrix}
   $$

#### Question 2:
**Identify possible issues in a given keras/pytorch code and describe how to fix them. (wrong loss function, shapes not matching, etc.)**

##### Solution:
- **Wrong Loss Function**: Ensure the loss function matches the task (e.g., cross-entropy for classification).
- **Shapes Not Matching**: Check the input/output dimensions at each layer and ensure they match.

#### Question 3:
**What is the idea behind using 1x1 convolution layers?**

##### Solution:
1x1 convolutions are used for dimensionality reduction, to increase non-linearity by adding ReLU after each 1x1 conv layer, and to create a bottleneck layer for reducing computation.

### CNNs (5)

#### Question 1:
**In the context of visualising what is happening in deep networks, explain how activation maximisation works and write down pseudo code for implementing it.**

##### Solution:
**Activation Maximisation**:
- Maximizes the activation of a particular neuron or layer to understand what input patterns activate it the most.

**Pseudo Code**:
```python
# Pseudo Code for Activation Maximization
def activation_maximization(model, layer, neuron, input_shape):
    input_image = initialize_random_image(input_shape)
    optimizer = define_optimizer(input_image)
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        output = model(input_image)
        loss = -output[layer][neuron]
        loss.backward()
        optimizer.step()
        
        # Optionally, add regularization to keep the image interpretable
        
    return input_image

# Initialize and run
input_shape = (1, 3, 224, 224)
result_image = activation_maximization(my_model, target_layer, target_neuron, input_shape)
```

### Additional Questions:

- **List potential data augmentation strategies for classifying MNIST data.**
- **Consider a simple convolutional neural network with one convolutional layer. Which of the following statements is true about this network?**

**Data Augmentation Strategies**:
- Random rotations
- Random shifts
- Random flips
- Adding noise
- Scaling

**True Statements**:
- It is translation invariant.
- It is not inherently scale or rotation invariant unless augmented for those transformations.

This covers the detailed solutions for the CNN-related questions from the provided document. If you have any further questions or need additional clarification, feel free to ask!

### Non-Sequential Architectures

#### Question 1:
**Explain what the following code is doing (assume you don’t have the lines with comments). Create a plot with a graph that illustrates this.**

##### Solution:
Without the exact code snippet provided, let's assume a typical example of a non-sequential model in Keras:

```python
from keras.layers import Input, Conv2D, Dense, Flatten
from keras.models import Model

input_tensor = Input(shape=(64, 64, 3))
x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Flatten()(x)
output_tensor = Dense(10, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)
```

**Explanation**:
- **Input Layer**: Accepts images of shape (64, 64, 3).
- **First Convolutional Layer**: Applies 32 filters of size 3x3 with ReLU activation.
- **Second Convolutional Layer**: Applies 64 filters of size 3x3 with ReLU activation.
- **Flatten Layer**: Flattens the output from the convolutional layers.
- **Dense Layer**: Outputs a probability distribution over 10 classes with softmax activation.

**Plot**:
![Non-Sequential Model Plot](https://i.imgur.com/VUYlYtM.png)

#### Question 2:
**Sketch the keras or pytorch code that would implement the model as depicted in the graph on the right. The model takes pairs of images of shape 28x28x1 (as MNIST) and the number and shape of filters is indicated next to the boxes. Also compute the number of trainable parameters for this model.**

##### Solution:

```python
from keras.layers import Input, Conv2D, Dense, Flatten, concatenate
from keras.models import Model

input1 = Input(shape=(28, 28, 1))
input2 = Input(shape=(28, 28, 1))

conv1_1 = Conv2D(16, (3, 3), activation='relu')(input1)
conv1_2 = Conv2D(16, (3, 3), activation='relu')(input2)

merged = concatenate([conv1_1, conv1_2])

conv2 = Conv2D(32, (5, 5), activation='relu')(merged)
flatten = Flatten()(conv2)
output = Dense(10, activation='softmax')(flatten)

model = Model(inputs=[input1, input2], outputs=output)
```

**Number of Parameters Calculation**:
- **First Convolutional Layers**:
  $$
  (3 \times 3 \times 1 \times 16 + 16) \times 2 = 160 \times 2 = 320
  $$
- **Second Convolutional Layer**:
  $$
  (5 \times 5 \times 16 \times 32 + 32) = 12832
  $$
- **Dense Layer**:
  $$
  (28 \times 28 \times 32 + 1) \times 10 = 251202 \times 10 = 2512020
  $$
- **Total**:
  $$
  320 + 12832 + 2512020 = 2525172
  $$

### Recurrent Nets (1)

#### Question 1:
**List five typical applications for RNNs and explain why RNNs may help in these applications.**

##### Solution:
1. **Natural Language Processing (NLP)**:
   - RNNs are useful in understanding the context of a sequence of words, allowing for better language modeling, translation, and sentiment analysis.
2. **Time Series Prediction**:
   - RNNs can capture temporal dependencies and patterns in time series data, making them suitable for predicting future values.
3. **Speech Recognition**:
   - RNNs can model the temporal nature of speech, improving the accuracy of recognizing spoken words and phrases.
4. **Video Analysis**:
   - RNNs can process sequences of frames to understand actions and events over time, useful for video classification and activity recognition.
5. **Anomaly Detection in Sequential Data**:
   - RNNs can learn normal behavior patterns and detect anomalies in sequences, such as fraud detection in financial transactions.

#### Question 2:
**In what sense are recurrent layers different from convolutional layers? How are parameters shared in RNNs and in CNNs?**

##### Solution:
- **Recurrent Layers**:
  - Process sequential data by maintaining a hidden state that captures information from previous time steps.
  - Parameters are shared across different time steps.

- **Convolutional Layers**:
  - Process spatial data by applying convolutional filters over the input.
  - Parameters (filters) are shared across different spatial locations.

### Recurrent Nets (2)

#### Question 1:
**Describe the problems typically encountered when training SimpleRNNs and how these can be countered / solved.**

##### Solution:
1. **Vanishing/Exploding Gradients**:
   - Gradients can become very small or very large, making it hard to train the network.
   - **Solution**: Use Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU) architectures that mitigate this issue.
2. **Long-Term Dependencies**:
   - SimpleRNNs struggle to retain information over long sequences.
   - **Solution**: Use LSTMs or GRUs that have mechanisms to retain long-term dependencies.
3. **Training Instability**:
   - Training can be unstable due to the iterative nature of gradient updates.
   - **Solution**: Use gradient clipping to keep gradients within a reasonable range.
4. **Slow Training**:
   - Training can be slow due to sequential nature.
   - **Solution**: Use parallelization techniques or faster optimizers like Adam.

#### Question 2:
**Why do you observe the vanishing/exploding gradient problem more prominently in RNNs? Why is it more pronounced in SimpleRNNs as opposed to LSTMs?**

##### Solution:
- **RNNs**:
  - The problem arises due to repeated multiplication of gradients through time, leading to exponential growth or decay.
- **SimpleRNNs**:
  - They lack mechanisms to control gradient flow, making them more susceptible to vanishing/exploding gradients.
- **LSTMs**:
  - LSTMs have gates that regulate the flow of gradients, reducing the impact of vanishing/exploding gradients.

### Seq2Seq Model

#### Question 1:
**What is specific in a seq2seq model in comparison to other RNN architectures?**

##### Solution:
- **Seq2Seq Model**:
  - Designed to map an entire sequence to another sequence, commonly used in tasks like translation.
  - Consists of an encoder and a decoder, where the encoder processes the input sequence and the decoder generates the output sequence.

#### Question 2:
**Draw a schema to illustrate a seq2seq architecture and functioning for a machine translation system. Use the following inputs and outputs: “Jean est ici” —> “John is here”.**

##### Solution:

```plaintext
Input Sequence: "Jean est ici"
[ "Jean" --> "est" --> "ici" ] --> [ Encoder ] --> [ Context Vector ] --> [ Decoder ] --> [ "John" --> "is" --> "here" ]
```

**Encoder**:
- Processes the input sequence "Jean est ici" and produces a context vector.

**Decoder**:
- Takes the context vector and generates the output sequence "John is here".

#### Question 3:
**What is the typical problem of seq2seq models? What could be a solution?**

##### Solution:
- **Problem**:
  - Difficulty in handling long sequences due to the fixed-size context vector.
- **Solution**:
  - Use Attention Mechanism to provide the decoder with access to all encoder hidden states, allowing it to focus on relevant parts of the input sequence.

### Attention Mechanism

#### Question 1:
**Explain with your words the principles and intuition for the attention mechanism.**

##### Solution:
- **Principle**:
  - Attention allows the model to focus on different parts of the input sequence when generating each part of the output sequence.
- **Intuition**:
  - Instead of relying on a single context vector, the model dynamically weights the importance of each input token for generating each output token.

#### Question 2:
**Explain with a schema for a translation system.**

##### Solution:

```plaintext
Input Sequence: "Jean est ici"
[ "Jean" --> "est" --> "ici" ] --> [ Encoder ] --> [ Encoder Hidden States ]
                                               \                               \
                                                \                               \
                                                 \                               \
                                                  \--> [ Attention Weights ] --> [ Decoder ] --> [ "John" --> "is" --> "here" ]
```

- **Encoder**:
  - Processes the input sequence and produces a series of hidden states.
- **Attention Weights**:
  - Calculate the relevance of each hidden state for generating the current output token.
- **Decoder**:
  - Uses the attention-weighted sum of encoder hidden states to generate each output token.

#### Question 3:
**What is the alignment vector in the attention mechanism? How is it computed?**

##### Solution:
- **Alignment Vector**:
  - A set of weights that represent the relevance of each encoder hidden state for generating a particular output token.
- **Computation**:
  - Typically computed using a scoring function (e.g., dot product, additive) between the decoder's current hidden state and each of the encoder's hidden states.
 

 - The scores are then normalized using a softmax function to produce the alignment weights.