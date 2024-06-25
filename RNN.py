# Import necessary libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Set the maximum number of words to be used (top words)
max_words = 10000
# Set the maximum number of words in each review
maxlen = 200

# Load the IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_words)

# Pad sequences to ensure uniform input length
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Split the training set to create a validation set
X_train, X_val = X_train[:20000], X_train[20000:]
y_train, y_val = y_train[:20000], y_train[20000:]

# Define the model
model = Sequential()

# Input shape is (maxlen,) for each review, output dimension is 1 for binary classification
model.add(Embedding(input_dim=max_words, output_dim=32, input_length=maxlen))
model.add(SimpleRNN(units=50, activation='tanh'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define training parameters
epochs = 5
batch_size = 32

# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
