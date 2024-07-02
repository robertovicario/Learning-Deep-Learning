from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Preprocessing
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

max_length = 500
X_train = pad_sequences(X_train, maxlen=500)
X_test = pad_sequences(X_test, maxlen=500)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = 2
input_shape = (max_length, 1)

# Reshape input data to 3D
X_train = X_train.reshape(X_train.shape[0], max_length, 1)
X_test = X_test.reshape(X_test.shape[0], max_length, 1)

# Model Definition
model = Sequential()

model.add(SimpleRNN(units=50, activation='tanh', input_shape=input_shape))  # Input Layer

model.add(Dense(num_classes, activation='softmax'))  # Output Layer

model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

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

# Plot History
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
