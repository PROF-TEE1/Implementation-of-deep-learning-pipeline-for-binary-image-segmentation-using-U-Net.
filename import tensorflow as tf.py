import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create the MLP model
model = Sequential([
    Dense(100, activation='relu', input_shape=(784,)),  # Hidden Layer 1
    Dense(10, activation='softmax')  # Output Layer
])