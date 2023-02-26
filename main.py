import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load data and split into training and testing
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# Create Neural Network Model
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# Hidden layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
# Output layer (10 digits)
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('handwritten.model')

# Load model
#model = tf.keras.models.load_model('handwritten.model')
