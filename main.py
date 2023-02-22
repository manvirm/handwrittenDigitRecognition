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
x_test = tf.keras.utils.normalize(x_test, aixs = 1)