# -*- coding: utf-8 -*-
"""xor2020_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1K0fP_adikJuOiCKCgptPymAPSdMQim0H
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from tensorflow.keras import optimizers

# 1. It is trying to learn a weight based estimate of the functio for 3 input XOR which is: F = C. This is also why it converges so quick as well since we want it to overfit

# ADDED: Random seed so weight init will always be the same
tf.random.set_seed(69)

# OLD TRAINING DATA
# training_data = np.array([[0,0],[0,1],[1,0],[1,1]])
# target_data = np.array([[0],[1],[1],[0]])

# ADDED: Changed training and target data to new stuff
training_data = tf.constant([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
target_data = np.array([[0],[1],[0],[1],[0],[1],[0],[1]])

# Two inputs, two hidden units and 1 output
model = Sequential()

# ADDED: Changed input dimension
model.add(Dense(2, input_dim=3, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

# ADDED: Increased learning rate since we want this to overfit since input is fixed and allows for faster training.
sgd = optimizers.SGD(lr=0.1,  momentum=0.8)

# ADDED: Optimizer was set to a string instead of the optimizer variable above.
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

# You will change number of epochs. You will want to use verbose=2 to see
# results for each epoch sometimes but it is very slow
print("Xor Starting")

# ADDED: Kept this the same, decreased epochs since I increased the learning rate and we have more features it converges faster.
model.fit(training_data, target_data, epochs=50,verbose=2)

# ADDED: prints original output, rounded outputs, and weights
pred = model.predict(training_data)
print(f'\nPrediction unrounded:\n{pred}')
print(f'\nPrediction rounded:\n{pred.round()}')
print(f'\nWeights:\n{model.get_weights()}')



