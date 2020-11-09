# -*- coding: utf-8 -*-
"""project1f20v1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12diLKJP3O_MxcbNlAZY0qMpWWrvkmkei
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from tensorflow.keras import optimizers

'''
You need to run this code (as is) 3 times and report the average accuracy on
the test data. This is the MNIST dataset with 30,000 training (1/2 not 
available) with 10,000 test images. Then you can add another dense layer
(or more) if you like and document that performance.  That is not necessary
 though.

You MUST create a convolutional model with at least 2 convolutional layers and 
at least 1 max or average pooling layer.  You should have, for this assignment,
a dense layer before your outputs.  Your goal is to achieve better accuracy
than this simple classifier.  You MUST leave the train set (30K examples) and
test set (10K examples) unchanged.  You can do whatever you like with validation
data.  I have included 3K examples not in test or train for validation.
I used tensorflow more than keras in dealing with the data below.  If 
interested in what I am doing below you can start with the links below.

https://colab.research.google.com/github/tensorflow/datasets/blob/master/docs/overview.ipynb
https://www.tensorflow.org/datasets/catalog/overview

CNN Example using Keras:
https://www.tensorflow.org/tutorials/images/cnn

You MUST provide the python (py) file that starts with your netid.  You MUST
put your name in the file.  You MUST say what you did, what result you got
and why you think it was better.  You MUST have in comments the results
you got as well as the previous requested information.


'''

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


(ds_train, dsvalid, ds_test), ds_info = tfds.load(
    'mnist',
# First 25% and last 25% from training, then validation data is 5%
# from 25% of train data to 30% and test is the usual 10K
    split=['train[:25%]+train[-25%:]','train[25%:30%]', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


dsvalid = dsvalid.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dsvalid = dsvalid.batch(64)
dsvalid = dsvalid.cache()
dsvalid = dsvalid.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

model.fit(
    ds_train,
    epochs=6,
#    validation_data=ds_test,
    validation_data=dsvalid,
)
results = model.evaluate(ds_test, batch_size=128)
print("test loss, test acc:", results)