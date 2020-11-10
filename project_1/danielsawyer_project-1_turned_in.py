''' Name and Info
Daniel Sawyer
danielsawyer@usf.edu
Data Mining Fall 2020
CNN MNIST
'''

''' Results 
Original:
1. test loss, test acc: 0.12075304985046387, 0.9613000154495239

2. test loss, test acc: 0.11850325763225555, 0.9642000198364258

3. test loss, test acc: 0.11949807405471802, 0.9650999903678894

Improved:
1. test loss, test acc: 0.05425889790058136, 0.9902999997138977

2. test loss, test acc: 0.037460219115018845, 0.9921000003814697

3. test loss, test acc: 0.04055272042751312, 0.9901000261306763
'''

''' Why it is better?
Well, there are quite a few reasons this is getting better performance than the original model. First, there are convolutional layers which extract feature discriptors that allow for better identification of the discerning features between the classes of numbers [0,9]. This allows for more robust detection in the unseen test set. Second I added batch normalization and dropout which help to prevent overfitting and give a more accurate result on the test sets. Finally, I used checkpoint callbacks which only save the best model/epoch and increased the epochs to 100. I was able to consistently get 99+% accuracy on the test set.

'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from tensorflow.keras import optimizers
import tensorflow.keras as tfk

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

# ADDED: Random seed so weight init will always be the same
# tf.random.set_seed(69)

def normalize_img(image, label):
	"""Normalizes images: `uint8` -> `float32`."""
	return tf.cast(image, tf.float32) / 255., label

# ADDED: Created function to load MNIST data
def load_mnist():
	(ds_train, ds_valid, ds_test), ds_info = tfds.load(
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


	ds_valid = ds_valid.map(
			normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	ds_valid = ds_valid.batch(64)
	ds_valid = ds_valid.cache()
	ds_valid = ds_valid.prefetch(tf.data.experimental.AUTOTUNE)

	return ds_train, ds_valid, ds_test

# ADDED: main function, not using main function is sloppy python code
def main():
	# ADDED: Loads data
	ds_train, ds_valid, ds_test = load_mnist()

	# ADDED: Callbacks for saving best epoch model
	model_path = 'mnist-checkpoint.h5'
	checkpoint = tfk.callbacks.ModelCheckpoint(
		filepath=model_path, 
		monitor='val_accuracy', 
		verbose=1, 
		save_best_only=True
	)
	callbacks = [checkpoint]

	### ADDED: Creates model START###
	# Input shape and layer
	input_shape = (28, 28, 1)
	input_layer = tfk.layers.Input(shape=input_shape)

	# First convolution and batch norm
	l = tfk.layers.Conv2D(32, 3)(input_layer)
	l = tfk.layers.Activation('relu')(l)
	l = tfk.layers.BatchNormalization()(l)

	# Second convolution and batchnorm
	l = tfk.layers.Conv2D(64, 3)(l)
	l = tfk.layers.Activation('relu')(l)
	l = tfk.layers.BatchNormalization()(l)

	# Max pooling layer and flattens for dense layers
	l = tfk.layers.AveragePooling2D()(l)
	l = tfk.layers.Flatten()(l)

	# First dense layer with dropout
	l = tfk.layers.Dense(128, activation='relu')(l)
	l = tfk.layers.Dropout(0.5)(l)

	# Output dense layer, 10 classes
	output_layer = tfk.layers.Dense(10, activation='softmax')(l)
	### ADDED: Creates model END###

	# ADDED: Compiles model with adamax optimizer, learning rate 0.001
	model = tfk.Model(input_layer, output_layer)
	opt = tfk.optimizers.Adamax(learning_rate=0.001)
	model.compile(
		loss='sparse_categorical_crossentropy',
		optimizer=opt,
		metrics=['accuracy']
	)

	# ADDED: Prints model summary
	model.summary()

	# ADDED: Trains model with increased epochs and saves best
	model.fit(
		ds_train,
		epochs=100,
		validation_data=ds_valid,
		callbacks = callbacks
	)

	# ADDED: Load best saved model from checkpoint callbacks
	model_best = tfk.models.load_model(model_path)

	# ADDED: Evaluates best model and prints accuracy
	results = model_best.evaluate(ds_test, batch_size=128)
	print(f'test loss, test acc: {results[0]}, {results[1]}')

if __name__ == '__main__':
	main()