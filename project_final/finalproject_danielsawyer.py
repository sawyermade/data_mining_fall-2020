import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow.keras as tfk
import sys

IMG_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

# # Normalize function from professor
# def normalize_img(image, label):
# 	"""Normalizes images: `uint8` -> `float32`."""
# 	return tf.cast(image, tf.float32) / 255., label

# def resize_and_rescale(image, label):
# 	image = tf.cast(image, tf.float32)
# 	image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
# 	image = (image / 255.0)
# 	return image, label

# def augment(image,label):
# 	image, label = resize_and_rescale(image, label)
# 	# Add 6 pixels of padding
# 	image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6) 
# 	# Random crop back to the original size
# 	image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
# 	image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
# 	image = tf.clip_by_value(image, 0, 1)
# 	return image, label

def loadmycifar100():
	(ds_train, ds_valid, ds_test), ds_info = tfds.load(
		'cifar100',
		split=['train[:45%]+train[-45%:]','train[45%:55%]', 'test'],
		shuffle_files=True,
		as_supervised=True,
		with_info=True
	)

	resize_and_rescale = tfk.Sequential([
		tfk.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
		tfk.layers.experimental.preprocessing.Rescaling(1./255)
	])

	data_augmentation = tfk.Sequential([
		tfk.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
		tfk.layers.experimental.preprocessing.RandomRotation(0.2),
	])

	# ds_train = ds_train.map(lambda x, y: (resize_and_rescale(x, training=True), y), num_parallel_calls=AUTOTUNE)
	# ds_train = ds_train.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
	ds_train = ds_train.cache()
	ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
	ds_train = ds_train.batch(128)
	# ds_train = ds_train.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
	ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
	
	# ds_test = ds_test.map(
	# 	normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
	# )
	ds_test = ds_test.batch(128)
	ds_test = ds_test.cache()
	ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

	# ds_valid = ds_valid.map(
	# 	normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
	# )
	ds_valid = ds_valid.batch(64)
	ds_valid = ds_valid.cache()
	ds_valid = ds_valid.prefetch(tf.data.experimental.AUTOTUNE)

	return ds_train, ds_valid, ds_test

def load_model_og():
	# Input layer
	input_shape = (32, 32, 3)
	input_layer = tfk.layers.Input(shape=input_shape)

	# First conv layers
	ly = tfk.layers.Conv2D(32, 3)(input_layer)
	ly = tfk.layers.BatchNormalization()(ly)
	ly = tfk.layers.Activation('relu')(ly)

	# Max pooling
	ly = tfk.layers.MaxPooling2D()(ly)

	# Second conv layers
	ly = tfk.layers.Conv2D(64, 3)(ly)
	ly = tfk.layers.BatchNormalization()(ly)
	ly = tfk.layers.Activation('relu')(ly)

	# Third conv layers
	ly = tfk.layers.Conv2D(64, 3)(ly)
	ly = tfk.layers.BatchNormalization()(ly)
	ly = tfk.layers.Activation('relu')(ly)

	# Pooling and flatten layers
	ly = tfk.layers.MaxPooling2D()(ly)
	ly = tfk.layers.Flatten()(ly)

	# Dense layers
	ly = tfk.layers.Dense(512)(ly)
	ly = tfk.layers.BatchNormalization()(ly)
	ly = tfk.layers.Activation('relu')(ly)
	ly = tfk.layers.Dropout(0.5)(ly)

	# Output layers
	ly = tfk.layers.Dense(100)(ly)
	output_layer = tfk.layers.Activation('softmax')(ly)

	# Builds model
	model = tfk.Model(input_layer, output_layer)
	opt = tfk.optimizers.Adam(learning_rate=0.001)
	model.compile(
		loss='sparse_categorical_crossentropy',
		optimizer=opt,
		metrics=['accuracy']
	)

	# Prints model summary
	model.summary()

	return model

def load_model_0():
	# Input shape and layer
	input_shape = (32, 32, 3)
	input_layer = tfk.layers.Input(shape=input_shape)

	# First convolution, batch norm, and dropout
	l = tfk.layers.Conv2D(32, 3)(input_layer)
	# l = tfk.layers.BatchNormalization()(l)
	l = tfk.layers.Activation('relu')(l)
	l = tfk.layers.BatchNormalization()(l)
	# l = tfk.layers.MaxPooling2D()(l)
	# l = tfk.layers.Dropout(0.1)(l)

	# Second convolution, batch norm, and dropout
	l = tfk.layers.Conv2D(64, 3)(l)
	# l = tfk.layers.BatchNormalization()(l)
	l = tfk.layers.Activation('relu')(l)
	l = tfk.layers.BatchNormalization()(l)
	l = tfk.layers.MaxPooling2D()(l)
	l = tfk.layers.Dropout(0.2)(l)

	# Third convolution, batch norm, and dropout
	l = tfk.layers.Conv2D(128, 3)(l)
	# l = tfk.layers.BatchNormalization()(l)
	l = tfk.layers.Activation('relu')(l)
	l = tfk.layers.BatchNormalization()(l)
	# l = tfk.layers.MaxPooling2D()(l)
	# l = tfk.layers.Dropout(0.1)(l)

	# Fourth convolution, batch norm, and dropout
	l = tfk.layers.Conv2D(256, 3)(l)
	# l = tfk.layers.BatchNormalization()(l)
	l = tfk.layers.Activation('relu')(l)
	l = tfk.layers.BatchNormalization()(l)
	l = tfk.layers.MaxPooling2D()(l)
	l = tfk.layers.Dropout(0.2)(l)

	# Flattens for dense layers
	# l = tfk.layers.Dropout(0.1)(l)
	l = tfk.layers.Flatten()(l)

	# First dense layer with batch norm & dropout
	l = tfk.layers.Dense(512)(l)
	l = tfk.layers.BatchNormalization()(l)
	l = tfk.layers.Activation('relu')(l)
	# l = tfk.layers.BatchNormalization()(l)
	l = tfk.layers.Dropout(0.7)(l)

	# Output dense layer, 100 classes
	l = tfk.layers.Dense(100)(l)
	output_layer = tfk.layers.Activation('softmax')(l)

	# Compiles model with adam optimizer
	model = tfk.Model(input_layer, output_layer)
	opt = tfk.optimizers.Adam(learning_rate=0.001)
	# opt = tfk.optimizers.SGD(learning_rate=0.1, nesterov=True, momentum=0.0)
	model.compile(
		loss='sparse_categorical_crossentropy',
		optimizer=opt,
		metrics=['accuracy']
	)

	# Prints model summary
	model.summary()

	return model

def main():
	# Gets epochs if passed as cmd line arg
	if len(sys.argv) > 1:
		epochs = int(sys.argv[1])
	else:
		epochs = 15

	# Load data
	ds_train, ds_valid, ds_test = loadmycifar100()

	# # Loads original model
	# model_path_og = 'cifar100_best_ckeckpt_og.h5'
	# model_og = load_model_og()

	# # Set up checkpoint callbacks for best model
	# checkpoint_og = tfk.callbacks.ModelCheckpoint(
	# 	filepath=model_path_og, 
	# 	monitor='val_accuracy',
	# 	verbose=1,
	# 	save_best_only=True
	# )
	# callbacks_og = [checkpoint_og]

	# # Trains model
	# model_og.fit(
	# 	ds_train,
	# 	epochs=epochs,
	# 	validation_data=ds_valid,
	# 	callbacks=callbacks_og
	# )

	# # Load best model weights from checkpoint and save results
	# model_og.load_weights(model_path_og)
	# res = model_og.evaluate(ds_test, batch_size=128)
	# print(f'\nRESULTS: test loss, test acc: {res[0] : 4.4f}, {res[1] : 4.4f}\n')

	model_path_0 = 'cifar100_best_ckeckpt_model_0.h5'
	model_0 = load_model_0()

	checkpoint_0 = tfk.callbacks.ModelCheckpoint(
		filepath=model_path_0, 
		monitor='val_accuracy',
		verbose=1,
		save_best_only=True
	)
	callbacks_0 = [checkpoint_0]

	model_0.fit(
		ds_train,
		epochs=epochs,
		validation_data=ds_valid,
		callbacks=callbacks_0
	)

	model_0.load_weights(model_path_0)
	res = model_0.evaluate(ds_test, batch_size=128)
	print(f'\nRESULTS: test loss, test acc: {res[0] : 4.4f}, {res[1] : 4.4f}\n')

if __name__ == '__main__':
	main()