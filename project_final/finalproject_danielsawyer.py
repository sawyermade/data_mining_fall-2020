import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import sys
# import matplotlib.pyplot as plt
# from PIL import Image

''' Explanation
# Data loading:
I used the keras datasets instead of tfds since it spits out easy to manipulate numpy arrays. I then normalized the values to [-1.0, 1.0] as float32s for better accuracy. I was planning on doing data augmentation but ended up not having time and this would of been much easier to do it with. I am, however, doing a 10% split for validation randomly through keras, so it is functionally the same to what you provided in the og code but in a format I personally prefer.

# Model 15 Epochs:
For the model, since we only have 15 epochs to improve, I decided to go with a wide network over a deep one. Triangle shaped, if you will. I do this by increasing convolution filter sizes starting with 256 and ending at 512. I then add a dense layer that widens even further to 1024. With only 4 conv layers and 2 dense(1 being the classification layer) layes, it has 17M params haha. Since we have a high number of classes, low number of samples for each class, and low max epochs I found it better to use more filters and less layers which allows for better fine-grained feature detection. Another problem is the small size of the images, which don't allow for too many conv layers due to the max pooling reducing the image size over and over again. At 32x32, we honestly dont have much room to be pooling it down to 3x3 and honestly do not want to get that small anyways since there isnt really much information to be had from it. I also have batchnorm and dropout to help prevent overfitting. In the end, I run the test set and get per sample predictions and compare to the ground truth. I then find the best and worst performing classes, then compare that overall accuracy to the keras evaluate accuracy to make sure they are the same value and that I'm doing it right. With this model at 15 epochs, I am able to consistently reach 58% for 100 classes.

# Model X Epochs, best performance, and snapshots:
I was unable to get my model for best performance with data aug and snapshots complete in time. I had another final project with a proper final report and a 30 minute presentation for another class due last week and was unable to complete the extra credit portions in time for submission :(
'''

# Keras cifar 100 loading and normalization from [-1.0, 1.0]
def load_keras_cifar100():
	(x_train, y_train), (x_test, y_test) = tfk.datasets.cifar100.load_data(label_mode='fine')

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	# for i, x in enumerate(x_test[:5]):
	# 	path = f'test-{str(i).zfill(5)}.png'
	# 	Image.fromarray(x).save(path)
	# print(f'\nx shape, y shape: {x_test.shape}, {y_test.shape}\n')
	# sys.exit()

	x_train -= 128
	x_test -= 128
	x_train /= 128
	x_test /= 128

	return (x_train, y_train, x_test, y_test)

# 15 epoch wide model, not very deep
def load_model_15(model_path, epochs, x_train, y_train):
	### Hyper params ###

	# Activation
	act = 'relu'

	# Padding
	# pad = 'same'
	pad = 'valid'

	# Batch Norm and placement. Max of one true at a time
	bn1 = True
	bn2 = False

	# Dropout values for convs and denses
	drop_c = 0.3
	drop_d = 0.7

	# Learning rate
	lr = 0.001

	# Conv width/filters
	conv1 = 256
	conv2 = 256
	conv3 = 512
	conv4 = 512

	# Dense parameter
	d1 = 1024

	# Number of classes
	c = 100


	# Input shape and layer
	img_shape_x, img_shape_y = x_train.shape[1:3]
	input_shape = (img_shape_x, img_shape_y, 3)
	input_layer = tfk.layers.Input(shape=input_shape)

	# 1st conv
	l = tfkl.Conv2D(conv1, 3, padding=pad)(input_layer)
	if bn1: l = tfkl.BatchNormalization()(l)
	l = tfkl.Activation(act)(l)
	if bn2: l = tfkl.BatchNormalization()(l)

	# 2nd conv
	l = tfkl.Conv2D(conv2, 3)(l)
	if bn1: l = tfkl.BatchNormalization()(l)
	l = tfkl.Activation(act)(l)
	if bn2: l = tfkl.BatchNormalization()(l)
	
	# Pooling and dropout
	l = tfkl.MaxPooling2D()(l)
	l = tfkl.Dropout(drop_c)(l)


	# 3rd conv
	l = tfkl.Conv2D(conv3, 3, padding=pad)(l)
	if bn1: l = tfkl.BatchNormalization()(l)
	l = tfkl.Activation(act)(l)
	if bn2: l = tfkl.BatchNormalization()(l)

	# 4th conv
	l = tfkl.Conv2D(conv4, 3)(l)
	if bn1: l = tfkl.BatchNormalization()(l)
	l = tfkl.Activation(act)(l)
	if bn2: l = tfkl.BatchNormalization()(l)
	
	# Pooling and dropout
	l = tfkl.MaxPooling2D()(l)
	l = tfkl.Dropout(drop_c)(l)


	# First dense
	l = tfkl.Flatten()(l)
	l = tfkl.Dense(d1)(l)
	if bn1 or bn2: l = tfkl.BatchNormalization()(l)
	l = tfkl.Activation(act)(l)
	l = tfkl.Dropout(drop_d)(l)

	# Dense Output
	l = tfkl.Dense(c)(l)
	output_layer = tfkl.Activation('softmax')(l)


	# Compiles model with optimizer
	model = tfk.Model(input_layer, output_layer)
	opt = tfk.optimizers.Adam(learning_rate=lr)
	# opt = tfk.optimizers.SGD(learning_rate=0.1, nesterov=True, momentum=0.0)
	model.compile(
		loss='sparse_categorical_crossentropy',
		optimizer=opt,
		metrics=['accuracy']
	)

	# Prints model summary
	model.summary()

	# Callbacks and checkpoint
	checkpoint = tfk.callbacks.ModelCheckpoint(
		filepath=model_path, 
		monitor='val_accuracy',
		verbose=1,
		save_best_only=True
	)
	callbacks = [checkpoint]

	# Trains model
	history = model.fit(
		x=x_train,
		y=y_train,
		batch_size=128,
		epochs=epochs,
		verbose=1,
		callbacks=callbacks,
		validation_split=0.10,
		shuffle=True
	)

	return model, history

def get_class_hist(preds, y_ar):
	hist = np.zeros(100)
	for p, y in zip(preds, y_ar):
		pred = p.argmax()
		actual = y[0]
		if pred == actual:
			hist[pred] += 1
	return hist

def main():
	# Gets epochs if passed as cmd line arg
	if len(sys.argv) > 1:
		epochs = int(sys.argv[1])
	else:
		epochs = 15

	# Load data
	x_train, y_train, x_test, y_test = load_keras_cifar100()

	# 15 Epoch model
	model_path_15 = 'cifar100_best_ckeckpt-15.h5'
	model_15, history_15 = load_model_15(model_path_15, epochs, x_train, y_train)
	model_15.load_weights(model_path_15)
	res_15 = model_15.evaluate(x_test, y_test, batch_size=128)

	# Gets predictions
	preds = model_15.predict(x_test)

	# Creates histogram for best and worst classes
	hist_m15 = get_class_hist(preds, y_test)
	best, worst = hist_m15.argmax(), hist_m15.argmin()
	print(f'\nbest, worst classes: {best}, {worst}')
	
	# Calculates accuracy to make sure it is same as keras eval
	right, total = hist_m15.sum(), y_test.shape[0]
	acc = right / total
	print(f'\nmy acc calc: {acc : 4.4f}')

	# Prints results
	print(f'\nRESULTS: test loss, test acc: {res_15[0] : 4.4f}, {res_15[1] : 4.4f}\n')

if __name__ == '__main__':
	main()