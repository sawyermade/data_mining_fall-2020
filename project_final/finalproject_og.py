# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow.keras as tfk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Conv2D, MaxPool2D, MaxPooling2D, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras import layers
import sys

'''
Some code modified from that provided by Daniel Sawyer.  This implementation
is done with functions for a different look.  You do not have to use it.
You will work with cifar100 as set up here (in terms of train, validation 
and test).  This is color images of size 32x32 of 100 classes. Hence, 3 
chanels R, G, B.    I took out 10% for validation.
You can change this around, but must be very clear on what was done and why.
You must improve on 44% accuracy (which is a fairly low bar).  You need to 
provide a best class accuracy and worst class accuracy. To improve, more epochs
can help, but that cannot be the only change you make.  You should show  better
performance at 15 epochs or argue why it is not possible. 

I also want you to use a snapshot ensemble of at least 5 snapshots.  One
way to choose the best class is to sum the per class outputs and take the
maximum.  Another is to vote for the class and break ties in some way.
Indicate if results are better or worse or the same. (This is 5
extra credit points of the grade).

You must clearly explain what you tried and why and what seemed to work 
and what did not.  That will be the major part of your grade.  Higher 
accuracy will also improve your grade. If you use an outside source, it 
must be disclosed and that source may be credited with part of the grade.
 The best accuracy in class will add
4 points to their overall average grade, second best 3 points and 3rd best 2
points and 4th best 1 point.

To get predictions:
predictions=model.predict(ds_test)
Prints the first test predition and you will see 100 predictions
print(predictions[0])

'''

def loadmycifar100():
  # cifar100 has 2 sets of labels.  The default is "label" giving you 100
  # predictions for the classes
  (ds_train, dsvalid, ds_test), ds_info = tfds.load(
    'cifar100',
# First 25% and last 25% from training, then validation data is 5%
# from 25% of train data to 30% and test is the usual 10K
    split=['train[:45%]+train[-45%:]','train[45%:55%]', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
#tf.keras.datasets.cifar100.load_data(label_mode="fine")

	
#  ds_train = ds_train.map(
#    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_train = ds_train.cache()
  ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
  ds_train = ds_train.batch(128)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

#  ds_test = ds_test.map(
#    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_test = ds_test.batch(128)
  ds_test = ds_test.cache()
  ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


#  dsvalid = dsvalid.map(
#    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dsvalid = dsvalid.batch(64)
  dsvalid = dsvalid.cache()
  dsvalid = dsvalid.prefetch(tf.data.experimental.AUTOTUNE)
  return ds_train, dsvalid, ds_test

def main():
    # Checks if runs arg was passed.  If you want to auto run multiple times
    '''
	if len(sys.argv) > 1:
		runs = int(sys.argv[1])
	else:
		runs = 3
    '''
# Loads data
    (ds_train, dsvalid, ds_test) = loadmycifar100()
    epochs = 15
# Callback for saving best epoch checkpoint weights
    model_path = 'cifar100_best_ckeckpt.h5'
    checkpoint = tfk.callbacks.ModelCheckpoint(
			filepath=model_path, 
			monitor='val_accuracy', 
			verbose=1, 
			save_best_only=True
		)
    callbacks = [checkpoint]
    # Input shape and layer.  This is rgb
    input_shape = (32, 32, 3)
    input_layer = tfk.layers.Input(shape=input_shape)
    
    # First convolution, batch norm
    ly = tfk.layers.Conv2D(32, 3)(input_layer)
    ly = tfk.layers.BatchNormalization()(ly)
    ly = tfk.layers.Activation('relu')(ly)
    
    ly = tfk.layers.MaxPooling2D()(ly)
    
    # Second convolution, batch norm 
    ly = tfk.layers.Conv2D(64, 3)(ly)
    ly = tfk.layers.BatchNormalization()(ly)
    ly = tfk.layers.Activation('relu')(ly)

    # Third convolutional layer
    
    ly = tfk.layers.Conv2D(64, 3)(ly)
    ly = tfk.layers.BatchNormalization()(ly)
    ly = tfk.layers.Activation('relu')(ly)
    
    # Max pooling layer and flattens for dense layers
    ly = tfk.layers.MaxPooling2D()(ly)
    ly = tfk.layers.Flatten()(ly)
    
    # First dense layer with batch norm & dropout
    ly = tfk.layers.Dense(512)(ly)
    ly = tfk.layers.BatchNormalization()(ly)
    ly = tfk.layers.Activation('relu')(ly)
    ly = tfk.layers.Dropout(0.5)(ly)
    
    # Output dense layer, 100 classes
    ly = tfk.layers.Dense(100)(ly)
    output_layer = tfk.layers.Activation('softmax')(ly)
    
    # Compiles model with adam optimizer
    model = tfk.Model(input_layer, output_layer)
    opt = tfk.optimizers.Adam(learning_rate=0.001)
    model.compile(
	  	loss='sparse_categorical_crossentropy',
			optimizer=opt,
			metrics=['accuracy']
      )
    
    # Prints model summary
    model.summary()
    
    # Trains model with increased epochs and saves best
    model.fit(
			ds_train,
			epochs=epochs,
			validation_data=dsvalid,
			callbacks=callbacks
      )
    
    # Load best model weights from checkpoint and save results
    model.load_weights(model_path)
    res = model.evaluate(ds_test, batch_size=128)
    # Prints results
    print("Results ")
    print(res)

# When called run main
if __name__ == '__main__':
	main()
