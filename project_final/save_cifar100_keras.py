import tensorflow.keras as tfk
import numpy as np
import os, sys

def main():
	# Args
	out_dir = 'cifar100'
	out_files = [
		'x_train.npy', 
		'y_train.npy', 
		'x_test.npy', 
		'y_test.npy'
	]
	out_paths = [os.path.join(out_dir, f) for f in out_files]

	# Sets up directories
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	# Pulls from keras
	(x_train, y_train), (x_test, y_test) = tfk.datasets.cifar100.load_data(label_mode='fine')
	cifar_list = [x_train, y_train, x_test, y_test]

	# Saves files
	for path, imgs in zip(out_paths, cifar_list):
		print(f'Saving {path}...')
		with open(path, 'wb') as f:
			np.save(f, imgs)
		print('Done\n')

if __name__ == '__main__':
	main()