import numpy as np, sys, os
from PIL import Image 
from ISR.models import RDN, RRDN
import time

def write_npys(img_list, cifar100_dir, i, tt, nt, min_width):
	print()
	fname = f'{tt}_{nt}-{min_width}_{i}.npy'
	fpath = os.path.join(cifar100_dir, fname)
	imgs = np.asarray(img_list, dtype=np.uint8)
	print(f'{fname} imgs shape: {imgs.shape}')
	with open(fpath, 'wb') as f:
		np.save(f, imgs)

def main():
	# Args
	min_width = 512
	# nt_list = ['noise-cancel', 'psnr-small', 'gans']
	nt_list = ['noise-cancel']
	img_out_dir = 'cifar100_enlarged_images'
	cifar100_dir = 'cifar100'
	cifar100_dir_out = 'cifar100_enlarged'
	cifar100_files = [
		'x_train.npy', 
		'x_test.npy'
	]
	cifar100_paths = [os.path.join(cifar100_dir, f) for f in cifar100_files]

	# Creates dirs needed
	if not os.path.exists(img_out_dir):
		os.makedirs(img_out_dir)
	if not os.path.exists(cifar100_dir_out):
		os.makedirs(cifar100_dir_out)

	# Opens all cifar100 stuff
	cifar100_data_list= [np.load(path) for path in cifar100_paths]
	cifar100_data_tt = ['x_train', 'x_test']

	# Enlarge images
	x_dict = {}
	for tt, imgs in zip(cifar100_data_tt, cifar100_data_list):
		x_dict.update({tt : {}})
		for nt in nt_list:
			x_dict[tt].update({nt : []})
			if nt == 'gans':
				rdn = RRDN(weights=nt)
			else:
				rdn = RDN(weights=nt)
			
			# Goes through all images
			time_start = time.time()
			print(f'\n\nStarting {tt} {nt}...')
			for i, img in enumerate(imgs):
				img_ss = img.copy()
				while(img_ss.shape[1] < min_width):
					# img_ss = rdn.predict(img_ss, by_patch_of_size=32)
					img_ss = rdn.predict(img_ss)
				x_dict[tt][nt].append(img_ss)
				if (i+1) % 100 == 0: print(f'{i+1} images complete in {time.time() - time_start} seconds.')
				if (i+1) % 1000 == 0:
					write_npys(x_dict[tt][nt], cifar100_dir_out, str((i+1)).zfill(5), tt, nt, min_width)
					x_dict[tt][nt].clear()
				# print(f'{i+1} images complete in {time.time() - time_start} seconds.')
			print(f'Done.')

				# img_new = Image.fromarray(img_ss)
				# new_x_train.append(img_new)
				# num = f'{i}'.zfill(5)
				# path = os.path.join(img_out_dir, f'img-{num}.png')
				# Image.fromarray(img).save(path)

	# Writes images
	# for i, img in enumerate(new_x_train):
	# 	num = f'{i}'.zfill(5)
	# 	path = os.path.join(img_out_dir, f'img_ss_{nt}-{num}.png')
	# 	img.save(path)

	# Writes dict as npy files
	# print()
	# for tt, nt_dict in x_dict.items():
	# 	for nt, img_list in nt_dict.items():
	# 		fname = f'{tt}_{nt}-{min_width}.npy'
	# 		fpath = os.path.join(cifar100_dir, fname)
	# 		imgs = np.asarray(img_list)
	# 		print(f'{fname} imgs shape: {imgs.shape}')
	# 		with open(fpath, 'wb') as f:
	# 			np.save(f, imgs)

if __name__ == '__main__':
	main()