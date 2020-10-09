import torch
import os
import numpy as np
# from torch.utils.serialization import load_lua
from skimage import io, transform
from matplotlib import pyplot as pl
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import scipy.io as sio
from sklearn.model_selection import train_test_split



import sys
import math
import logging
import argparse

from shutil import rmtree, copyfile


from scipy.io import loadmat
import cv2
import pathlib



VISUALIZE = False  # visualize each image (for debugging)


#train and test image directories






def parse_args():
	"""Parse input arguments."""
	desc = ('This script generates training file list.')
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--verify', dest='do_verify',
						help='show and verify each images',
						action='store_true')
	args = parser.parse_args()
	return args




def convert_one_folder():
	f=open("code/train57ind_class.csv","r")
	
	img2class = {}  # ind vs code
	for l in f:
		# print(l)
		tok = l.split(',')
		# print(tok[0],tok[1])
		img2class[tok[0]] = tok[1]
	# print(tok[0], class2code[tok[1]])

	f.close()
	labels=[]
	threshold = 150
	fileind=[]
	filename=[]
	ii=0
	labcount = np.zeros(58)
	for key, val in img2class.items():
		if int(val)>0:# and labcount[int(val)]<2000:
			labels.append(int(val))
			labcount[int(val)] = labcount[int(val)] + 1
			src_png = str(key) + '.png'
			# fileind.append(key)
			filename.append('data/ImageCLEFmed2009_train.02/' + src_png)


			# image = cv2.imread('data/ImageCLEFmed2009_train.02' + '/' + src_png)#, cv2.IMREAD_GRAYSCALE)
			# print(image.shape)
			# image = cv2.imread('data/ImageCLEFmed2009_test.03' + '/' + src_png)#, cv2.IMREAD_GRAYSCALE)

			# cv2.imwrite('testim.png', squared_image)
			# img = cv2.imread('testim.png',cv2.IMREAD_GRAYSCALE)

			# print(img)
			# make all pixels < threshold black
			# r,imgg=cv2.threshold(squared_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			# imgg = 1.0 * (img > threshold)
			# cv2.imwrite('testbin.png', imgg)

			# img = Image.open('data/ImageCLEFmed2009_test.03' + '/' + src_png)
			# t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

			ii=ii+1

	frames = [f for f in os.listdir('newaug') if f.endswith('png')]
	for i, frame1 in enumerate(frames):
		frame = frame1.split('_')[0]
		val=img2class[frame]

		if int(val) > 0 and labcount[int(val)] < 250:
			labels.append(int(val))
			labcount[int(val)] = labcount[int(val)] + 1
			src_png = frame1
			filename.append('newaug/' + src_png)
			# image = cv2.imread('newaug/' + src_png)#, cv2.IMREAD_GRAYSCALE)
			# image = cv2.imread('data/ImageCLEFmed2009_test.03' + '/' + src_png, cv2.IMREAD_GRAYSCALE)
			# squared_image = resize_image(image, size=(240, 240))
			# cv2.imwrite('testim.png', squared_image)
			# img = Image.open('testim.png')
			# r,imgg=cv2.threshold(squared_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			# imgg=squared_image

			# img = Image.open('data/ImageCLEFmed2009_test.03' + '/' + src_png)
			# t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
			# t_img = Variable((to_tensor(imgg)).unsqueeze(0))
	#
	# 		# t_img.data = t_img.data * 255
	# 		ww[ii] = t_img.data
			ii = ii + 1

			
	
	np.save('filenamesaug240.npy', filename)
	np.save('filelabsaug240.npy', labels)








def train_names():
	

	convert_one_folder()



def main():
	"""main"""
	logging.basicConfig(level=logging.DEBUG)

	logging.info('Generating training file list...')
	train_names()

	logging.info('All done.')


if __name__ == '__main__':
	main()
	sys.exit()
