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
is1=220
scaler = transforms.Resize((is1, is1))
#normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()
# ww = torch.FloatTensor(21945,3,is1,is1)
countrn=12677-46
countst=1733-94
ts=57*500
ww = torch.FloatTensor(countst,3,is1,is1)





def parse_args():
	"""Parse input arguments."""
	desc = ('This script generates test images.')
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--verify', dest='do_verify',
						help='show and verify each images',
						action='store_true')
	args = parser.parse_args()
	return args
def resize_image(img, size=(220,220)):

    h, w = img.shape[:2]
    # print(size)
    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w


    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2
	# c=3

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, 3), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)



def convert_one_folder():
	
	f = open("code/test57ind_class.csv", "r")
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
	ii=0
	labcount = np.zeros(58)
	for key, val in img2class.items():
		if int(val)>0:# and labcount[int(val)]<2000:
			labels.append(int(val))
			labcount[int(val)] = labcount[int(val)] + 1
			src_png = str(key) + '.png'
			fileind.append(key)



			
			image = cv2.imread('data/ImageCLEFmed2009_test.03' + '/' + src_png)#, cv2.IMREAD_GRAYSCALE)
			squared_image = resize_image(image, size=(220, 220))
			# cv2.imwrite('testim.png', squared_image)
			# img = cv2.imread('testim.png',cv2.IMREAD_GRAYSCALE)

			# print(img)
			# make all pixels < threshold black
			# r,imgg=cv2.threshold(squared_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			# imgg = 1.0 * (img > threshold)
			# cv2.imwrite('testbin.png', imgg)

			# img = Image.open('data/ImageCLEFmed2009_test.03' + '/' + src_png)
			# t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
			imgg = squared_image#/255.0
			t_img = Variable((to_tensor(imgg)).unsqueeze(0))
			# t_img = Variable(normalize(to_tensor(imgg)).unsqueeze(0))

			# t_img.data = t_img.data * 255
			ww[ii] = t_img.data
			ii=ii+1

	
	ll = torch.LongTensor(labels)

	
	torch.save(ww[0:ii,:,:,:], 'tstset240.pt')
	torch.save(ll, 'tsttar240.pt')
	# np.save('filename240.npy', fileind)








def test_gen():
	

	convert_one_folder()



def main():
	"""main"""
	logging.basicConfig(level=logging.DEBUG)

	logging.info('Copying png files and converting annotations...')
	test_gen()

	logging.info('All done.')


if __name__ == '__main__':
	main()
	sys.exit()
