import torch
import os
import math
import numpy as np
# from torch.utils.serialization import load_lua
from skimage import io, transform
from matplotlib import pyplot as pl
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import scipy.io as sio
import imutils
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
is1=55
scaler = transforms.Resize((is1, is1))
#normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()
# ww = torch.FloatTensor(21945,3,is1,is1)
countrn=12677-46
countst=1733-94
ww = torch.FloatTensor(countrn,1,is1,is1)





def parse_args():
	"""Parse input arguments."""
	desc = ('This script generates images using rotation translation contrast changes etc.')
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--verify', dest='do_verify',
						help='show and verify each images',
						action='store_true')
	args = parser.parse_args()
	return args
def resize_image(img, size=(56,56)):

    h, w = img.shape[:2]
    # print(img.shape)
    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)



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
	labcount=np.zeros(58)
	labcount1 = np.zeros(58)
	for key, val in img2class.items():
		ii = 0
		if int(val)>0:

			labcount[int(val)]=labcount[int(val)]+1
			labcount1[int(val)] = labcount[int(val)]
	# print(labcount)
	for key, val in img2class.items():
		if labcount1[int(val)]<1000 and int(val)>0:

			src_png = str(key) + '.png'
			limit=1000-labcount[int(val)]
			aug=math.floor(limit/labcount[int(val)])+1



			image = cv2.imread('data/ImageCLEFmed2009_train.02' + '/' + src_png, cv2.IMREAD_GRAYSCALE)
			labcount1[int(val)]=labcount1[int(val)]+aug
			ii=1

			for r in np.linspace(-10,10,91):
			 	if ii>aug:
			 		break
			 	rotated = imutils.rotate_bound(image, r)
			 	cv2.imwrite('newaug/'+key+'_'+str(ii)+'.png',rotated)
			 	ii = ii + 1
			alpha = 1.01
			for beta in range(25):
				if ii > aug:
					break
				new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
				cv2.imwrite('newaug/' + key + '_' + str(ii) + '.png', new_image)
				ii = ii + 1

			# alph=[1,1.1,1.2]
			alpha=1.02
			for beta in range(25):
				if ii>aug:
					break
				new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
				cv2.imwrite('newaug/' + key + '_' + str(ii) + '.png', new_image)
				ii = ii + 1

			alpha = 1.03
			for beta in range(25):
				if ii > aug:
					break
				new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
				cv2.imwrite('newaug/' + key + '_' + str(ii) + '.png', new_image)
				ii = ii + 1

			alpha = 1.04
			for beta in range(25):
				if ii > aug:
					break
				new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
				cv2.imwrite('newaug/' + key + '_' + str(ii) + '.png', new_image)
				ii = ii + 1

			alpha = 1.05
			for beta in range(25):
				if ii > aug:
					break
				new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
				cv2.imwrite('newaug/' + key + '_' + str(ii) + '.png', new_image)
				ii = ii + 1

			alpha = 1.06
			for beta in range(25):
				if ii > aug:
					break
				new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
				cv2.imwrite('newaug/' + key + '_' + str(ii) + '.png', new_image)
				ii = ii + 1

			alpha = 1.07
			for beta in range(25):
				if ii > aug:
					break
				new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
				cv2.imwrite('newaug/' + key + '_' + str(ii) + '.png', new_image)
				ii = ii + 1

			alpha = 1.08
			for beta in range(25):
				if ii > aug:
					break
				new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
				cv2.imwrite('newaug/' + key + '_' + str(ii) + '.png', new_image)
				ii = ii + 1

			alpha = 1.09
			for beta in range(25):
				if ii > aug:
					break
				new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
				cv2.imwrite('newaug/' + key + '_' + str(ii) + '.png', new_image)
				ii = ii + 1

			alpha = 1.1
			for beta in range(25):
				if ii > aug:
					break
				new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
				cv2.imwrite('newaug/' + key + '_' + str(ii) + '.png', new_image)
				ii = ii + 1
			# print(labcount1)

			

	







def data_aug():
	"""Generate multiple copies of each image in the training set. You can select how many images for each class you want. The current example uses 1000. You can change it to a bigger or smaller number. Rotations are taken from -10 to 10 degrees. alpha is 1.01 to 1.1 and beta is 0 to 25. You can modify these values as per your requirement.

    """
	# create a folder newaug to save the images

	convert_one_folder()



def main():
	"""main"""
	logging.basicConfig(level=logging.DEBUG)

	logging.info('Generating images...')
	data_aug()

	logging.info('All done.')


if __name__ == '__main__':
	main()
	sys.exit()
