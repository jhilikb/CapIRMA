from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import torchvision.transforms as transforms
import cv2
from torch.autograd import Variable

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

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
class DLibdata:
    
    
    def __init__(self,train=True):
        
        self.train = train  # training set or test set
        # self.indx = ind
        # self.filen= fn


        

        if self.train:


            self.images = np.load('filenamesaug240.npy')
            labels = np.load('filelabsaug240.npy')
            self.train_labels=torch.LongTensor(labels)
            self.train_labels = self.train_labels - 1
            
            
                
        else:
        	

            self.test_data = torch.load('tstset240.pt')
            self.test_data = self.test_data.float()
            self.test_labels = torch.load('tsttar240.pt')
            self.test_labels=self.test_labels-1

            
                

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target,index) where target is index of the target class.
        """
        if self.train:
            # img, target = self.train_data[index], self.train_labels[index]
            img_name = self.images[index]
            # print(img_name)
            img = cv2.imread(img_name)
            img = resize_image(img, size=(220, 220))
            #img=img/255.0

            img = Variable((to_tensor(img))
            # print(img)

            target = self.train_labels[index]
        else:
            img, target,index = self.test_data[index], self.test_labels[index],index

        

        return img, target,index

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_data)

 
