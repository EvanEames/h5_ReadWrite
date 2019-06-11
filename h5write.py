#READING IN IMAGES AND OUTPUTTING AN h5 FILE
#Note: It's not necessary that the files are the same size, however they should all be the same file type
#Note: Using height=width=224 and 8000 files the resulting h5 is roughly 1Gb in size.

import cv2
import datetime as dt
import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import numpy as np
import os
import pandas as pd
import sys
from glob import glob

#Specify here the directory in which the training data is found
SOURCE_IMAGES = './datasets/train_cars'
#And also the file type
images = sorted(glob(os.path.join(SOURCE_IMAGES, "*.jpg")))

#Specify the files containing the labels
#Note: Here I use a .txt file, however it can be any file type. The point is that, regardless of how you do it, at the end you should save them all into a 1D np array named 'labels'
labelsDF = pd.read_csv('devkit/train_perfect_preds.txt',header=None)
labels = np.zeros(0,)
for i in labelsDF[0]:
	labels = np.append(labels,i)

#How big do you want the images (they will all have their Height and Width adjusted to these values)
NUM_IMAGES = len(images)
HEIGHT = 224
WIDTH = 224
CHANNELS = 3
SHAPE = (HEIGHT, WIDTH, CHANNELS)
image = np.zeros((NUM_IMAGES, HEIGHT, WIDTH, CHANNELS))

#Verify that the number of labels and images are equal
if len(labels) != NUM_IMAGES:
	sys.exit("ERROR: The # of labels must equal the # of images!")

#Now we will write the h5 file:
with h5py.File('./datasets/train_cars.h5', 'w') as hf:
	#First, combine all the 
	for i,img in enumerate(images):
		print("Now reading file "+str(i+1)+"/"+str(NUM_IMAGES))
		image_tmp = cv2.imread(img)
		image[i][:][:][:] = cv2.resize(image_tmp, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC).astype(int)
	print("Now creating the h5 file, this may take some time...")
	#Compress and write the images
	Xset = hf.create_dataset(
		name='train_set_x',
		data=image,
		shape=(NUM_IMAGES,HEIGHT, WIDTH, CHANNELS),
		maxshape=(NUM_IMAGES,HEIGHT, WIDTH, CHANNELS),
		dtype = np.uint8,
		compression="gzip",
		compression_opts=9)
	#Compress and write the labels
	yset = hf.create_dataset(
		name='train_set_y',
		data = (labels),
		shape=(NUM_IMAGES,),
		maxshape=(NUM_IMAGES,),
		dtype = np.uint8,
		compression="gzip",
		compression_opts=9)
