#Reading h5 files is really easy
from PIL import Image
import numpy as np
import h5py
import io

#Just read in the h5 file:
train_dataset = h5py.File('datasets/train_cars.h5', "r")
#And break it up into the features and labels:
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

#Also, assuming this is image data created with the accopanying h5_write.py file, the resulting variables should have the shape:
#np.shape(train_set_x_orig) = (#Images,ImageHeight,ImageWidth,Channels)
#np.shape(train_set_y_orig) = (#Images,)

#Now you're ready to use them in CNN/R-CNN programs! :)
