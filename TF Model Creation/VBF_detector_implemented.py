'''Creates a model identical to the one we found to be the best 
implementable VBF detector network. Also contains reshape functions
for changing the shapes of the histograms. The architecture already 
given was optimised for 25 by 25 image granularity.'''

import tensorflow as tf
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
from tensorflow import keras
import pickle 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from scipy import mean 


pickle_in = open("Your file path here","rb")
X = pickle.load(pickle_in)

pickle_in = open("Your file path here","rb")
y = pickle.load(pickle_in)

pickle_in = open("Your file path here","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("Your file path here","rb")
y_test = pickle.load(pickle_in)



def reshape_function_train(length, width):
    A = []
    for i in range(0, len(X)):
        A.append(cv2.resize(X[i][:,:,0], (length,width)))
    A = np.array(A).reshape(-1, length, width, 1)
    A = np.array(A)
    A = np.array(A/np.amax(A))
    return A 

def reshape_function_validate(length, width):
    A = []
    for i in range(0, len(X_test)):
        A.append(cv2.resize(X_test[i][:,:,0], (length, width)))
    A = np.array(A).reshape(-1, length, width, 1)
    A = np.array(A)
    A = np.array(A/np.amax(A))
    return A 


model = tf.keras.models.Sequential()  
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(9, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(80, activation = tf.nn.relu))
model.add(Dropout(0.2))
model.add(tf.keras.layers.Dense(30, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.compile(optimizer='Adam',  
                  loss='binary_crossentropy',  
                  metrics=['accuracy'], verbosity = 1) 
model.fit(reshape_function_train(25,25), y, epochs=6)
model.evaluate(reshape_function_validate(25, 25), y_test)













