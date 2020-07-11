'''Creates model with identical architecture to the one we found 
to be the best possible TTB detector (NOT implemented). 
Uses convolutional layers and dense layers using full resolution
2D Histograms. '''

import tensorflow as tf
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

pickle_in = open("Your file path here","rb")
X = pickle.load(pickle_in)

pickle_in = open("Your file path here","rb")
y = pickle.load(pickle_in)

pickle_in = open("Your file path here","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("Your file path here","rb")
y_test = pickle.load(pickle_in)

X = np.array(X/np.amax(X))
y = np.array(y)

X_test = np.array(X_test/np.amax(X_test))
y_test = np.array(y_test)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=X.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
                  optimizer= 'adam',
              metrics=['accuracy'])


model.fit(X, y, batch_size=32, epochs=3, validation_split = 0.2)
_, accuracy = model.evaluate(X_test, y_test)
model.evaluate(X_test, y_test)
model.summary()