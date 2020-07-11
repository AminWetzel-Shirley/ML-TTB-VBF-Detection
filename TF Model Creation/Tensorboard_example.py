'''Example for using TensorBoard to maximise model accuracy. 
The code creates and trains new networks and evaluates their
accuracy for combinations of given hyperparameters. Saves the accuracies 
and architectures in a folder which can then be viewed using TensorBoard.
Also includes functions for reshaping the 2D histograms as required with a given
length and width.'''


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
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp


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

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([10, 20, 30, 40, 50, 60, 70, 80]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

def ANNE2types(hparams):
    model = tf.keras.models.Sequential()  
    model.add(tf.keras.layers.Flatten()) 
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(30))
    model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT]))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    
    model.compile(hparams[HP_OPTIMIZER],  
                  loss='binary_crossentropy',  
                  metrics=['accuracy'], verbosity = 2)  
    model.fit(reshape_function_train(25,25), y, epochs = 2)
    _ , accuracy = model.evaluate(reshape_function_validate(25,25), y_test)
    return accuracy 


def run_ANNEbinary(run_dir, hparams): 
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)
    accuracy = ANNE2types(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step = 1)
    
     
session_num = 0
for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run_ANNEbinary('logs/' + run_name, hparams)
      session_num += 1