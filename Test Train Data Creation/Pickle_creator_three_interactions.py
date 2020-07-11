import numpy as np
import os
from tqdm import tqdm
import random
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from datetime import datetime

VBF, TTB, SNG = [], [], []
CATEGORIES = ["VBFH_numpy", "TTBar_numpy", "SingleNeutrino_numpy"]

def list_switch_statement(cat):
    if cat == "VBFH_numpy":
        list_to_return = VBF
    elif cat == "TTBar_numpy":
        list_to_return = TTB
    elif cat == "SingleNeutrino_numpy":
        list_to_return = SNG
    else:
        sys.stderr.write("file " + cat + " has no list to write to")
    return list_to_return

def create_training_data_lists(DATADIR = "Your file path here"):
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)
        label_array = np.array([0,0])
        class_num = CATEGORIES.index(category) #0 for VBF, 1 for TTBar, 2 for Neutrino (BACKGROUND)
        if class_num == 2:
            pass
        else:
            label_array[class_num] = 1 #CURRENTLY THIS WORKS OFF ASSUMING TTB AND VBF DO NOT OCCUR TOGETHER, CHANGE IF YOU NEED
        for img in tqdm(os.listdir(path)):  
            try:
                img_array = np.load(os.path.join(path,img)) 
                relevant_list = list_switch_statement(category)
                relevant_list.append([img_array, label_array])  
            except Exception:  
                sys.stderr.write("image " + img + " failed")
                
def partition_lists(lst, main = 3000, reserve = 1000):
    if reserve > main:
        sys.stderr.write("reserve cannot be greater than main")
    if reserve > len(lst):
        sys.stderr.write("reserve cannot be greater than total samples")
    print("WARNING: PARTIONING DATA BY " + str(len(lst) - reserve) + ":" + str(reserve))
    return lst[0:main], lst[(len(lst)-reserve):]
                
create_training_data_lists()
VBF_t, VBF_ev = partition_lists(VBF)
TTB_t, TTB_ev = partition_lists(TTB)
SNG_t, SNG_ev = partition_lists(SNG)

training_data = VBF_t + TTB_t + SNG_t  
external_validation_data = VBF_ev + TTB_ev + SNG_ev

random.shuffle(training_data)

x, y = [], []

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, 120, 72, 1) 
x = np.array(x/np.amax(x)) 
y = np.array(y)

pickle_out = open("Your file path here","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("Your file path here","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

random.shuffle(external_validation_data)

x, y = [], []

for features, label in external_validation_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, 120, 72, 1) #what is this doing?
x = np.array(x/np.amax(x)) #and this?
y = np.array(y)
pickle_out = open("Your file path here","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("Your file path here","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print("FILES SAVED")
