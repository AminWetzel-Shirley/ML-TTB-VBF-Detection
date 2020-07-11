
'''Creates Pickle files for training/testing. '''

import numpy as np
import os
from tqdm import tqdm
import random
import pickle
import sys

VBFH, SNG, TTBAR = [], [], []
CATEGORIES = ["VBFH_numpy", "TTBAR_numpy","SNG_numpy"]


def list_switch_statement(cat):
    if cat == "VBFH_numpy":
        list_to_return = VBFH
    elif cat == "SNG_numpy":
        list_to_return = SNG
    elif cat == "TTBAR_numpy":
        list_to_return = TTBAR
    else:
        sys.stderr.write("file " + cat + " has no list to write to")
    return list_to_return

def create_training_data_lists(DATADIR = "Your file path here"):
    for category in CATEGORIES:  
        path = os.path.join(DATADIR,category)
        label_array = np.array([0])
        class_num = CATEGORIES.index(category) 
        if class_num == 1:
            pass
        else:
            label_array[class_num] = 1
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
TTBAR_t, TTBAR_ev = partition_lists(TTBAR)
VBFH_t, VBFH_ev = partition_lists(VBFH)
SNG_t, SNG_ev = partition_lists(SNG)

training_data = VBFH_t + SNG_t + TTBAR_t
external_validation_data = VBFH_ev + SNG_ev + TTBAR_ev 

random.shuffle(training_data)

x, y = [], []

for features, label in training_data:
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

random.shuffle(external_validation_data)

x_val, y_val = [], []

for features, label in external_validation_data:
    x_val.append(features)
    y_val.append(label)

x_val = np.array(x_val).reshape(-1, 120, 72, 1) #what is this doing?
x_val = np.array(x_val/np.amax(x_val)) #and this?
y_val = np.array(y_val)
pickle_out = open("Your file path here","wb")
pickle.dump(x_val, pickle_out)
pickle_out.close()

pickle_out = open("Your file path here","wb")
pickle.dump(y_val, pickle_out)
pickle_out.close()

print("FILES SAVED")
