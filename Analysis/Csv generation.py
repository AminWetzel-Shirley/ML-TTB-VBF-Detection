import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle
import cv2

pickle_in = open(r"array_SNG_TTBar_external.pickle","rb")
x = pickle.load(pickle_in)

def reshape_function(length, width):
    A = []
    for i in range(0, len(x)):
        A.append(cv2.resize(x[i][:,:,0], (length,width)))
    A = np.array(A).reshape(-1, length, width, 1)
    A = np.array(A)
    A = np.array(A/np.amax(A))
    return A 

x = reshape_function(25,10)

with open('arraysTTB', 'w') as f:
    wr = csv.writer(f)
    
    for template in x:
        template = list(template.flatten())
        wr.writerow(template)
    
print("DONE!")