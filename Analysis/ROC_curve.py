'''Example code for producing ROC plots for evaluating model performance.
Also includes reshape function as before. Takes in a pre-made model and compares 
model output to target output to find false positive rate and true positive rates 
for plotting.'''


import tensorflow as tf
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy 
from scipy import mean 
import statistics as s
from sklearn.metrics import roc_curve, auc

length = 25 #dimensions of images model was trained with 
width = 25

def reshape_function_validate(length, width):
    pickle_in = open("Your file path here","rb")
    X_test = pickle.load(pickle_in)
    X_test = np.array(X_test)
    
    A = []
    for i in range(0, (len(X_test))):
        A.append(cv2.resize(X_test[i][:,:,0], (length, width)))  #resize images as necessary 
    A = np.array(A).reshape(-1, length, width, 1)
    A = np.array(A)
    A = np.array(A/np.amax(A))
    return A 

pickle_in = open("Your file path here","rb") #load in target values 
y_test = pickle.load(pickle_in)

def model_predictionTTB(path_to_model, length, width):
    model = tf.keras.models.load_model(path_to_model)
    predictions = model.predict(reshape_function_validate(length, width))
    return predictions #make model predictions using previously saved model 

fpr, tpr, _ =  roc_curve(y_test, model_predictionTTB("Your file path to model here", length, width)) #calculates false positive rate and true positive rate
plt.plot(fpr, tpr)  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() 

print('The AUC value is: ', auc(fpr,tpr)) #finds area under curve values for fpr and tpr 
    


