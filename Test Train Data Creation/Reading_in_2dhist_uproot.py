import uproot 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os 

'''Takes in a root file and extracts the appropriate data, then uses uproot to view the 2D Histogram image.''''

BASE_PATH = "Your file path" 

for subdir, dirs, files in os.walk("Your file path"):
    i = 0
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith(".root"):
            f = uproot.open(filepath)
            a = f["caloGrid"].values
            plt.imshow(a)
            plt.show()
            break 