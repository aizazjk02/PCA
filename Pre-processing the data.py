import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

#random image visualization 
df = pd.read_csv("face_data.csv") # reading dataset
labels = df["target"] # getting labels
pixels = df.drop(["target"],axis=1) # getting pixels
pixels = pixels.sample(n=50) # generating 5 random people's images



def show_orignal_images(pixels):
     #Displaying Orignal Images
     fig, axes = plt.subplots(5, 10, figsize=(11, 7),subplot_kw={'xticks':[], 'yticks':[]})
     x = np.array(pixels)
     for i, ax in enumerate(axes.flat):
 	    ax.imshow(x[i].reshape(64, 64), cmap='gray')
     plt.show()

show_orignal_images(pixels)
