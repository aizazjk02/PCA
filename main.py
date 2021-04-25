import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC

##Helper functions. Use when needed. 
def show_orignal_images(pixels):
	#Displaying Orignal Images
	fig, axes = plt.subplots(6, 10, figsize=(11, 7),
	                         subplot_kw={'xticks':[], 'yticks':[]})
	for i, ax in enumerate(axes.flat):
	    ax.imshow(np.array(pixels)[i].reshape(64, 64), cmap='gray')
	plt.show()




## Step 1: Read dataset and visualize it.
df = pd.read_csv("face_data.csv")
labels = df["target"]
pixels = df.drop(["target"],axis=1)
# show_orignal_images(pixels)

## Step 2: Split Dataset into training and testing
x_train,y_train,x_test,y_test = train_test_split(pixels,labels)

show_orignal_images(x_test)
## Step 3: Perform PCA.
pca = PCA(n_components=200).fit(x_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.show()


