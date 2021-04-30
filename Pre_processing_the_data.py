import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import mglearn


#loading the dataset 
data = np.load("olivetti_faces.npy")
target = np.load("olivetti_faces_target.npy")

#showing the information about the dataset
print("Displaying the information about the dataset:")

print("There are {} images in the dataset".format(len(data)))
print("There are {} unique targets in the dataset".format(len(np.unique(target))))
print("Size of each image is {}x{}".format(data.shape[1],data.shape[2]))

#Displaying all the targets
print("Unique target number:",np.unique(target))

#displaying the random images from the dataset
def show_40_distinct_people(images, unique_ids):
    #Creating 4X10 subplots in  18x9 figure size
    fig, axarr=plt.subplots(nrows=4, ncols=10, figsize=(18, 9))
    #For easy iteration flattened 4X10 subplots matrix to 40 array
    axarr=axarr.flatten()
    
    #iterating over user ids
    for unique_id in unique_ids:
        image_index=unique_id*10
        axarr[unique_id].imshow(images[image_index], cmap='gray')
        axarr[unique_id].set_xticks([])
        axarr[unique_id].set_yticks([])
        axarr[unique_id].set_title("face id:{}".format(unique_id))
    plt.suptitle("There are 40 distinct people in the dataset")
    plt.show()

# show_40_distinct_people(data, np.unique(target)) # Displaying the random images of the people

#converting the matrix data into the vector 
X=data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
print("X shape:",X.shape)

#split the data into train and the test
#the dataset will be split into 70:30 ratio 
# where 70 is train and the 30 is test

X_train, X_test, y_train, y_test=train_test_split(X, 
                target, test_size=0.3,  
                stratify=target, random_state=0)
print("X_train shape:",X_train.shape)
print("y_train shape:{}".format(y_train.shape))

#displaying the number of samples for each images 
# i.e the train data contains the 7 of 10 images of the person 
# and then 3 are given to the test
y_frame=pd.DataFrame()
y_frame['subject ids']=y_train
y_frame.groupby(['subject ids']).size().plot.bar(figsize=(15,8),
                title="Number of Samples for Each Classes")
# plt.show() # show the plot of training samples


#finding the optimum number of components 
pca=PCA()
pca.fit(X)

plt.figure(1, figsize=(12,8))

plt.plot(pca.explained_variance_, linewidth=2)
 
plt.xlabel('Components')
plt.ylabel('Explained Variaces')
# plt.show()

#computing the pca with 90 components 
n_components = 90
pca=PCA(n_components=n_components, whiten=True)
pca.fit(X_train)
