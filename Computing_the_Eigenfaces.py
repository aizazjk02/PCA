import Pre_processing_the_data as pp 
"""download the pre_processing_data.py file from the source code """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


#displaying the average face or mean face 
fig,ax=plt.subplots(1,1,figsize=(8,8))
ax.imshow(pp.pca.mean_.reshape((64,64)), cmap="gray")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Average Face')

#computing the eigenfaces 

"""
here I have limited the number of eigenfaces to dispplay to 50 
as the screen size was not big enough to display the total number of eigenfaces
you can change the number of rows and columns of plot to the variables row and cols 
if you want to display all the images 
and remove the limit condition from the loop
"""
number_of_eigenfaces=len(pp.pca.components_)
eigen_faces=pp.pca.components_.reshape((number_of_eigenfaces, pp.data.shape[1], pp.data.shape[2]))

cols=10
rows=int(number_of_eigenfaces/cols)
fig, axarr=plt.subplots(nrows=5, ncols=10, figsize=(cols,rows*2))
axarr=axarr.flatten()
for i in range(number_of_eigenfaces):
    if i < 50:  # limit condition
        axarr[i].imshow(eigen_faces[i],cmap="gray")
        axarr[i].set_xticks([])
        axarr[i].set_yticks([])
        axarr[i].set_title(format(i))
plt.suptitle("All Eigen Faces".format(10*"=", 10*"="))
# plt.show()

"""
 display the coefficient matrices from the scikit PCA function \
    and also printing this information 
    which is related to the covariance matrices
"""
X_train_pca = pp.pca.transform(pp.X_train)
print(pp.pca.singular_values_)
print(pp.pca.n_features_in_)
print(X_train_pca)
print(pp.pca.components_)