# Script used during dev of yolt assignemnt
"""
Build a python program that trains a classifier on the iris dataset, e.g. using sklearn.datasets ,
and displays the model performance.
"""

#%% Import modules:
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA


#%% User Defined Variables
train_fraction = .7

#%% First impressions:
# Loading data:
data = sklearn.datasets.load_iris() 
print(data)
"""
3 output classes, 4 input dimensions.
Discription of dataset mentions the following:

One class is linearly separable from the other 2; the 
latter are NOT linearly separable from each other
"""

# I prefer to maintain input X, output y, so:
X = data['data']
y = data['target']

# Some quantifiables of data:
print(X.shape, y.shape)
print(len(np.unique(y)))
print(np.mean(X,axis=0), np.std(X,axis=0))
"""
Low amount of samples and low amount of dimensions.
Next steps:
Data could be improved a little bit by normalization.
Classifiers will have to be relatively simple due to low amount of data.
Given that it was mentioned previously that data is NOT linearly separable,
my first choice would be KNN, before kernel based SVM
"""

#%% Preprocessing data:
xmean   = np.mean(X,axis=0)
xstd    = np.std(X,axis=0)

X = (X-xmean)/xstd

# Test/train split:
np.random.seed(2021) # For reproducibility
ind = np.random.rand(X.shape[0])
ind = ind < np.quantile(ind,q=train_fraction)

# Train data:
X_train = X[ind,:]
y_train = y[ind]

# Test data:
X_test = X[np.logical_not(ind),:]
y_test = y[np.logical_not(ind)]

#%% Creating the first classifier:
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X, y)
y_pred = knn.predict(X_test)
# Assessing results:
sklearn.metrics.confusion_matrix(y_test, y_pred)

"""
And just like that, it seems the problem is solved. Not a single classification error.
"""

