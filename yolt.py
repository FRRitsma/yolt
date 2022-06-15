# # Try imports:
# try:
#     import numpy as np
#     from sklearn.neighbors import KNeighborsClassifier
#     import numpy as np
#     from sklearn import datasets
#     from sklearn.decomposition import PCA
# except:


#%% Import modules:
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA

#%%
class Classifier():
    def __init__(self, train_fraction:float = .7, random:bool = True):
        data = sklearn.datasets.load_iris() 
        X = data['data']
        y = data['target']
        self.xmean   = np.mean(X,axis=0)
        self.xstd    = np.std(X,axis=0)

        # Normalize
        X = self.norm(X)

        # Test/train split:
        if not random:
            np.random.seed(2021) # For reproducibility
        ind = np.random.rand(X.shape[0])
        ind = ind < np.quantile(ind,q=train_fraction)

        # Train data:
        X_train = X[ind,:]
        y_train = y[ind]

        # Test data:
        X_test = X[np.logical_not(ind),:]
        y_test = y[np.logical_not(ind)]

        # Fit:
        self.knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        self.knn.fit(X, y)

        # Assessing results:
        y_pred = self.knn.predict(X_test)
        self.ErrorMat = sklearn.metrics.confusion_matrix(y_test, y_pred)


    def norm(self, input):
        return (input - self.xmean)/self.xstd

    def Error(self):
        s = self.ErrorMat.shape
        return float(np.sum(self.ErrorMat[np.eye(s[0],s[1])==0])/np.sum(self.ErrorMat).ravel())

    def Predict(self, X):
        x = (X-self.xmean)/self.xstd
        return self.knn.predict(x)

