import numpy as np
from scipy.spatial.distance import cdist


class KNN:
    '''KNN Classifier'''
    def __init__(self,k):
        self.k = k

    def fit(self,X,Y):
        self.X = X
        self.y = Y

    def predict(self,X_test):
        all_dist = cdist(X_test, self.X) ## Computing all distances between points in test and training set
        nearest_neighbors = np.argsort(all_dist,axis=1)[:,:self.k] ## Sorting and keeping index of K(+1) nearest neighbours for each test point

        target_nearest = np.array(self.y[nearest_neighbors],dtype=int).T ## Corresponding the nearest point to their classification
        
        m = target_nearest.shape[1]  
        n = target_nearest.max()+1
        tn1 = target_nearest + (n*np.arange(m))
        out = np.bincount(tn1.ravel(),minlength=n*m).reshape(m,-1)  ## Choosing the most class that is the closest to the points

        y_hat = np.argmax(out,axis=1)
        return y_hat    

    def score(self,X_test,y_test):
        '''Compute Accuracy'''
        y_hat = self.predict(X_test)
        errors  = np.count_nonzero(y_hat-y_test)

        acc = 1- (errors / len(y_test))
        return acc



def knn_predict(K,Xtest,Xtrain,ttrain):
    '''
    implementation of KNN classifier
    @param :
        K : Number of neighbour
        Xtest : features of test set
        Xtrain : features of training set
        ttrain : target of training set
    '''
    all_dist = cdist(Xtest, Xtrain) ## Computing all distances between points in test and training set
    nearest_neighbors = np.argsort(all_dist,axis=1)[:,:K] ## Sorting and keeping index of K(+1) nearest neighbours for each test point

    target_nearest = np.array(ttrain[nearest_neighbors],dtype=int).T ## Corresponding the nearest point to their classification

    m = target_nearest.shape[1]    
    n = target_nearest.max()+1
    tn1 = target_nearest + (n*np.arange(m))
    out = np.bincount(tn1.ravel(),minlength=n*m).reshape(m,-1)  ## Choosing the most class that is the closest to the points

    that = np.argmax(out,axis=1)
    return that     