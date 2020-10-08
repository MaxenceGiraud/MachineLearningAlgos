import numpy as np
from scipy.spatial.distance import cdist


class KNN:
    '''KNN Classifier'''
    def __init__(self,k,metric='minkowski'):
        self.k = k
        self.metric = metric

    def fit(self,X,Y):
        self.X = X
        self.y = Y

    def predict(self,X_test):
        all_dist = cdist(X_test, self.X,metric=self.metric) ## Computing all distances between points in test and training set
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

class KNN_Regressor:
    '''KNN Regressor with uniform weight'''
    def __init__(self,k,metric='minkowski',weight='uniform'):
        assert weight in ['uniform','distance'], "weight must either be uniform or distance"
        self.k = k
        self.metric = metric
        self.weight = weight

    def fit(self,X,Y):
        self.X = X
        self.y = Y

    def predict(self,X_test):
        all_dist = cdist(X_test, self.X,metric=self.metric) ## Computing all distances between points in test and training set
        nearest_neighbors = np.argsort(all_dist,axis=1)[:,:self.k] ## Sorting and keeping index of K nearest neighbours for each test point

       
        if self.weight == 'uniform' :
            target_nearest = np.array(self.y[nearest_neighbors]).T ## Corresponding the nearest point to their target value
            y_hat = np.mean(target_nearest,axis=0)
        elif self.weight == 'distance':
            dist_nn = all_dist.T[nearest_neighbors][:,:,0]
            dist_nn = (dist_nn.T / np.sum(dist_nn,axis=1)).T # Normalize the distances
            y_hat = np.sum(self.y[nearest_neighbors] * dist_nn,axis=1)

        return y_hat    

    def score(self,X,y):
        '''Compute MSE for the prediction  of the model with X/y'''
        y_hat = self.predict(X)
        mse = np.sum((y - y_hat)**2) / len(y)
        return mse