import numpy as np
from ..base import BaseClassifier,BaseRegressor
from ..kernels import RBF

class BaseKernelRidge:
    def __init__(self,kernel=RBF(),lambd=1):
        self.kernel = kernel
        self.lambd = lambd

    def fit(self,X,y):
        self.alpha = (np.linalg.pinv(self.kernel(X,X) + self.lambd*np.eye(X.shape[0])) @ y).reshape(-1,1)
        self.X = X
         
    def predict(self,X):
        return np.sum(self.alpha * self.kernel(self.X,X),axis=0)


class KernelRidgeRegressor(BaseKernelRidge,BaseRegressor):
    ''' Kernel Ridge Regressor

    Parameters
    ---------
    kernel : callable (x,y) that returns shape (x.shape[0],y.shape[0]),
        Kernel used
    lambd : float,
        l2 regularization parameter
    '''

    pass

class KernelRidgeClassifier(BaseKernelRidge,BaseClassifier):
    ''' Kernel Ridge Classifier

    Parameters
    ---------
    kernel : callable (x,y) that returns shape (x.shape[0],y.shape[0]),
        Kernel used
    lambd : float,
        l2 regularization parameter
    '''

    def fit(self,X,y):
        self.labels = np.unique(y)
        assert len(self.labels) == 2, "Only 2 class can be given to this classifier"

        # Renamed labels as 1 and -1
        y_new = np.zeros(y.shape)
        y_new[y == self.labels[0]] = 1
        y_new[y == self.labels[1]] = -1

        super().fit(X,y_new)
    
    def predict(self,X): 
        pred = super().predict(X)
        y_hat  = np.where(pred >0,self.labels[0],self.labels[1])
        return y_hat