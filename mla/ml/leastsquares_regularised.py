from .leastsquares import BaseLeastSquares
from ..base import BaseClassifier,BaseRegressor
import numpy as np

class BaseRegularizedLeastSquares(BaseLeastSquares):
    def __init__(self,lambd=1,degree=1):
        self.lambd  = lambd
        self.beta = 0
        self.degree = degree

class BaseRidge(BaseRegularizedLeastSquares):

    def fit(self,X,y):
        X = self.X_to_poly(X)
        self.beta = np.linalg.inv(self.lambd*np.identity(X.shape[1]) + X.T @ X) @ X.T @ y

class RidgeRegressor(BaseRidge,BaseRegressor):
    ''' Ridge Regressor
    Parameters
    ----------
    lambd : float,
            regularization term
    degree : int >=1
             degrees of features to consider,ex for 2 features x1,x2 with degree =2,
             the input features are going to be x1,x2,x1^2,x2^2
    '''
    pass

class RidgeClassifier(BaseRidge,BaseClassifier): 
    ''' Ridge Classifier
    Parameters
    ----------
    lambd : float,
            regularization term
    degree : int >=1
             degrees of features to consider,ex for 2 features x1,x2 with degree =2,
             the input features are going to be x1,x2,x1^2,x2^2
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

class BaseLasso(BaseRegularizedLeastSquares):

    def fit(self,X,y):
        X = self.X_to_poly(X)

        self.beta = np.zeros(X.shape[1])
        # Forward stepwise regression 
        

        # Least angle regression


        
class LassoRegressor(BaseLasso):
    '''Lasso linear regressor, fitted using Least angle regression
    
    Parameters
    ----------
    lambd : float,
            regularization term
    degree : int >=1
             degrees of features to consider,ex for 2 features x1,x2 with degree =2,
             the input features are going to be x1,x2,x1^2,x2^2
    '''
    def fit(self,X,y):
        X = self.X_to_poly(X)
        self.beta  = np.zeros(X.shape[1])
        
        fitting = []
        for _ in range(X.shape[1]):
            pass


class LassoClassifier(BaseLasso):
    '''Lasso linear Classifier, fitted using    
    
    Parameters
    ----------
    lambd : float,
            regularization term
    degree : int >=1
             degrees of features to consider,ex for 2 features x1,x2 with degree =2,
             the input features are going to be x1,x2,x1^2,x2^2
    '''
    def fit(self,X,y):
        self.labels = np.unique(y)
        assert len(self.labels) == 2, "Only 2 class can be given to this classifier"

        # Renamed labels as 1 and -1
        y[y == self.labels[0]] = 1
        y[y == self.labels[1]] = -1
        
        super().fit(X,y)
    
    def predict(self,X): 
        pred = super().predict(X)
        y_hat  = np.where(pred >0,self.labels[0],self.labels[1])
        return y_hat    
