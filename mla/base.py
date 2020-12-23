import numpy as np
from abc import abstractmethod
from .metrics import mean_squared_error,accuracy_score

class BaseAny:
    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self,X,y):
        raise NotImplementedError

    @abstractmethod
    def predict(self,X):
        raise NotImplementedError
    
    def score(self,X,y,metric):
        y_hat = self.predict(X)
        return metric(y,y_hat)      

class BaseRegressor(BaseAny):

    def score(self,X,y,metric=mean_squared_error):
        return super().score(X,y,metric)

class BaseClassifier(BaseAny):

    def score(self,X,y,metric=accuracy_score):
        return super().score(X,y,metric)