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

class BaseRegressor(BaseAny):

    def score(self,X,y,metric=mean_squared_error,weights=1):
        y_hat = self.predict(X)
        return metric(y,y_hat,metric,weights=weights)

class BaseClassifier(BaseAny):

    def score(self,X,y,metric=accuracy_score):
        y_hat = self.predict(X)
        return metric(y,y_hat,metric)