import numpy as np
from abc import abstractmethod
from .metrics import mean_squared_error,accuracy_score
import inspect

class BaseAny:
    def __init__(self):
        pass
        
    def __repr__(self):
        signa = inspect.signature(self.__init__)
        param_str = ""
        for param in signa.parameters.keys():
            param_str += str(param) +"="+ str(getattr(self,param)) +","
        param_str = param_str[:-1]
        return str(self.__class__.__name__)+ "("+param_str+")"
    
    def __str__(self):
        return self.__repr__()

class BaseML(BaseAny):
    @abstractmethod
    def fit(self,X,y):
        raise NotImplementedError

    @abstractmethod
    def predict(self,X):
        raise NotImplementedError

class BaseRegressor(BaseML):

    def score(self,X,y,metric=mean_squared_error,weights=1):
        y_hat = self.predict(X)
        return metric(y,y_hat,metric,weights=weights)

class BaseClassifier(BaseML):

    def score(self,X,y,metric=accuracy_score):
        y_hat = self.predict(X)
        return metric(y,y_hat,metric)