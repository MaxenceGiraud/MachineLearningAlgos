from ..base import BaseAny
from abc import abstractmethod
from ..metrics import accuracy_score

class BaseSemi(BaseAny):
    def __init__(self):
        self.fitted = False
    
    @abstractmethod
    def fit_predict(self,X,y):
        raise NotImplementedError
    
    def score(self,X,y,metric=accuracy_score):
        y_hat = self.fit_predict(X,y)
        return metric(y,y_hat)