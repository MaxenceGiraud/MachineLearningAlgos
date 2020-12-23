from abc import abstractmethod
from ..base import BaseAny

class BaseUnsupervized(BaseAny):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self,*args,**kwargs):
        pass

    @abstractmethod
    def predict(self,*args,**kwargs):
        pass

    def fit_predict(self,X,*args,**kwargs):
        self.fit(X)
        return self.predict(X)