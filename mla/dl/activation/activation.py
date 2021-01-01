import numpy as np
from abc import abstractmethod

class BaseActivation:
    @abstractmethod
    def f(self,X):
        pass

    def deriv(self,X):
        pass