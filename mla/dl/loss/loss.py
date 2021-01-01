import numpy as np
from ..layers.base_layer import BaseLayer
from abc import abstractmethod

class Loss(BaseLayer):
    def __init__(self):
        self.input_unit = None
        self.output_unit = None

        self.loss = 0
        self.loss_d = 0

    def plug(self,intputlayer):
        self.input_shape = intputlayer.output_shape
        self.output_shape = self.input_shape

        self.input_unit = intputlayer
        intputlayer.output_unit = self
    
    def forward(self,X,y,weights=1):
        self.zin = self.input_unit.forward(X)
        self.loss = self.loss_function(self.zin,y,weights=weights)
        return self.loss

    def backprop(self,y,weights=1):
        self.loss_d = self.deriv(self.zin,y.reshape(self.zin.shape),weights=weights)
        return self.loss_d

    @abstractmethod
    def loss_function(self,y_pred,y_true,weights,*args):
        raise NotImplementedError("Function not supposed to call, class is only a base")

    @abstractmethod
    def deriv(self,y_pred,y_true,weights,*args):
        raise NotImplementedError("Function not supposed to call, class is only a base")