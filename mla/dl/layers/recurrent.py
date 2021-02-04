import numpy as np
from .base_layer import BaseLayer
from ..activation import Linear


class Recurrent(BaseLayer):
    def __init__(self,units,activation=Linear(),return_all=False):
        self.units = units 
        self.activation = activation
        self.return_all = return_all
    
    def plug(self,inputlayer):
        self.input_shape = inputlayer.output_shape
        self.input_unit = inputlayer
        inputlayer.output_unit = self

        self.zin = 0
        self.zout = 0

        # Init Weights
        self.w = np.zeros(1)
        self.b = np.zeros(self.units)
        # Init Deriv
        self.dw = 0
        self.db = 0


    @property
    def nparams(self):
        return self.w.size + self.b.size

    def forward(self,X):
        pass
    
    def get_gradients(self):
        return self.dw,self.db

    def update_weights(self,weights_diff):
        uw,ub = weights_diff
        self.w += uw
        self.b += ub
    
    def backprop(self,delta):
        raise NotImplementedError
        # return delta