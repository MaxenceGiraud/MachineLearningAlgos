import numpy as np
from ..activation import Linear
from .base_layer import BaseLayer

class Dense(BaseLayer):
    def __init__(self,units,activation=Linear()):
        self.units = units
        self.activation = activation

        self.zin = 0
        self.zout = 0
        self.output_unit = None
    
    @property
    def nparams(self):
        return self.w.size + self.b.size
    
    def plug(self,inputlayer):
        assert len(inputlayer.output_shape) == 1, "Input of Dense layer must be a vector"
        self.input_shape = inputlayer.output_shape[0]
        self.output_shape = [self.units]

        self.input_unit = inputlayer
        inputlayer.output_unit = self

        # Weights
        self.w = np.random.randn(self.units*self.input_shape).reshape((self.input_shape,self.units))
        self.b = np.random.randn(self.units) # bias
        # Deriv
        self.dw = np.zeros(self.units)
        self.db = np.zeros(self.units)
        self.delta = np.zeros(self.units)
    
    def forward(self,X):
        self.zin = self.input_unit.forward(X) 
        self.zout = self.activation.f(self.zin @ self.w  + self.b) 
        return self.zout

    def backprop(self,delta):
        self.delta = delta @ self.w.T * self.activation.deriv(self.zin)
        self.dw =  delta.T @ self.zin 
        self.db = np.sum(delta,axis=0) # same as delta.T @ np.ones(...)
        return self.delta
    
    def get_gradients(self):
        return self.dw.T,self.db
    
    def update_weights(self,weights_diff):
        ''' Update weights given the update'''
        uw,ub = weights_diff
        self.w += uw
        self.b += ub 