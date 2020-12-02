import numpy as np
from ..activation.activation import Linear

class Dense:
    def __init__(self,units,activation=Linear()):
        self.units = units
        self.activation = activation

        self.zin = 0
        self.zout = 0
        self.output_unit = None
    
    @property
    def nparams(self):
        return self.w.size + self.b.size
    
    def plug(self,intputlayer):
        assert len(intputlayer.output_shape) == 1, "Input of Dense layer must be a vector"
        self.input_shape = intputlayer.output_shape[0]
        self.output_shape = [self.units]

        self.input_unit = intputlayer
        intputlayer.output_unit = self

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
        self.db = np.sum(delta,axis=0)
        return self.delta

    def update(self,lr,noise_std=0):
        self.w -= (self.dw.T+np.random.normal(0,noise_std,size=self.dw.size)) * lr  
        self.b -= (self.db+np.random.normal(0,noise_std)) * lr  