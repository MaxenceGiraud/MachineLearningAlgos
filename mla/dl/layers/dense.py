import numpy as np
from ..activation.activation import Linear

class Dense:
    def __init__(self,units,activation=Linear()):
        self.units = units
        self.activation = activation

        self.zin = 0
        self.zout = 0
        self.output_unit = None
    
    def plug(self,intputlayer):
        assert len(intputlayer.output_shape) == 1, "Input of Dense layer must be a vector"
        self.input_shape = intputlayer.output_shape[0]
        self.output_shape = self.input_shape

        self.input_unit = intputlayer
        intputlayer.output_unit = self

        # Weights
        self.w = np.random.randn((self.units,self.input_shape))
        self.b = np.random.randn(self.units) # bias
        # Deriv
        self.w_d = np.zeros(self.units)
        self.b_d = np.zeros(self.units)
    
    def forward(self,X):
        self.zin = self.input_unit.forward(X) 
        self.zout = self.activation.f(self.w @ self.zin + self.b) 
        return self.zout

    def backprop(self,X,delta):
        # TODO
        # self.w_d = self.activation.deriv(self.zout)
        # self.b_d = self.activation.deriv(self.zout)
        pass

    def update(self,lr):
        self.w -= self.w @ self.w_d * lr  
        self.b -= self.b @ self.b_d * lr  