import numpy as np
from ..activation.activation import Linear

class Dense:
    def __init__(self,units,activation=Linear()):
        self.units = units
        self.activation = activation

        self.output_unit = None
    
    def plug(self,intputlayer):
        self.input_shape = intputlayer.output_shape
        self.output_shape = self.input_shape

        self.input_unit = intputlayer
        intputlayer.output_unit = self

        # Weights
        self.w = np.random.randn(self.input_shape)
        self.b = np.random.randn() # bias
        # Deriv
        self.w_d = 0
        self.b_d = 0
    
    def forward(self,idx):
        pass

    def backprop(self,idx):
        pass

    def update(self,idx):
        pass