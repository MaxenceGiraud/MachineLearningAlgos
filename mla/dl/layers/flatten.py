import numpy as np

class Flatten:
    def __init__(self):
        pass

    @property
    def nparams(self):
        return 0
    
    def plug(self,intputlayer):
        self.input_shape = intputlayer.output_shape[0]
        self.output_shape = [np.prod(self.input_shape)]

        self.input_unit = intputlayer
        intputlayer.output_unit = self

    def forward(self,X):
        return X.reshape((-1,))

    def backprop(self,X,delta):
        return delta.reshape(self.input_shape)
    
    def update(self,*args,**kwargs):
        pass
    