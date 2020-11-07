import numpy as np

class Loss:
    def __init__(self):
        self.input_unit = None
        self.output_unit = None

    def plug(self,intputlayer):
        assert len(intputlayer.output_shape) == 1, "Input of Loss layer must be a vector"
        self.input_shape = intputlayer.output_shape[0]
        self.output_shape = self.input_shape

        self.input_unit = intputlayer
        intputlayer.output_unit = self
    
    def forward(self,X,y):
        y_hat = self.input_unit.forward(i)
        return self.loss_function(y,y_hat)

    def backprop(self,X,y):
        pass

    def loss_function(self,y_true,y_pred):
        return 0

class MSE(Loss):
    pass

class RMSE(MSE):
    pass

class MAE(Loss):
    pass