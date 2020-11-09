import numpy as np

class Loss:
    def __init__(self):
        self.input_unit = None
        self.output_unit = None

        self.loss = 0
        self.loss_d = 0

    def plug(self,intputlayer):
        assert len(intputlayer.output_shape) == 1, "Input of Loss layer must be a vector"
        self.input_shape = intputlayer.output_shape[0]
        self.output_shape = self.input_shape

        self.input_unit = intputlayer
        intputlayer.output_unit = self

        self.zin = 0
    
    def forward(self,X,y):
        self.zin = self.input_unit.forward(X)
        self.loss = self.loss_function(self.zin,y)
        return self.loss

    def backprop(self,y):
        self.loss_d = self.deriv(self.loss,y)
        return self.loss

    def loss_function(self,y_pred,y_true):
        raise NotImplementedError("Function not supposed to call, class is only a base")
    
    def deriv(self,y_pred,y_true):
        raise NotImplementedError("Function not supposed to call, class is only a base")

class MSE(Loss):
    def loss_function(self,y_pred,y_true):
        return np.sum(np.square(y_true-y_pred)) / y_true.size
    
    def deriv(self,y_pred,y_true):
        return 2*np.sum(y_pred - y_true) / y_true.size

class MAE(Loss):
    def loss_function(self,y_true,y_pred):
        return np.sum(np.abs(y_pred-y_true)) / y_true.size

    def deriv(self,y_pred,y_true):
        return np.sum(np.sign(y_pred-y_true)) / y_true.size