import numpy as np

class Loss:
    def __init__(self):
        self.input_unit = None
        self.output_unit = None

        self.loss = 0

    def plug(self,intputlayer):
        assert len(intputlayer.output_shape) == 1, "Input of Loss layer must be a vector"
        self.input_shape = intputlayer.output_shape[0]
        self.output_shape = self.input_shape

        self.input_unit = intputlayer
        intputlayer.output_unit = self

        self.zin = 0
    
    def forward(self,X):
        self.zin = self.input_unit.forward(X)
        return self.zin

    def backprop(self,X,y):
        # TODO rework
        y_hat = self.forward(X)
        self.loss = self.loss_function(y,y_hat)
        return self.loss

    def loss_function(self,y_true,y_pred):
        return 0

class MSE(Loss):
    def loss_function(self,y_true,y_pred):
        return np.sum(np.square(y_true-y_pred))

class RMSE(Loss):
    def loss_function(self,y_true,y_pred):
        return np.srt(np.sum(np.square(y_true-y_pred)))

class MAE(Loss):
    def loss_function(self,y_true,y_pred):
        return np.sum(np.abs(y_true-y_pred))