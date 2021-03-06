from .loss import Loss
import numpy as np

class BinaryCrossEntropy(Loss):
    def loss_function(self,y_pred,y_true,weights=1):
        return np.mean((-y_true * np.log(y_pred.reshape(y_true.shape)) - (1 - y_true) * np.log(1 - y_pred.reshape(y_true.shape)))*weights.reshape(y_pred.shape))
    
    def deriv(self,y_pred,y_true,weights=1):
        return (y_pred - y_true.reshape(y_pred.shape)) * weights.reshape(y_pred.shape)