from .loss import Loss
import numpy as np

class BinaryCrossEntropy(Loss):
    def loss_function(self,y_pred,y_true,weights=1):
        return np.mean((-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))*weights)
    
    def deriv(self,y_pred,y_true,weights=1):
        return (y_pred - y_true) * weights