from .loss import Loss
import numpy as np

class MSE(Loss):
    def loss_function(self,y_pred,y_true,weights=1):
        return np.mean(weights.reshape(y_true.shape)*(y_pred.reshape(y_true.shape)- y_true)**2)
    
    def deriv(self,y_pred,y_true,weights=1):
        return 2*(y_pred - y_true.reshape(y_pred.shape)) *weights.reshape(y_pred.shape)