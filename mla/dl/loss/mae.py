from .loss import Loss
import numpy as np

class MAE(Loss):
    def loss_function(self,y_pred,y_true,weights=1):
        return np.mean(np.abs(y_pred.reshape(y_true.shape)-y_true)*weights.reshape(y_true.shape))

    def deriv(self,y_pred,y_true,weights=1):
        return np.sign((y_pred-y_true.reshape(y_pred.shape))*weights.reshape(y_pred.shape))