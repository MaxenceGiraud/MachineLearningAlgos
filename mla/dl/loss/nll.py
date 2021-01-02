from .loss import Loss
import numpy as np

class NLLloss(Loss):
    ''' Negative Log Likelihood 
    Takes as input probabilities for each class as the prediction and true classes as int for the true class
    '''
    def loss_function(self,y_pred,y_true,weights=1):
        l = np.sum(-np.log(y_pred[np.arange(y_true.shape[0]),y_true])*weights.reshape(-1,1))
        return l

    def deriv(self,y_pred,y_true,weights=1):
        d = np.zeros(y_pred.shape)
        d[np.arange(y_true.shape[0]),y_true] = -1/y_pred[np.arange(y_true.shape[0]),y_true]
        return d