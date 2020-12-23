import numpy as np

def zero_one_loss(y_true,y_pred):
    return np.where(y_true!=y_pred,1,0).mean() 