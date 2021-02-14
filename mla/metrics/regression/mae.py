import numpy as np

def mean_absolute_error(y_true,y_pred,weights=1):
    return np.mean(weights *np.abs(y_pred-y_true))