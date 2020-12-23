import numpy as np

def mean_absolute_error(y_true,y_pred):
    return np.abs(y_pred-y_true).mean()