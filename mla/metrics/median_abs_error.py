import numpy as np

def median_absolute_error(y_true,y_pred):
    return np.median(np.abs(y_pred-y_true))