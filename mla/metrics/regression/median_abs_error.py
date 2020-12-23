import numpy as np

def median_absolute_error(y_true,y_pred,weights=1):
    return np.median(weights * np.abs(y_pred-y_true))