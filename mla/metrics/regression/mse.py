import numpy as np

def mean_squared_error(y_true,y_pred,weights=1):
    return np.mean(weights * (y_pred-y_true)**2)