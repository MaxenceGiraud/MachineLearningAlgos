import numpy as np
from .mse import mean_squared_error

def root_mean_squared_error(y_true,y_pred,weights=1):
    return np.sqrt(mean_squared_error(y_true,y_pred,weights=weights))