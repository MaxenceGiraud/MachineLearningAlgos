import numpy as np

def r2_score(y_true,y_pred):
    y_bar = np.mean(y_true,axis=0)
    return 1 - np.sum((y_true-y_pred)**2) / np.sum((y_true-y_bar)**2)