import numpy as np

def accuracy_score(y_true,y_pred):
    return np.where(y_pred==y_true,1,0).mean()