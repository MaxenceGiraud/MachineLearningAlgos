import numpy as np

def accuracy_score(y_true,y_pred,weights=1):
    return np.where(y_pred==y_true,1,0).mean()