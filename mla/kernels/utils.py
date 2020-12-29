import numpy as np

def chi_metric(x,y):
    return np.sum((x-y)**2/(x+y))