import numpy as np

def chi_metric(x,y):
    return np.sum((x-y)**2/(x+y))

def intersection_measure(x,y):
    return np.sum(x==y) / (np.sum(x)+np.sum(y))