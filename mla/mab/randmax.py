import numpy as np

def randmax(A):
    ''' A function that returns an argmax at random in case of multiple maximizers'''
    maxValue=max(A)
    index = [i for i in range(len(A)) if A[i]==maxValue]
    return np.random.choice(index)