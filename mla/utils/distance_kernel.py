import numpy as np

def distance_from_kernel(kernel,x,y):
    '''Compute a distance between 2 sets of datapoints using a kernel'''

    if hasattr(kernel, 'to_precompute') and ('distance' in kernel.to_precompute or 'distance_manhattan' in kernel.to_precompute):
        ## Prevent useless computations  when k(x,x) gives 1
        kxx = 1
        kxtxt = 1
    else :
        kxx = np.diag(kernel(x,x))
        kxtxt = np.diag(kernel(y,y))
    dist =  kxx.reshape(1,-1) -2 * kernel(x,y) + kxtxt.reshape(-1,1)

    return dist