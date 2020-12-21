import numpy as np
from ..base import BaseRegressor,BaseClassifier
from ..kernels.rbf import RBF
import itertools


class GaussianProcessRegressor(BaseRegressor):
    def __init__(self,sigma_noise = 0.4,kernel = RBF(0.1,50)):
        self.kernel = kernel
        self.sigman = sigma_noise

        self.K = None

    def _compute_kernel(self,x,xs):    
        return np.array([self.kernel(x[i],xs[j]) for (i, j) in itertools.product(np.arange(x.shape[0]), np.arange(xs.shape[0]))]).reshape(x.shape[0],xs.shape[0])

    def _compute_params(self,x,xs,y,K=None):
        if K is None : 
            self.K = self._compute_kernel(x,x)
            self.K_inv = np.linalg.inv(self.K)

        Ks = self._compute_kernel(x,xs)
        Kss = self._compute_kernel(xs,xs)

        cov = Kss - Ks.T @ self.K_inv @ Ks
        mu = Ks.T @ self.K_inv @ y
        return mu,cov

    def fit(self,X,y):
        self.K = self._compute_kernel(X,X)
        self.K_inv = np.linalg.inv(self.K + self.sigman**2 * np.eye(X.shape[0]))

        self.X = X
        self.y = y 

    def predict(self,n_samples=1):
        self.mu,self.cov = self._compute_params(self.X,X,self.y,self.K)
        return np.random.multivariate_normal(self.mu,self.cov,size=n_samples)