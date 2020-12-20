import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    ''' Gaussian Mixture Model clustering algorithm
    Ref :

    Parameters
    ----------
    n_clusters : int,
        Number of cluster/gaussians to consider
    iter_max : int,
        maximum number of iterations of the algo
    eps : float,
        stopping criterion
    '''
    def __init__(self,n_cluster=3,iter_max=100,eps=1e-3):
        self.n_cluster = n_cluster
        self.iter_max = iter_max
        self.eps = eps


    def _init_params(self,X):
        self.mu = np.random.uniform(low=np.min(X,axis=0), high=np.max(X,axis=0), size=(self.n_cluster,X.shape[1]))  # init the means

        # Init Cov matrix
        self.sigma  =np.zeros((self.n_cluster,X.shape[1],X.shape[1]))
        for k in range(self.n_cluster):
            self.sigma[k] = np.cov(X.T)
            min_eig = np.min(np.real(np.linalg.eigvals(self.sigma[k]))) # Make cov PSD
            if min_eig < 0:
                self.sigma[k] -= 10*min_eig * np.eye(*self.sigma[k].shape)
        
        self.pi = np.ones(self.n_cluster)/X.shape[0] # mixing coef
        self.r = np.zeros((X.shape[0],self.n_cluster)) # responsability

    def _compute_resp(self,X):
        r = np.zeros((X.shape[0],self.n_cluster))
        for k in range(self.n_cluster):
            r[:,k] = self.pi[k] *  multivariate_normal.pdf(x=X,mean=self.mu[k],cov=self.sigma[k])
        r /= np.sum(r,axis=1).reshape(-1,1)
        return r

    def _expectation_step(self,X):
        self.r = self._compute_resp(X)

    def _maximization_step(self,X):
        self.N = np.sum(self.r,axis=0)
        for k in range(self.n_cluster):
            # Update Means
            self.mu[k] = (self.r[:,k] @ X) / self.N[k] 

            # Update Covariance Matrix
            self.sigma[k] = np.zeros((X.shape[1],X.shape[1]))
            xmu = X - self.mu[k]
            self.sigma[k] = self.r[:,k].reshape(1,-1) * xmu.T @ xmu
            self.sigma[k] = np.sqrt(np.abs(self.sigma[k]))/ self.N[k] + np.eye(X.shape[1]) * 1e-6
            
        self.pi = self.N / X.shape[0]

    def fit(self,X):
        
        self._init_params(X)

        iter = 0
        nll = np.inf
        old_nll = 0
        while iter < self.iter_max and np.abs(nll-old_nll) > self.eps:
            print(iter)

            # Expectation step           
            self._expectation_step(X)

            # Maximimzation step
            self._maximization_step(X)
            
            old_nll = nll
            nll = 0.0
            for k in range(self.n_cluster):
                nll += self.pi[k] * multivariate_normal.pdf(x=X,mean=self.mu[k],cov=self.sigma[k])
            nll = -np.sum(np.log(nll))
            print("Neg log likelihood =",nll)
            iter +=1