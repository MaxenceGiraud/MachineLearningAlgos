#%%
# 
import numpy as np
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
        # self.sigma = np.random.uniform(low=np.std(X,axis=0)/2, high=np.std(X,axis=0), size=(self.n_cluster,X.shape[1],X.shape[1]))  # init the stds
        self.sigma  =np.zeros((self.n_cluster,X.shape[1],X.shape[1]))
        for k in range(self.n_cluster):
            self.sigma[k] = np.cov(X.T)
            min_eig = np.min(np.real(np.linalg.eigvals(self.sigma[k])))
            if min_eig < 0:
                self.sigma[k] -= 10*min_eig * np.eye(*self.sigma[k].shape)

        self.pi = np.repeat(self.n_cluster/X.shape[0],self.n_cluster) # mixing coef
        self.r = np.zeros((X.shape[0],self.n_cluster)) # responsability

    def _expectation_step(self,X):
        for k in range(self.n_cluster):
            self.r[:,k] = self.pi[k] *  multivariate_normal.pdf(x=X,mean=self.mu[k],cov=self.sigma[k])
            self.r[:,k] /= np.sum(self.r[:,k])

    def _maximization_step(self,X):
        self.N = np.sum(self.r,axis=0)
        for k in range(self.n_cluster):
            self.mu[k] = (self.r[:,k] @ X) / self.N[k] # update mean
            self.sigma[k] = np.zeros((X.shape[1],X.shape[1]))
            for j in range(X.shape[0]):
                xmu = X[j] - self.mu[k]
                self.sigma[k] +=  self.r[j,k] * np.outer(xmu,xmu) # update std
            self.sigma[k] /= self.N[k]
        
        self.pi = self.N / X.shape[0]

    def fit(self,X):
       
        self._init_params(X)

        iter = 0
        nll = 10
        while iter < self.iter_max and nll > self.eps:# add escape condition
            print(iter)

            # Expectation step           
            self._expectation_step(X)

            # Maximimzation step
            self._maximization_step(X)

            nll = 0.0
            for k in range(self.n_cluster):
                nll += self.pi[k] * multivariate_normal.pdf(x=X,mean=self.mu[k],cov=self.sigma[k])
            nll = -np.sum(np.log(nll))
            print("Neg log likelihood =",nll)
            iter +=1

    def predict(self,X):
        return

    def display(self):
        # TODO display Gaussians 
        pass
#%%