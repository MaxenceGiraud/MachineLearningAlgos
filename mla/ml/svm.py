import numpy as np
from ..base import BaseRegressor,BaseClassifier
from ..kernels.rbf import RBF

class BaseSVM():
    def __init__(self,kernel=RBF(),gamma=1,,C=1):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.lagrange_multipliers = None
        self.b = 0
        self._dual_coefs = None
        
class SVC(BaseSVM,BaseClassifier):
    def fit(self,X,y):
        self.labels = np.unique(y)
        assert len(self.labels) == 2, "Only 2 class can be given to this classifier"

        # Renamed labels as 1 and -1
        y[y == self.labels[0]] = 1
        y[y == self.labels[1]] = -1

        self.lagrange_multipliers = np.zeros(X.shape[0])
        K = self.kernel(X,X) 
        idx_kl = np.array([[i,j] for i in range(X.shape[0]) for k in range(X.shape[0])]).reshape(X.shape[0],X.shape[0],2)
        Q = K * y[idx_kl]


        L_primal = 2 * self.lagrange_multipliers.sum() - self.lagrange_multipliers.T @ Q @ self.lagrange_multipliers # maximize, subject to  0 <= alpha_i*y_i <= C

        # Compute bias ??

        ## Keep points where alpga are non zero (i.e. margin points / support vectors)
        support_idx = np.where(self.lagrange_multipliers>1e-6,1,0)
        self.supports = X[support_idx]
        self.supports_label = y[support_idx]
        self._dual_coefs = self.lagrange_multipliers[support_idx]
    
    def _decision_function(self,X):
        return (self.kernel(self.supports,X) * self._dual_coefs) @ self.supports_label + self.b 

    def predict_probs(self,X):
        d = self._decision_function(X)
        probs = (d + min(d))/(max(d)+min(d)) # Put values between 0 and 1
        return probs

    def predict(self,X):
        d = self._decision_function(X)
        return np.where(np.sign(d)==1,self.labels[0],self.labels[1])


class SVR(BaseSVM,BaseRegressor):
    def __init__(self,kernel=RBF(),gamma=1,C=1,eps=0.1,):
        self.lagrange_multipliers_star = None 
        self.eps = eps
        super().__init__(kernel=kernel,gamma=gamma,C=C)


    def fit(self,X,y):
        self.lagrange_multipliers = np.zeros(X.shape[0])
        K = self.kernel(X,X) 
        
        a_a = (self.lagrange_multipliers - self.lagrange_multipliers_star)
        L_dual = 0.5 * a_a.T @ K @ a_a + self.eps * np.ones(X.shape[0]) @  (self.lagrange_multipliers + self.lagrange_multipliers_star) - y.T @ a_a #minimize,  subject to sum(alpha - alpha_star) = 0 and all 0<=alpha,alpha_*<=C


        ## Keep points where coefs are non zero (i.e. margin points / support vectors)
        support_idx = np.where(self.lagrange_multipliers-self.lagrange_multipliers_star>1e-6,1,0)
        self.supports = X[support_idx]
        self.supports_label = y[support_idx]
        self._dual_coefs = self.lagrange_multipliers[support_idx] - self.lagrange_multipliers_star[support_idx]

    def predict(self,X):
        return self._dual_coefs @ self.kernel(self.supports,X) + self.b