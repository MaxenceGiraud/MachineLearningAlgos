import numpy as np
from ..base import BaseRegressor,BaseClassifier
from ..kernels.rbf import RBF

class BaseSVM():
    def __init__(self,kernel=RBF(),gamma=1,C=1,max_iter = 20):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.lagrange_multipliers = None
        self.b = 0
        self._dual_coefs = None

        ## SMO param
        self.tol = 1e-4 
        self.max_iter = max_iter 

class SVC(BaseSVM,BaseClassifier):
    def _compute__L_dual(self):
        idx_kl = np.array([[i,j] for i in range(X.shape[0]) for j in range(X.shape[0])]).reshape(X.shape[0],X.shape[0],2)
        self.Q = self.K * y[idx_kl]
        return 2 * self.lagrange_multipliers.sum() - self.lagrange_multipliers.T @ self.Q @ self.lagrange_multipliers # maximize, subject to  0 <= alpha_i*y_i <= C 
    
    
    def _smo(self,X,y):
        ''' SMO Optmizer for SVM, is not guaranteed to reach the full solution
        ref : Fast Training of Support Vector Machines using Sequential Minimal Optimization John C. Platt, Microsoft Research
        Simplified version :<http://cs229.stanford.edu/materials/smo.pdf>
        '''
        for _ in range(self.max_iter) :
            num_changed_alphas= 0
            for i in range(X.shape[0]):
                Ei = self._decision_function(X[i],self.K[i]) - y[i]
                yEi = y[i] * Ei
                if ( (yEi < -self.tol) and  (self.lagrange_multipliers[i] < self.C) ) or ( (yEi >self.tol) and (self.lagrange_multipliers[i] > 0)): # Check KKT conditions
                    # j = np.random.choice(np.concatenate((np.arange(i),np.arange(i+1,X.shape[0])))) # Random index different from i or iterate on all j != i
                    for j in range(X.shape[0]):
                        if j == i:
                            continue
                        Ej = self._decision_function(X[j],self.K[j]) - y[j]
                        old_lagrange_multipliers = np.copy(self.lagrange_multipliers) # Save lagrange multipliers values

                        # Compute bounds of Lagrange multipliers
                        if y[i] == y[j]:
                            L = max(0,self.lagrange_multipliers[j]+self.lagrange_multipliers[i]-self.C)
                            H = min(self.C,self.lagrange_multipliers[j]+self.lagrange_multipliers[i])
                        else : 
                            L = max(0,self.lagrange_multipliers[j]-self.lagrange_multipliers[i])
                            H = min(self.C,self.C+self.lagrange_multipliers[j]-self.lagrange_multipliers[i])

                        if L == H :
                            continue
                        eta = 2 * self.K[j,i] - self.K[i,i] - self.K[j,j]
                        # if eta >= 0 : # ??????
                        #     continue
                        self.lagrange_multipliers[j] = self.lagrange_multipliers[j] - y[j] *(Ei -Ej)/eta # 
                        if self.lagrange_multipliers[j] > H :
                            self.lagrange_multipliers[j] = H
                        elif self.lagrange_multipliers[j] < L :
                            self.lagrange_multipliers[j] = L
                        
                        if abs(self.lagrange_multipliers[j] - old_lagrange_multipliers[j]) < 1e-5 :  # If no change then  skip
                            continue

                        self.lagrange_multipliers[i] = self.lagrange_multipliers[i] - y[i]*y[j] *(self.lagrange_multipliers[j] - old_lagrange_multipliers[j]) 

                        b1 = self.b -Ei - y[i]* (self.lagrange_multipliers[i]-old_lagrange_multipliers[i])*self.K[i,i] - y[j]* (self.lagrange_multipliers[j]-old_lagrange_multipliers[j])*self.K[j,i]
                        b2 = self.b -Ej - y[i]* (self.lagrange_multipliers[i]-old_lagrange_multipliers[i])*self.K[i,j] - y[j]* (self.lagrange_multipliers[j]-old_lagrange_multipliers[j])*self.K[j,j]

                        if self.lagrange_multipliers[i] > 0 and self.lagrange_multipliers[i] < self.C :
                            self.b = b1
                        elif self.lagrange_multipliers[j] > 0 and self.lagrange_multipliers[j] < self.C :
                            self.b = b2
                        else :
                            self.b = (b1+b2) /2
                        
                        num_changed_alphas += 1
            if num_changed_alphas == 0 :
                break


    def fit(self,X,y):
        self.labels = np.unique(y)
        assert len(self.labels) == 2, "Only 2 class can be given to this classifier"

        # Renamed labels as 1 and -1
        y[y == self.labels[1]] = 1
        y[y == self.labels[0]] = -1

        ## Init
        self.lagrange_multipliers = np.zeros(X.shape[0])
        self.b = 0

        # Tmp Values
        self.supports = X
        self.supports_label = y

        # Precompute Full Kernel
        self.K = self.kernel(X,X) 
        

        self._smo(X,y) # Optimize

        # Keep points where alpga are non zero (i.e. margin points / support vectors)
        support_idx = np.where(self.lagrange_multipliers!=0)[0]
        self.supports = X[support_idx]
        self.supports_label = y[support_idx]
        self.lagrange_multipliers = self.lagrange_multipliers[support_idx]
    
    def _decision_function(self,X,K=None):
        if K is None :
            f= (self.kernel(X,self.supports) * self.lagrange_multipliers) @ self.supports_label + self.b 
        else : 
            f = (K* self.lagrange_multipliers) @ self.supports_label + self.b 
        return f

    def predict_probs(self,X):
        # TODO remake this properly (see Murphy's book)
       pass

    def predict(self,X):
        d = self._decision_function(X)
        return np.where(np.sign(d)==1,self.labels[1],self.labels[0])


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