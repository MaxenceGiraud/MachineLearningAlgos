import multiprocessing as mp
from functools import partial
import numpy as np
from ..base import BaseClassifier,BaseRegressor
from ..ml.decison_tree import DecisionTreeClassifier,DecisionTreeRegressor

base_learner_reg = partial(DecisionTreeRegressor,3)
base_learner_clf = partial(DecisionTreeClassifier,3)

class BaseGradientBoosting:
    def __init__(self,base_model,basemodel_params,n_estimators=100,learning_rate=0.1,loss='l2',iter_max=100,subsample=0.9):
        self.base_model = base_model
        self.basemodel_params = basemodel_params
        self.n_estimators = n_estimators
        self.iter_max = iter_max
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.estimators = []

        assert learning_rate > 0 and learning_rate <= 1, "Learning rate must be between 0 and 1"

    def fit_one_learner(self,model,X,y):

        # Init prediction
        y_hat_train = np.mean(y) * np.ones(y.shape)

        residual = y - y_hat_train # Compute residual

        for i in range(self.iter_max):
            model.fit(X,residual)

            y_hat_train = self.learning_rate * model.predict(X)

            residual = y - y_hat_train # Update residual
        
        return model
            

    def fit(self,X,y):
        
        # Create base estimators
        self.estimators = []
        [self.estimators.append(self.base_model(**self.basemodel_params)) for _ in range(self.n_estimators)]

        # Fit the estimators
        for i in range(self.n_estimators):
            # Draw bootstrap samples
            n_sample = int(X.shape[0] * self.subsample)
            samples_idx = np.random.choice(X.shape[0],size=self.n_estimators*n_sample ,replace=True).reshape(self.n_estimators,n_sample)

            self.estimators[i] = self.fit_one_learner(self.estimators[i],X[samples_idx],y[samples_idx])
           

    def predict_learners(self,X):
        return [self.estimators[i].predict(X) for i in range(self.n_estimators)]

class GradientBoostingClassifier(BaseGradientBoosting,BaseClassifier):
    ''' Gradient Boosting Classifer 
    Ref :  Friedman, Jerome H. Greedy function approximation: A gradient boosting machine. Ann. Statist. 29 (2001), no. 5, 1189--1232. doi:10.1214/aos/1013203451. https://projecteuclid.org/euclid.aos/1013203451

    Parameters
    ----------
    base_model : Classifer class,
        Base estimator, best to use with weak learners. Default to a Decision Tree with max_depth = 3.
    basemodel_params : dict,
        Parameters of the base estimators,
    n_estimators : int,
        Number of base estimators
    '''
    def __init__(self,base_model = base_learner_clf,basemodel_params = {}, n_estimators = 20):
        super().__init__(base_model=base_model,basemodel_params=basemodel_params,n_estimators=n_estimators)

    def predict(self,X):
        res = super().predict_learners(X)

        # Take the most common value
        res = np.array(res)
        decision = []
        for col in range(res.shape[1]):
            values,counts = np.unique(res[:,col],return_counts=True)
            decision.append(values[counts.argmax()])
        return decision


class GradientBoostingRegressor(BaseGradientBoosting,BaseRegressor):
    ''' Gradient Boosting Regressor 
    Ref : Friedman, Jerome H. Greedy function approximation: A gradient boosting machine. Ann. Statist. 29 (2001), no. 5, 1189--1232. doi:10.1214/aos/1013203451. https://projecteuclid.org/euclid.aos/1013203451

    Parameters
    ----------
    base_model : Regressor class,
        Base estimator, best to use with weak learners. Default to a Decision Tree with max_depth = 3.
    basemodel_params : dict,
        Parameters of the base estimators,
    n_estimators : int,
        Number of base estimators
    '''
    def __init__(self,base_model = base_learner_reg,basemodel_params = {}, n_estimators = 20):
        super().__init__(base_model=base_model,basemodel_params=basemodel_params,n_estimators=n_estimators)
    
    def predict(self,X):
        res = self.predict_learners(X)

        return np.mean(res,axis=0)